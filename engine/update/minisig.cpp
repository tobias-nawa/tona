#include "update/minisig.h"

#include <algorithm>

#include "update/crypto.h"

namespace tona::update {

namespace {

constexpr std::string_view kUntrustedPrefix = "untrusted comment: ";
constexpr std::string_view kTrustedPrefix = "trusted comment: ";
constexpr size_t kSigLineBytes = 74;  // "ED" + key id (8) + signature (64)
constexpr size_t kGlobalSigBytes = 64;
constexpr size_t kPublicKeyBytes = 42;  // "Ed" + key id (8) + key (32)

struct ParsedSignature {
    std::array<uint8_t, 8> key_id{};
    std::array<uint8_t, 64> sig{};
    std::string trusted_comment;
    std::array<uint8_t, 64> global_sig{};
};

SignatureCheck fail(std::string why) {
    SignatureCheck r;
    r.error = std::move(why);
    return r;
}

bool printable_line(std::string_view line) {
    return std::none_of(line.begin(), line.end(), [](char c) {
        const auto u = static_cast<unsigned char>(c);
        return u < 0x20 || u == 0x7f;
    });
}

// Four lines separated by '\n'; one final '\n' is allowed. Anything else (CR, NUL, extra lines) is rejected.
std::optional<std::array<std::string_view, 4>> split_lines(std::string_view text) {
    if (text.ends_with('\n')) text.remove_suffix(1);
    std::array<std::string_view, 4> lines;
    for (size_t i = 0; i < lines.size(); i++) {
        const size_t nl = text.find('\n');
        // A separator after the fourth line means a fifth (or an empty) line
        if ((nl == std::string_view::npos) != (i == lines.size() - 1)) return std::nullopt;
        lines[i] = text.substr(0, nl);
        if (!printable_line(lines[i])) return std::nullopt;
        if (nl != std::string_view::npos) text.remove_prefix(nl + 1);
    }
    return lines;
}

std::optional<Bytes> decode_exact(std::string_view b64, size_t bytes) {
    auto raw = base64_decode_strict(b64);
    if (!raw || raw->size() != bytes) return std::nullopt;
    return raw;
}

// Returns an error text, empty on success
std::string parse_signature(std::string_view text, ParsedSignature& out) {
    if (text.size() > kMaxSignatureBytes) return "signature file too large";
    const auto lines = split_lines(text);
    if (!lines) return "signature file is not exactly four lines";
    if (!(*lines)[0].starts_with(kUntrustedPrefix)) return "missing untrusted comment line";
    if (!(*lines)[2].starts_with(kTrustedPrefix)) return "missing trusted comment line";
    const auto head = decode_exact((*lines)[1], kSigLineBytes);
    if (!head) return "malformed signature line";
    if ((*head)[0] != 'E' || (*head)[1] != 'D') return "signature is not in the prehashed format (ED)";
    const auto global = decode_exact((*lines)[3], kGlobalSigBytes);
    if (!global) return "malformed global signature line";
    std::copy_n(head->begin() + 2, 8, out.key_id.begin());
    std::copy_n(head->begin() + 10, 64, out.sig.begin());
    std::copy(global->begin(), global->end(), out.global_sig.begin());
    out.trusted_comment = std::string((*lines)[2].substr(kTrustedPrefix.size()));
    return "";
}

const TrustedKey* select_key(const std::vector<TrustedKey>& keys, const std::array<uint8_t, 8>& id, bool& ambiguous) {
    const TrustedKey* found = nullptr;
    for (const TrustedKey& k : keys) {
        if (k.id != id) continue;
        ambiguous = ambiguous || found != nullptr;
        found = &k;
    }
    return found;
}

}  // namespace

std::optional<TrustedKey> parse_public_key(std::string_view base64_line) {
    const auto raw = decode_exact(base64_line, kPublicKeyBytes);
    if (!raw || (*raw)[0] != 'E' || (*raw)[1] != 'd') return std::nullopt;
    TrustedKey k;
    std::copy_n(raw->begin() + 2, 8, k.id.begin());
    std::copy_n(raw->begin() + 10, 32, k.pub.begin());
    return k;
}

SignatureCheck verify_minisig(std::string_view file, std::string_view signature_text,
                              const std::vector<TrustedKey>& keys) {
    ParsedSignature sig;
    if (std::string why = parse_signature(signature_text, sig); !why.empty()) return fail(std::move(why));
    bool ambiguous = false;
    const TrustedKey* key = select_key(keys, sig.key_id, ambiguous);
    if (ambiguous) return fail("two trusted keys share one key id");
    if (!key) return fail("signature was made with an unknown key");
    const auto digest = blake2b512(file);
    const std::string_view hashed(reinterpret_cast<const char*>(digest.data()), digest.size());
    if (!ed25519_verify(key->pub, hashed, sig.sig)) return fail("file signature is invalid");
    std::string global_message(reinterpret_cast<const char*>(sig.sig.data()), sig.sig.size());
    global_message += sig.trusted_comment;
    if (!ed25519_verify(key->pub, global_message, sig.global_sig)) return fail("global signature is invalid");
    SignatureCheck ok;
    ok.ok = true;
    ok.trusted_comment = std::move(sig.trusted_comment);
    return ok;
}

}  // namespace tona::update
