// Signature check of the updater: the vectors under data/update (made by generate.py and by the real minisign), and
// every way a signature file or a signed file can be wrong.
#include <string>

#include "tests/test_support.h"
#include "tests/update_testkit.h"
#include "update/keys.h"
#include "update/minisig.h"

using namespace tona::test;
namespace {

std::string g_dir;

std::string vec(const std::string& name) { return read_file(g_dir + "/" + name); }

// Second line of a minisign public key file
std::string key_line(const std::string& file) {
    const size_t a = file.find('\n') + 1;
    return file.substr(a, file.find('\n', a) - a);
}

up::TrustedKey key_of(const std::string& pub_file) {
    const auto k = up::parse_public_key(key_line(vec(pub_file)));
    EXPECT(k.has_value());
    return k.value_or(up::TrustedKey{});
}

// Contents of a base64 text that must decode
up::Bytes decoded(const std::string& text) {
    auto b = up::base64_decode_strict(text);
    EXPECT(b.has_value());
    return b.value_or(up::Bytes{});
}

up::SignatureCheck check(const std::string& sig_name, const std::string& manifest = "") {
    return up::verify_minisig(manifest.empty() ? vec("manifest.json") : manifest, vec(sig_name), {key_of("key.pub")});
}

void expect_rejected(const std::string& sig_name, const std::string& why) {
    const auto r = check(sig_name);
    EXPECT(!r.ok);
    EXPECT(r.error.contains(why));
    EXPECT(r.trusted_comment.empty());
}

void test_valid() {
    const auto r = check("valid.minisig");
    EXPECT(r.ok);
    EXPECT(r.trusted_comment == "tona schema=1 seq=3 version=0.3.0");
}

void test_cpp_signing_matches_python_vector() {
    // The C++ test signer and the Python generator are independent; Ed25519 is deterministic, so bytes must agree
    const std::string ours = main_key().sign_file(vec("manifest.json"), comment_for(3, "0.3.0"));
    EXPECT(ours == vec("valid.minisig"));
}

void test_real_minisign_vector() {
    const auto parsed = up::parse_public_key(key_line(read_file(g_dir + "/real/real.pub")));
    EXPECT(parsed.has_value());
    const up::TrustedKey key = parsed.value_or(up::TrustedKey{});
    const std::string manifest = read_file(g_dir + "/real/manifest.json");
    const auto r = up::verify_minisig(manifest, read_file(g_dir + "/real/manifest.json.minisig"), {key});
    EXPECT(r.ok);
    EXPECT(r.trusted_comment == "tona schema=1 seq=3 version=0.3.0");
    // One flipped byte in the signed file
    std::string bad = manifest;
    bad[10] ^= 1;
    EXPECT(!up::verify_minisig(bad, read_file(g_dir + "/real/manifest.json.minisig"), {key}).ok);
}

void test_rejections() {
    expect_rejected("wrong_key.minisig", "file signature is invalid");
    expect_rejected("unknown_id.minisig", "unknown key");
    expect_rejected("tampered_comment.minisig", "global signature is invalid");
    expect_rejected("legacy_ed.minisig", "prehashed");
    expect_rejected("truncated.minisig", "malformed");
    expect_rejected("extra_line.minisig", "four lines");
    expect_rejected("blank_line.minisig", "four lines");
    expect_rejected("crlf.minisig", "four lines");
    expect_rejected("too_long.minisig", "too large");
}

void test_comment_mismatch_is_signed_but_not_equal() {
    // Valid signature; the manifest layer (not this one) rejects the mismatch, but the comment must be the signed one
    const auto r = check("comment_mismatch.minisig");
    EXPECT(r.ok);
    EXPECT(r.trusted_comment == "tona schema=1 seq=4 version=0.3.0");
}

void test_changed_file() {
    std::string m = vec("manifest.json");
    for (const size_t pos : {size_t(0), m.size() / 2, m.size() - 1}) {
        std::string bad = m;
        bad[pos] ^= 1;
        EXPECT(!check("valid.minisig", bad).ok);
    }
    EXPECT(!check("valid.minisig", m + " ").ok);
}

void test_key_selection() {
    const std::string sig = vec("valid.minisig"), m = vec("manifest.json");
    const up::TrustedKey main = key_of("key.pub"), other = key_of("other.pub");
    EXPECT(up::verify_minisig(m, sig, {other, main}).ok);
    EXPECT(!up::verify_minisig(m, sig, {other}).ok);
    EXPECT(!up::verify_minisig(m, sig, {}).ok);
    // Same id, different key: configuration error, not "try the next one"
    up::TrustedKey twin = other;
    twin.id = main.id;
    const auto r = up::verify_minisig(m, sig, {main, twin});
    EXPECT(!r.ok);
    EXPECT(r.error.contains("share one key id"));
}

void test_signature_text_mutations() {
    const std::string m = vec("manifest.json"), good = vec("valid.minisig");
    const up::TrustedKey key = key_of("key.pub");
    EXPECT(up::verify_minisig(m, good.substr(0, good.size() - 1), {key}).ok);  // final newline is optional
    EXPECT(!up::verify_minisig(m, "", {key}).ok);
    std::string nul = good;
    nul[30] = '\0';
    EXPECT(!up::verify_minisig(m, nul, {key}).ok);
    std::string prefix = good;
    prefix.replace(0, 18, "untrusted commen: ");
    EXPECT(!up::verify_minisig(m, prefix, {key}).ok);
    std::string trusted = good;
    trusted.replace(trusted.find("trusted comment: "), 17, "trusted Comment:  ");
    EXPECT(!up::verify_minisig(m, trusted, {key}).ok);
    std::string pad = good;
    pad.insert(good.find('\n') + 1, " ");
    EXPECT(!up::verify_minisig(m, pad, {key}).ok);
}

void test_base64_strict() {
    EXPECT(decoded("TWFu") == up::Bytes({'M', 'a', 'n'}));
    EXPECT(decoded("TWE=") == up::Bytes({'M', 'a'}));
    EXPECT(decoded("TQ==") == up::Bytes({'M'}));
    EXPECT(!up::base64_decode_strict("").has_value());
    EXPECT(!up::base64_decode_strict("TWE").has_value());    // padding missing
    EXPECT(!up::base64_decode_strict("TWF u").has_value());  // whitespace
    EXPECT(!up::base64_decode_strict("TWFu\n").has_value());
    EXPECT(!up::base64_decode_strict("TR==").has_value());      // non-canonical trailing bits
    EXPECT(!up::base64_decode_strict("TWF=").has_value());      // non-canonical trailing bits
    EXPECT(!up::base64_decode_strict("TQ==TWFu").has_value());  // padding inside
    EXPECT(!up::base64_decode_strict("T===").has_value());
    EXPECT(!up::base64_decode_strict("TW-u").has_value());  // URL-safe alphabet
}

void test_public_key_parsing() {
    const std::string line = key_line(vec("key.pub"));
    EXPECT(up::parse_public_key(line).has_value());
    EXPECT(!up::parse_public_key(line.substr(0, line.size() - 4)).has_value());
    EXPECT(!up::parse_public_key("").has_value());
    // A signature line is not a key: wrong algorithm bytes and length
    EXPECT(!up::parse_public_key(vec("valid.minisig").substr(vec("valid.minisig").find('\n') + 1, 100)).has_value());
}

void test_release_build_trusts_nothing_yet() {
    // The constants are placeholders until the maintainer generates the real keys; an empty set accepts no release
    EXPECT(up::release_keys().empty() == (up::kMainPublicKey.empty() && up::kSparePublicKey.empty()));
    EXPECT(!up::verify_minisig(vec("manifest.json"), vec("valid.minisig"), up::release_keys()).ok);
}

// Fails in CI when a key constant in keys.h is set but malformed, so a broken key can never ship silently
void test_compiled_in_keys_are_well_formed() {
    std::vector<up::TrustedKey> parsed;
    for (const std::string_view line : {up::kMainPublicKey, up::kSparePublicKey}) {
        if (line.empty()) continue;
        const auto k = up::parse_public_key(line);
        EXPECT(k.has_value());
        if (k) parsed.push_back(*k);
    }
    EXPECT(up::release_keys().size() == parsed.size());
    if (parsed.size() == 2) EXPECT(parsed[0].id != parsed[1].id && parsed[0].pub != parsed[1].pub);
    EXPECT(up::kMainPublicKey.empty() || up::kMainPublicKey != up::kSparePublicKey);
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::guard("update_minisig", [&] {
        if (argc < 2) return 2;
        g_dir = argv[1];
        return run("update_minisig", [] {
            test_valid();
            test_cpp_signing_matches_python_vector();
            test_real_minisign_vector();
            test_rejections();
            test_comment_mismatch_is_signed_but_not_equal();
            test_changed_file();
            test_key_selection();
            test_signature_text_mutations();
            test_base64_strict();
            test_public_key_parsing();
            test_release_build_trusts_nothing_yet();
            test_compiled_in_keys_are_well_formed();
        });
    });
}
