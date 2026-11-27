#include "update/crypto.h"

#include <openssl/evp.h>
#include <sys/random.h>

#include <algorithm>
#include <stdexcept>

namespace tona::update {

struct Sha256::Impl {
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    ~Impl() { EVP_MD_CTX_free(ctx); }
};

Sha256::Sha256() : impl_(std::make_unique<Impl>()) {
    if (!impl_->ctx || EVP_DigestInit_ex(impl_->ctx, EVP_sha256(), nullptr) != 1)
        throw std::runtime_error("SHA-256 initialization failed");
}

Sha256::~Sha256() = default;

void Sha256::update(const void* data, size_t n) {
    if (EVP_DigestUpdate(impl_->ctx, data, n) != 1) throw std::runtime_error("SHA-256 update failed");
}

Sha256Digest Sha256::finish() {
    Sha256Digest out{};
    unsigned len = 0;
    if (EVP_DigestFinal_ex(impl_->ctx, out.data(), &len) != 1 || len != out.size())
        throw std::runtime_error("SHA-256 finish failed");
    return out;
}

std::array<uint8_t, 64> blake2b512(std::string_view data) {
    std::array<uint8_t, 64> out{};
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    unsigned len = 0;
    const bool ok = ctx && EVP_DigestInit_ex(ctx, EVP_blake2b512(), nullptr) == 1 &&
                    EVP_DigestUpdate(ctx, data.data(), data.size()) == 1 &&
                    EVP_DigestFinal_ex(ctx, out.data(), &len) == 1 && len == out.size();
    EVP_MD_CTX_free(ctx);
    if (!ok) throw std::runtime_error("BLAKE2b-512 failed");
    return out;
}

bool ed25519_verify(const std::array<uint8_t, 32>& pub, std::string_view message, const std::array<uint8_t, 64>& sig) {
    EVP_PKEY* key = EVP_PKEY_new_raw_public_key(EVP_PKEY_ED25519, nullptr, pub.data(), pub.size());
    if (!key) return false;
    EVP_MD_CTX* ctx = EVP_MD_CTX_new();
    const bool ok = ctx && EVP_DigestVerifyInit(ctx, nullptr, nullptr, nullptr, key) == 1 &&
                    EVP_DigestVerify(ctx, sig.data(), sig.size(), reinterpret_cast<const uint8_t*>(message.data()),
                                     message.size()) == 1;
    EVP_MD_CTX_free(ctx);
    EVP_PKEY_free(key);
    return ok;
}

namespace {

constexpr std::string_view kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

int b64_value(char c) {
    const size_t pos = kB64.find(c);
    return pos == std::string_view::npos ? -1 : static_cast<int>(pos);
}

// One group of four characters; `pad` is the number of '=' at its end (0, 1 or 2)
bool decode_group(std::string_view g, int pad, Bytes& out) {
    uint32_t acc = 0;
    for (int i = 0; i < 4 - pad; i++) {
        const int v = b64_value(g[i]);
        if (v < 0) return false;
        acc = (acc << 6) | static_cast<uint32_t>(v);
    }
    acc <<= 6 * pad;
    // Non-canonical encodings (nonzero bits below the data) would give two texts for one value
    if (pad == 1 && (acc & 0xFF) != 0) return false;
    if (pad == 2 && (acc & 0xFFFF) != 0) return false;
    out.push_back(static_cast<uint8_t>(acc >> 16));
    if (pad < 2) out.push_back(static_cast<uint8_t>(acc >> 8));
    if (pad < 1) out.push_back(static_cast<uint8_t>(acc));
    return true;
}

}  // namespace

std::optional<Bytes> base64_decode_strict(std::string_view text) {
    if (text.empty() || text.size() % 4 != 0) return std::nullopt;
    Bytes out;
    out.reserve(text.size() / 4 * 3);
    for (size_t i = 0; i < text.size(); i += 4) {
        const std::string_view g = text.substr(i, 4);
        const bool last = i + 4 == text.size();
        int pad = 0;
        if (g[3] == '=') pad = g[2] == '=' ? 2 : 1;
        // '=' only at the end of the whole text
        if (pad > 0 && !last) return std::nullopt;
        if (!decode_group(g, pad, out)) return std::nullopt;
    }
    return out;
}

std::string to_hex(const uint8_t* data, size_t n) {
    static constexpr char kDigits[] = "0123456789abcdef";
    std::string s;
    s.reserve(n * 2);
    for (size_t i = 0; i < n; i++) {
        s += kDigits[data[i] >> 4];
        s += kDigits[data[i] & 15];
    }
    return s;
}

bool is_lower_hex(std::string_view s, size_t len) {
    if (s.size() != len) return false;
    for (const char c : s)
        if ((c < '0' || c > '9') && (c < 'a' || c > 'f')) return false;
    return true;
}

void random_bytes(uint8_t* out, size_t n) {
#if defined(__APPLE__)
    // getentropy reads the kernel CSPRNG like getrandom(flags 0), at most 256 bytes per call (getentropy(2))
    for (size_t done = 0; done < n;) {
        const size_t chunk = std::min<size_t>(n - done, 256);
        if (getentropy(out + done, chunk) != 0) throw std::runtime_error("getentropy failed");
        done += chunk;
    }
#else
    size_t done = 0;
    while (done < n) {
        const ssize_t r = getrandom(out + done, n - done, 0);
        if (r <= 0) throw std::runtime_error("getrandom failed");
        done += static_cast<size_t>(r);
    }
#endif
}

}  // namespace tona::update
