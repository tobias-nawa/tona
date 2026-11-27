// Thin wrappers over OpenSSL 3 (EVP) for the updater: SHA-256, BLAKE2b-512, Ed25519 verification, strict Base64 and
// hex. OpenSSL is already needed for HTTPS, so no second crypto library comes with the updater.
#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tona::update {

using Bytes = std::vector<uint8_t>;
using Sha256Digest = std::array<uint8_t, 32>;

// Incremental SHA-256
class Sha256 {
public:
    Sha256();
    ~Sha256();
    Sha256(const Sha256&) = delete;
    Sha256& operator=(const Sha256&) = delete;
    void update(const void* data, size_t n);
    // Ends the hash; the object is not usable afterwards
    Sha256Digest finish();

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

std::array<uint8_t, 64> blake2b512(std::string_view data);

// True only for a valid signature. A key or signature of the wrong length is simply false.
bool ed25519_verify(const std::array<uint8_t, 32>& pub, std::string_view message, const std::array<uint8_t, 64>& sig);

// Standard alphabet, mandatory padding, no whitespace, canonical (unused trailing bits are zero). nullopt otherwise.
std::optional<Bytes> base64_decode_strict(std::string_view text);

std::string to_hex(const uint8_t* data, size_t n);
// Exactly `len` characters of [0-9a-f]
bool is_lower_hex(std::string_view s, size_t len);

// Cryptographically random bytes from the kernel; throws if it fails
void random_bytes(uint8_t* out, size_t n);

}  // namespace tona::update
