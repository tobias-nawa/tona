// Verification of minisign signatures (https://jedisct1.github.io/minisign/), prehashed format only.
// Strict on purpose: exactly four lines, exact lengths, algorithm `ED` (BLAKE2b-512 of the file) and nothing else.
#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace tona::update {

constexpr size_t kMaxSignatureBytes = 4096;

struct TrustedKey {
    std::array<uint8_t, 8> id{};
    std::array<uint8_t, 32> pub{};
};

// The base64 line of a minisign public key file (`RWQ...`: "Ed" + key id + key). nullopt for anything else.
std::optional<TrustedKey> parse_public_key(std::string_view base64_line);

struct SignatureCheck {
    bool ok = false;
    std::string error;            // set when !ok
    std::string trusted_comment;  // covered by the global signature; only valid when ok
};

// Checks `signature_text` (the content of a .minisig file) over `file` against the key whose id the signature names.
// The key is chosen by id, never by trying all of them. The trusted comment is returned only after both the file
// signature and the global signature are valid.
SignatureCheck verify_minisig(std::string_view file, std::string_view signature_text,
                              const std::vector<TrustedKey>& keys);

}  // namespace tona::update
