#include "update/keys.h"

namespace tona::update {

std::vector<TrustedKey> release_keys() {
    std::vector<TrustedKey> keys;
    for (const std::string_view line : {kMainPublicKey, kSparePublicKey}) {
        if (line.empty()) continue;
        // Skipping is safe (fewer keys trust less); test_update_minisig fails in CI if a non-empty constant does not
        // parse
        if (auto key = parse_public_key(line)) keys.push_back(*key);
    }
    return keys;
}

}  // namespace tona::update
