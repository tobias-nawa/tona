// FNV-1a 64 over bytes: the hash of the prompt-cache key. Same function as server::fnv, so a fingerprint that is
// built from model parts stays equal to the one built before the split (the disk caches of users stay valid).
#pragma once

#include <cstddef>
#include <cstdint>

namespace tona {

class Hasher {
public:
    static constexpr uint64_t BASIS = 0xcbf29ce484222325ull;

    explicit Hasher(uint64_t state = BASIS) : h_(state) {}

    void bytes(const void* p, size_t n) {
        const auto* b = static_cast<const uint8_t*>(p);
        for (size_t i = 0; i < n; i++) h_ = (h_ ^ b[i]) * 0x100000001b3ull;
    }
    template <class V>
    void value(const V& v) {
        bytes(&v, sizeof v);
    }
    uint64_t digest() const { return h_; }

private:
    uint64_t h_;
};

}  // namespace tona
