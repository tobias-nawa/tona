// The expert profile: how often each expert of each layer was used (TONA_EXPERT_PROFILE), the file that stores it, and
// the order of a layer's experts by it that every backend uses to fill its fast memory first.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/expert_layout.h"

namespace tona::backend {

// Version 1 (every layer has the same number of experts): "IEXP" (little endian), layers, experts; then the counts per
// layer as uint32.
// Version 2 (any layout): kProfileMagicV2, layers, total entries; then layers + 1 offsets (uint32: the first entry of
// each layer, the last is the total) and the counts of all layers back to back. The writer uses version 1 whenever it
// can, so that a Coder profile stays readable by older builds; the reader accepts both.
constexpr uint32_t kProfileMagic = 0x50584549u;
constexpr uint32_t kProfileMagicV2 = 0x32584549u;  // "IEX2"

namespace detail {

// All `count` values or false. The stream state is checked as well as the count: a failed read leaves the file position
// indeterminate, so nothing may read on after one (also what the stream analyzer of clang-tidy demands).
inline bool read_values(FILE* f, uint32_t* dst, size_t count) {
    return std::fread(dst, sizeof(uint32_t), count, f) == count && !std::feof(f) && !std::ferror(f);
}

// The offsets of a version 2 file must be those of the layout
inline bool offsets_match(FILE* f, const ExpertLayout& layout) {
    std::vector<uint32_t> off(layout.layers() + 1ull);
    if (!read_values(f, off.data(), off.size())) return false;
    for (uint32_t l = 0; l < layout.layers(); l++)
        if (off[l] != layout.offset(l)) return false;
    return off.back() == layout.total();
}

inline bool header_matches(FILE* f, const ExpertLayout& layout) {
    uint32_t head[3];
    if (!read_values(f, head, 3)) return false;
    if (head[0] == kProfileMagic)
        return layout.is_uniform() && head[1] == layout.layers() && head[2] == layout.experts(0);
    if (head[0] != kProfileMagicV2) return false;
    return head[1] == layout.layers() && head[2] == layout.total() && offsets_match(f, layout);
}

}  // namespace detail

// Counts [layer][expert] of the file, flat (ExpertLayout::offset); empty if the file is missing (start without a
// profile); throws if it is not a profile of this model. `who` names the backend in the message.
inline std::vector<uint32_t> read_expert_profile(const char* path, const ExpertLayout& layout, const char* who) {
    FILE* f = std::fopen(path, "rb");
    if (!f) return {};
    std::vector<uint32_t> v(layout.total());
    const bool ok = detail::header_matches(f, layout) && detail::read_values(f, v.data(), v.size());
    std::fclose(f);
    if (!ok) throw std::runtime_error(std::string(who) + ": profile " + path + " does not match the model");
    return v;
}

// false if the file cannot be written
inline bool write_expert_profile(const char* path, const ExpertLayout& layout, const std::vector<uint32_t>& counts) {
    FILE* f = std::fopen(path, "wb");
    if (!f) return false;
    if (layout.is_uniform() && layout.layers() > 0) {
        const uint32_t head[3] = {kProfileMagic, layout.layers(), layout.experts(0)};
        std::fwrite(head, sizeof head, 1, f);
    } else {
        const uint32_t head[3] = {kProfileMagicV2, layout.layers(), (uint32_t)layout.total()};
        std::fwrite(head, sizeof head, 1, f);
        for (uint32_t l = 0; l < layout.layers(); l++) {
            const uint32_t o = (uint32_t)layout.offset(l);
            std::fwrite(&o, sizeof o, 1, f);
        }
        const uint32_t end = (uint32_t)layout.total();
        std::fwrite(&end, sizeof end, 1, f);
    }
    std::fwrite(counts.data(), sizeof(uint32_t), counts.size(), f);
    return std::fclose(f) == 0;
}

// The experts of one layer, most used first, ties by index; without counts (nullptr) by index
inline std::vector<int> experts_by_count(const uint32_t* cnt, int E) {
    std::vector<int> ord(E);
    std::iota(ord.begin(), ord.end(), 0);
    if (cnt) std::stable_sort(ord.begin(), ord.end(), [&](int a, int b) { return cnt[a] > cnt[b]; });
    return ord;
}

}  // namespace tona::backend
