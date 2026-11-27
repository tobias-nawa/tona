// Bounds of the expert-grouped MoE buffers (xg, G, U, O hold one row per (token, rank) assignment, grouped by expert).
// Shared by the TONA_PF_CHECK kernel and the CPU unit test.
#pragma once

#include <cstdint>

#ifdef __HIPCC__
#define TONA_PF_HD __host__ __device__
#else
#define TONA_PF_HD
#endif

namespace tona::gpu {

// Row r belongs to expert e: off[e] <= r < off[e] + cnt[e], and it lies inside the n * ne rows of the chunk. O is
// carved for C * ne rows with n <= C (carve_moe), so this also keeps out[r * d + k] inside b.O.
TONA_PF_HD inline bool pf_row_ok(int64_t r, int e, int E, int64_t rows, const int32_t* off, const int32_t* cnt) {
    if (e < 0 || e >= E) return false;
    return r >= off[e] && r < (int64_t)off[e] + cnt[e] && r >= 0 && r < rows;
}

// The MoE count area (cnt_d, and the host copy pf_hostc_ up to the per-token targets that follow it) for a size class
// of max_e experts per layer (expert_cap: 256 for the Coder, 512 for the wide class): counts [0, E), offsets
// [max_e, + E), fill level [2 * max_e, + E), the NaN word, the flags of TONA_PF_CHECK 24 words before the end. For 256
// these are the words 256, 512, 768, 1000 and 1024 of the Coder.
constexpr int route_off_word(int max_e) { return max_e; }
constexpr int route_fill_word(int max_e) { return 2 * max_e; }
// Word in which k_route counts the tokens with a NaN logit
constexpr int route_nan_word(int max_e) { return 3 * max_e; }
constexpr int route_flag_word(int max_e) { return 4 * max_e - 24; }
// Size of the area in ints
constexpr int route_count_words(int max_e) { return 4 * max_e; }

// A NaN router logit as -inf: every comparison with NaN is false, so route_best would keep its start index 1 << 30 for
// the expert, and the counts, offsets and row map of the scatter would read and write 4 GiB away from their arrays
TONA_PF_HD inline float pf_clean_logit(float x) { return x == x ? x : -__builtin_inff(); }

// The counts of the router must add up to one row per assignment, each expert's range must start where the previous
// one ends and none may be negative
inline bool pf_counts_ok(const int32_t* cnt, int E, int64_t rows) {
    int64_t acc = 0;
    for (int e = 0; e < E; e++) {
        if (cnt[e] < 0) return false;
        acc += cnt[e];
    }
    return acc == rows;
}

}  // namespace tona::gpu
