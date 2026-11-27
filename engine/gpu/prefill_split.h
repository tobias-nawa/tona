// Row-wise prefill launches in pieces. A kernel that runs 1 ms or longer is followed by a stall of about 280 us before
// the next queued kernel starts (research/measurements/2026-11-16-prefill-idle-gaps.md: 0 % of the kernels below 1 ms,
// 78 % of those from 1 ms on). The kernels split here work per token row, so a piece is the same kernel on shifted row
// pointers: every output is computed from the same inputs in the same order.
#pragma once

#include <algorithm>
#include <cstdint>

#include "backend/env.h"

namespace tona::gpu {

// Longest piece in rows (TONA_PF_SPLIT_ROWS, 0: no split). 2048 rows: k_moe_sum, the longest of the split kernels,
// needs 0.41 us per row, so a piece stays at 0.84 ms or less
inline int split_rows_max() {
    static const int v = (int)tona::env::integer("TONA_PF_SPLIT_ROWS", 2048);
    return v;
}

// Calls f(row0, rows) for equal pieces of at most split_rows_max() rows that cover [0, n)
template <class F>
void for_row_pieces(int n, F&& f) {
    const int mx = split_rows_max();
    const int pieces = mx > 0 ? std::max(1, (n + mx - 1) / mx) : 1;
    for (int p = 0, r0 = 0; p < pieces; p++) {
        const int r1 = (int)((int64_t)n * (p + 1) / pieces);
        f(r0, r1 - r0);
        r0 = r1;
    }
}

}  // namespace tona::gpu
