// Where the KV cache of the full-attention layers lives when it is split between VRAM and pinned RAM (Ornith, Bolt).
// The decode attention reads every cached position of every attention layer once per pass: over PCIe from RAM
// (18 to 24 GB/s), from VRAM at the memory bandwidth. A VRAM budget therefore holds the first `hot` positions of ALL
// layers; positions from `hot` on live in pinned RAM. A context up to `hot` is entirely in VRAM, at any max_ctx
// (research/measurements/2026-11-20-kv-hot-head.md). The earlier split by layer left the contexts above 140K with
// some layers fully in RAM, and one layer in RAM costs 3.6 ms per pass at 64K against 0.3 ms in VRAM.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace tona::gpu {

struct KvSplit {
    // Cells of every main attention layer in VRAM: 0 = all in RAM, >= max_ctx = all in VRAM, in between = split.
    // -1: not planned, the model follows Mega::kv_host_ (the Coder).
    int64_t hot = -1;
    bool mtp_host = false;   // the MTP layer's cache (always whole: its attention runs at every draft step)
    double host_frac = 1.0;  // share of the KV bytes in RAM, for the pinned-RAM budget

    bool planned() const { return hot >= 0; }
    // Cells of buffer idx (the main layers in allocation order, then the MTP layer) that stay in VRAM
    int64_t cells_in_vram(size_t idx, size_t n_main, int64_t max_ctx) const {
        if (idx >= n_main) return mtp_host ? 0 : max_ctx;
        return std::min(hot, max_ctx);
    }

    // The MTP layer first (a whole buffer, or RAM if the budget does not even hold it), then as many cells of the main
    // layers as the rest holds, rounded down to 256 cells. Fewer than min_cells is not worth the VRAM: all in RAM.
    static KvSplit plan(size_t n_main, size_t n_mtp, int64_t max_ctx, size_t row, size_t budget) {
        constexpr int64_t kMinCells = 2048;
        KvSplit p;
        const size_t buf = (size_t)max_ctx * row, total = (n_main + n_mtp) * buf;
        if (budget >= total || n_main == 0) {
            p.hot = max_ctx, p.host_frac = 0.0;
            return p;
        }
        p.mtp_host = n_mtp > 0 && budget < n_mtp * buf;
        const size_t left = p.mtp_host ? budget : budget - n_mtp * buf;
        int64_t cells = (int64_t)(left / n_main / row) / 256 * 256;
        p.hot = cells < kMinCells ? 0 : std::min(cells, max_ctx);
        const double in_vram = (double)n_main * (double)p.hot * (double)row + (p.mtp_host ? 0.0 : (double)n_mtp * buf);
        p.host_frac = 1.0 - in_vram / (double)total;
        return p;
    }

    // VRAM for the KV cache when nothing sets it: enough to hold the first 128,000 cells of every layer (coding
    // sessions reach 100K and more) and the whole MTP layer, but at most 13.5 % of the VRAM that is free after the
    // reserve. Measured at ctx 262144 (research/measurements/2026-11-20-kv-hot-head.md, block 4): fewer cells than the
    // 182K of 2.4 GB win at 2K, 16K and 64K because the expert slots they free pay more (Ornith 1.8 GB, 130K cells:
    // decode +2.2 / +3.2 / +1.8 % against 2.4 GB; 1.2 GB, 78K cells: +3.7 / +4.3 / +2.7 %, but its 128K point is not
    // covered). The 13.5 % is the 1.8 GB of Ornith out of the 13.4 GB free there.
    static size_t default_budget(size_t n_main, size_t n_mtp, int64_t max_ctx, size_t row, int64_t free_bytes,
                                 int64_t reserve_bytes) {
        constexpr int64_t kCoverCells = 128000;
        const size_t want = (n_main * (size_t)std::min(kCoverCells, max_ctx) + n_mtp * (size_t)max_ctx) * row;
        const size_t cap = (size_t)std::max<int64_t>(0, (int64_t)(0.135 * (double)(free_bytes - reserve_bytes)));
        return std::min(want, cap);
    }
};

}  // namespace tona::gpu
