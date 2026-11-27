// Radix selection of the n_top best scores of one token by one block of TK threads: the shared device code of the QSA
// block selection in prefill (prefill_topk.hip) and decode (op_topk of the megakernel). Both produce the same cells
// byte for byte (threshold = n_top-th largest score as an unsigned integer, ties go to the earlier block); the
// attention reads the cells in this order, so a different order would change its numbers. Included by translation
// units that define WAVE; there is no -fgpu-rdc, so a header is the only way to share device code.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "gpu/matvec_tables.h"

namespace tona::gpu {
namespace {

constexpr int TK = 512, NW = TK / WAVE;
constexpr int NBIN = 2048;  // histogram of one radix pass: 11, 11 and 10 bits of the 32-bit score
constexpr int ROWS = 4;     // loads in flight per lane while a wave walks its stripe

struct TopkLds {
    __attribute__((aligned(16))) int hist[NBIN];
    int wave_sum[NW], wave_gt[NW], wave_eq[NW];
    int pick[2];
};

// The contiguous slice [lo, hi) of the token's scores one wave walks, 32 consecutive elements per lane-step
struct TopkStripe {
    const uint32_t* sc;
    int lo, hi;
};

// Calls f(i, v) for each of the wave's elements, ROWS x 32 at a time; i is -1 for lanes beyond the stripe. Every lane
// calls f the same number of times, so f may use wave-wide operations.
template <class F>
__device__ __forceinline__ void stripe_each(const TopkStripe s, F&& f) {
    const int lane = threadIdx.x % WAVE;
    for (int base = s.lo; base < s.hi; base += WAVE * ROWS) {
        uint32_t v[ROWS];
        int idx[ROWS];
#pragma unroll
        for (int u = 0; u < ROWS; u++) {
            idx[u] = base + u * WAVE + lane;
            v[u] = idx[u] < s.hi ? s.sc[idx[u]] : 0u;
        }
#pragma unroll
        for (int u = 0; u < ROWS; u++) f(idx[u] < s.hi ? idx[u] : -1, v[u]);
    }
}

// State of the radix search: the digits found so far (higher ones) and the rank still wanted among the candidates
struct RadixSel {
    uint32_t thr;
    int k;
};

constexpr int SHIFT[3] = {21, 10, 0};

// The digit at `shift` for which the candidates with a larger digit are fewer than k and those with at least this
// digit are k or more; suffix sums of the histogram by one 4-bin slice per thread, wave scan, then the 16 wave totals
__device__ __forceinline__ RadixSel topk_pick(TopkLds& L, const RadixSel cur, int shift) {
    const int th = threadIdx.x, lane = th % WAVE, w = th / WAVE;
    const int4 h = *reinterpret_cast<const int4*>(&L.hist[4 * th]);
    const int mine = h.x + h.y + h.z + h.w;
    int x = mine;
#pragma unroll
    for (int off = 1; off < WAVE; off <<= 1) {
        const int y = __shfl_down(x, off, WAVE);
        if (lane + off < WAVE) x += y;
    }
    if (lane == 0) L.wave_sum[w] = x;
    __syncthreads();
    int above = x - mine;
    for (int j = w + 1; j < NW; j++) above += L.wave_sum[j];
    if (above < cur.k && cur.k <= above + mine) {  // exactly one thread holds the crossing
        const int bins[4] = {h.w, h.z, h.y, h.x};
        int cum = above;
        for (int b = 0; b < 4; b++) {
            if (cum < cur.k && cur.k <= cum + bins[b]) {
                L.pick[0] = 4 * th + 3 - b;
                L.pick[1] = cur.k - cum;
            }
            cum += bins[b];
        }
    }
    __syncthreads();
    return {cur.thr | ((uint32_t)L.pick[0] << shift), L.pick[1]};
}

// One radix pass: histogram of the digit among the elements that agree with the digits found so far
__device__ __forceinline__ RadixSel topk_radix_pass(const TopkStripe s, TopkLds& L, const RadixSel cur, int pass) {
    const int shift = SHIFT[pass];
    const uint32_t keep = pass == 0 ? 0u : ~0u << SHIFT[pass - 1], dmask = pass == 2 ? 1023u : 2047u;
    for (int i = threadIdx.x; i < NBIN; i += TK) L.hist[i] = 0;
    __syncthreads();
    stripe_each(s, [&](int i, uint32_t v) {
        if (i >= 0 && (v & keep) == cur.thr) atomicAdd(&L.hist[(v >> shift) & dmask], 1);
    });
    __syncthreads();
    return topk_pick(L, cur, shift);
}

__device__ __forceinline__ int wave_sum_int(int v) {
#pragma unroll
    for (int off = WAVE / 2; off > 0; off >>= 1) v += __shfl_xor(v, off, WAVE);
    return v;
}

// Per wave: how many scores of its stripe lie above the threshold and how many equal it
__device__ __forceinline__ void topk_count(const TopkStripe s, TopkLds& L, uint32_t thr) {
    int gt = 0, eq = 0;
    stripe_each(s, [&](int i, uint32_t v) {
        gt += i >= 0 && v > thr;
        eq += i >= 0 && v == thr;
    });
    gt = wave_sum_int(gt);
    eq = wave_sum_int(eq);
    if (threadIdx.x % WAVE == 0) L.wave_gt[threadIdx.x / WAVE] = gt, L.wave_eq[threadIdx.x / WAVE] = eq;
}

// Where the wave's first cell goes and how many of the scores equal to the threshold it may still take: the stripes
// before it take their share of the need_eq earliest equal scores first
struct WaveStart {
    int out, eq_left;
};
__device__ __forceinline__ WaveStart topk_wave_start(const TopkLds& L, int need_eq) {
    int out = 0, eq_before = 0;
    for (int j = 0; j < (int)threadIdx.x / WAVE; j++) {
        out += L.wave_gt[j] + std::clamp(need_eq - eq_before, 0, L.wave_eq[j]);
        eq_before += L.wave_eq[j];
    }
    return {out, std::max(0, need_eq - eq_before)};
}

// Writes the cells of the selected blocks of the wave's stripe in ascending order: a ballot per 32 elements gives each
// selected lane its slot
__device__ __forceinline__ void topk_emit(const TopkStripe s, uint32_t thr, WaveStart st, int32_t* cells, int blk) {
    const uint32_t below = (1u << (threadIdx.x % WAVE)) - 1;
    stripe_each(s, [&](int i, uint32_t v) {
        const bool eq = i >= 0 && v == thr;
        const uint32_t me = (uint32_t)__ballot(eq);
        const bool take = (i >= 0 && v > thr) || (eq && (int)__popc(me & below) < st.eq_left);
        const uint32_t mt = (uint32_t)__ballot(take);
        if (take) {
            const int slot = st.out + __popc(mt & below);
            for (int k = 0; k < blk; k++) cells[slot * blk + k] = i * blk + k;
        }
        st.out += __popc(mt);
        st.eq_left = std::max(0, st.eq_left - (int)__popc(me));
    });
}

// The stripe of the n_pool scores `sc` the wave of the calling thread walks: 16 contiguous slices of whole 32-element
// rows
__device__ __forceinline__ TopkStripe topk_stripe(const uint32_t* sc, int n_pool) {
    const int w = threadIdx.x / WAVE;
    const int stripe = ((n_pool + NW - 1) / NW + WAVE - 1) / WAVE * WAVE;
    const int lo = std::min(w * stripe, n_pool);
    return {sc, lo, std::min(n_pool, lo + stripe)};
}

// Writes the cells of the n_top best scores of the stripes in ascending block order: three radix passes find the
// threshold, a count and a ballot emit place the cells. Every thread of the block must call it; the caller
// synchronizes before it reuses L. k_topk of the prefill spells these steps out itself, so that moving the code into
// this header leaves its machine code unchanged.
__device__ __forceinline__ void topk_cells(const TopkStripe s, int n_top, int blk, int32_t* cells, TopkLds& L) {
    RadixSel sel{0u, n_top};
    for (int pass = 0; pass < 3; pass++) sel = topk_radix_pass(s, L, sel, pass);
    topk_count(s, L, sel.thr);
    __syncthreads();
    topk_emit(s, sel.thr, topk_wave_start(L, sel.k), cells, blk);
}

}  // namespace
}  // namespace tona::gpu
