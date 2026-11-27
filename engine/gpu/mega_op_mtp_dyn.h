// Dynamic rows of the MTP draft head: a low-rank scorer ranks every token outside the static list, the best rows of
// each partition get exact logits (they are copied from the LM head of the target into a buffer the matvec of the
// head reads). Three steps: project x, score and select, gather the rows.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Partitions of the candidate tiles. Fixed (not the grid size) so the selection does not depend on the number of
// blocks: a block handles partition b, b + gridDim.x, ...
constexpr int MTP_PARTS = 32;
// Scores per lane (tiles of 32 rows per wave): MTP_PARTS * MWAVES * MTP_TPW tiles cover 262,144 candidates
constexpr int MTP_TPW = 16;

// p = B^T x for the last token of the pass. p: basis^T float [r][d], x (row stride i[2] floats), out float [r]; i: r, d
// (d a multiple of 4: 16-byte loads)
__device__ __noinline__ void op_mtp_proj(const MegaOp& op, const Ctx& c) {
    const float4* bt = (const float4*)op.p[0];
    const float4* x = (const float4*)((const float*)op.p[1] + (int64_t)(c.nb - 1) * op.i[2]);
    float* out = (float*)op.p[2];
    const int r = op.i[0], d4 = op.i[1] / 4;
    for (int64_t j = c.gw; j < r; j += c.GW) {
        float s = 0;
        for (int k = c.lane; k < d4; k += WAVE) {
            const float4 a = bt[j * d4 + k], b = x[k];
            s += a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
        }
        s = wave_sum(s);
        if (c.lane == 0) out[j] = s;
    }
}

// Order-preserving key of a score: bigger score, bigger key; 0 is reserved for "no candidate"
__device__ __forceinline__ uint32_t score_key(float v) {
    const uint32_t u = __float_as_uint(logit_clamp(v));
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

__device__ __forceinline__ int sext8(uint32_t w, int k) { return (int)(int8_t)(w >> (8 * k)); }

// Score of the row `lane` of tile `tile`: 16 int8 values of the row per 16-byte load, the 32 lanes of the wave next to
// each other. pv: p in LDS (16-byte aligned).
__device__ __forceinline__ float tile_score(const uint4* tiles, int64_t tile, int r16, int lane, const float* pv) {
    float acc = 0;
    for (int kk = 0; kk < r16; kk++) {
        const uint4 q = tiles[(tile * r16 + kk) * WAVE + lane];
        const uint32_t w[4] = {q.x, q.y, q.z, q.w};
        const float4* pp = (const float4*)pv + kk * 4;
        for (int a = 0; a < 4; a++) {
            const float4 p4 = pp[a];
            acc += (float)sext8(w[a], 0) * p4.x + (float)sext8(w[a], 1) * p4.y + (float)sext8(w[a], 2) * p4.z +
                   (float)sext8(w[a], 3) * p4.w;
        }
    }
    return acc;
}

// Copies of the histogram of a radix pass: the lanes of a wave hit the same few bins (the scores lie close together),
// so the lanes count in 8 copies that are added when the boundary is searched
constexpr int HIST_COPIES = 8;
static_assert(THREADS >= 256, "the boundary search of radix_pass uses one thread per bin");

// Histogram of one byte of the keys (those whose upper bytes equal `prefix`) in HIST_COPIES x 256 ints of LDS
__device__ __forceinline__ void radix_count(const uint32_t* keys, int shift, uint32_t prefix, int* hist) {
    const int t = threadIdx.x;
    for (int i = t; i < HIST_COPIES * 256; i += THREADS) hist[i] = 0;
    __syncthreads();
    int* mine = hist + ((t % WAVE) % HIST_COPIES) * 256;
    _Pragma("unroll") for (int i = 0; i < MTP_TPW; i++) {
        const uint32_t k = keys[i];
        if (k != 0 && (shift == 24 || (k >> (shift + 8)) == prefix)) atomicAdd(&mine[(k >> shift) & 255], 1);
    }
    __syncthreads();
}

// Boundary digit of the histogram: the largest digit with at least `need` keys at or above it, and how many keys are
// still needed in that digit (res; 0 and 0 if there are fewer). Bins from 255 down, one per thread of the first
// 256: running sum over the block. wt: one slot per wave
__device__ __forceinline__ void radix_boundary(int need, const int* hist, int* res, uint32_t* wt) {
    const int t = threadIdx.x, lane = t % WAVE;
    if (t == 0) res[0] = 0, res[1] = 0;
    const int b = 255 - t;
    int v = 0;
    if (t < 256)
        for (int c = 0; c < HIST_COPIES; c++) v += hist[c * 256 + b];
    int inc = v;
    for (int off = 1; off < WAVE; off *= 2) {
        const int o = __shfl_up(inc, off, WAVE);
        if (lane >= off) inc += o;
    }
    if (lane == WAVE - 1) wt[t / WAVE] = (uint32_t)inc;
    __syncthreads();
    int before = inc - v;
    for (int w = 0; w < t / WAVE; w++) before += (int)wt[w];
    if (t < 256 && before < need && before + v >= need) res[0] = b, res[1] = need - before;
    __syncthreads();
}

// Exclusive scan over the threads of the block of a packed pair of counters (gt | tie << 16), in thread order
__device__ __forceinline__ uint32_t block_scan_pair(uint32_t v, uint32_t* wtot) {
    const int lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
    uint32_t inc = v;
    for (int off = 1; off < WAVE; off *= 2) {
        const uint32_t o = __shfl_up(inc, off, WAVE);
        if (lane >= off) inc += o;
    }
    if (lane == WAVE - 1) wtot[w] = inc;
    __syncthreads();
    uint32_t base = 0;
    for (int i = 0; i < w; i++) base += wtot[i];
    return base + inc - v;
}

// LDS of a partition: p, the histogram of a radix pass, its result, the sums per wave
struct DynLds {
    float* pv;
    int* hist;
    int* res;
    uint32_t* wtot;
};

__device__ __forceinline__ DynLds dyn_lds(float* sm) {
    DynLds l;
    l.pv = sm, l.hist = (int*)(sm + 256), l.res = l.hist + HIST_COPIES * 256, l.wtot = (uint32_t*)(l.res + 4);
    return l;
}

// One radix pass over the keys of the partition
__device__ __forceinline__ void radix_pass(const uint32_t* keys, int shift, uint32_t prefix, int need,
                                           const DynLds& l) {
    radix_count(keys, shift, prefix, l.hist);
    radix_boundary(need, l.hist, l.res, l.wtot);
}

// Keys of the scores of this wave's tiles of partition `part` (0: no candidate)
__device__ __forceinline__ void mtp_dyn_keys(const MegaOp& op, int part, const float* pv, uint32_t* keys) {
    const uint4* tiles = (const uint4*)op.p[0];
    const float* scale = (const float*)op.p[1];
    const int32_t* ids = (const int32_t*)op.p[2];
    const uint8_t* excl = (const uint8_t*)op.p[3];
    const int lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
    _Pragma("unroll") for (int i = 0; i < MTP_TPW; i++) {
        const int64_t tile = (int64_t)(w + MWAVES * i) * MTP_PARTS + part;
        keys[i] = 0;
        if (tile >= op.i[0]) continue;
        const int64_t row = tile * WAVE + lane;
        const float s = tile_score(tiles, tile, op.i[1] / 16, lane, pv) * scale[row];
        if (ids[row] >= 0 && !excl[row]) keys[i] = score_key(s);
    }
}

// The best m keys, in the order of (wave, lane, i): the keys above the boundary of the radix passes, then as many of
// those at the boundary as are missing. Token ids into out[0 .. m).
__device__ __forceinline__ void mtp_dyn_emit(const MegaOp& op, int part, const uint32_t* keys, const DynLds& l) {
    const int32_t* ids = (const int32_t*)op.p[2];
    int32_t* out = (int32_t*)op.p[5] + (int64_t)part * op.i[3];
    const int m = op.i[3];
    const int lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
    radix_pass(keys, 24, 0, m, l);
    const uint32_t d1 = l.res[0];
    const int need1 = l.res[1];
    __syncthreads();  // everybody has read the first result before the next pass clears it
    radix_pass(keys, 16, d1, need1, l);
    const uint32_t b16 = d1 << 8 | (uint32_t)l.res[0];
    const int ties = l.res[1];
    uint32_t mine = 0;
    _Pragma("unroll") for (int i = 0; i < MTP_TPW; i++) {
        const uint32_t k16 = keys[i] >> 16;
        mine += keys[i] != 0 ? (k16 > b16 ? 1u : (k16 == b16 ? 0x10000u : 0u)) : 0u;
    }
    const uint32_t ex = block_scan_pair(mine, l.wtot);
    uint32_t* total = l.wtot + MWAVES;
    if (threadIdx.x == THREADS - 1) *total = (ex & 0xFFFF) + (mine & 0xFFFF);
    __syncthreads();
    const uint32_t total_gt = *total;
    uint32_t pg = ex & 0xFFFF, pt = ex >> 16;
    _Pragma("unroll") for (int i = 0; i < MTP_TPW; i++) {
        const uint32_t k = keys[i], k16 = k >> 16;
        if (k == 0) continue;
        const int id = ids[((int64_t)(w + MWAVES * i) * MTP_PARTS + part) * WAVE + lane];
        const bool gt = k16 > b16, tie = k16 == b16 && (int)pt < ties;
        const uint32_t slot = gt ? pg : total_gt + pt;
        if ((gt || tie) && slot < (uint32_t)m) out[slot] = id;
        pg += gt, pt += k16 == b16;
    }
}

// Rows of the head of the target for the tokens this partition selected, into its slots of the row buffer
// (p[6]: head, p[7]: buffer; i[4]: bytes per row). The partition's selection is complete in this block, no other block
// is needed.
__device__ __forceinline__ void mtp_dyn_gather(const MegaOp& op, int part) {
    const uint8_t* head = (const uint8_t*)op.p[6];
    uint8_t* dst = (uint8_t*)op.p[7];
    const int32_t* slots = (const int32_t*)op.p[5] + (int64_t)part * op.i[3];
    const int rb = op.i[4], lane = threadIdx.x % WAVE;
    for (int s = threadIdx.x / WAVE; s < op.i[3]; s += MWAVES) {
        const int32_t t = slots[s];
        if (t < 0) continue;
        copy_row(dst + ((int64_t)part * op.i[3] + s) * rb, head + (int64_t)t * rb, rb, lane, WAVE);
    }
}

// One partition: scores of its tiles, then the best `m` of them
__device__ __forceinline__ void mtp_dyn_part(const MegaOp& op, int part, float* sm) {
    const DynLds l = dyn_lds(sm);
    int32_t* out = (int32_t*)op.p[5] + (int64_t)part * op.i[3];
    for (int i = threadIdx.x; i < op.i[1]; i += THREADS) l.pv[i] = ((const float*)op.p[4])[i];
    for (int i = threadIdx.x; i < op.i[3]; i += THREADS) out[i] = -1;
    __syncthreads();
    uint32_t keys[MTP_TPW];
    mtp_dyn_keys(op, part, l.pv, keys);
    mtp_dyn_emit(op, part, keys, l);
    __syncthreads();
    mtp_dyn_gather(op, part);
}

// Scores, selection and the rows of the selected tokens per partition. p: tiles uint4 [ntiles][r/16][32], scale float
// and token ids int32 and excluded (uint8) per candidate row in tile order, p vector, out slots, head, row buffer;
// i: ntiles, r, -, m (slots per partition), words per row
__device__ __noinline__ void op_mtp_dyn(const MegaOp& op, const Ctx& c) {
    for (int part = blockIdx.x; part < MTP_PARTS; part += gridDim.x) mtp_dyn_part(op, part, c.sm);
}

}  // namespace
}  // namespace tona::gpu
