// The flash-attention kernel with WMMA over the sparse selection (QSA prefill), first form: see k_attn_wmma below.
// Device code only, included by prefill_attn.hip (there is no -fgpu-rdc). Kept unchanged as the byte-for-byte
// reference of attn_wmma2.h and as the fallback (TONA_ATTN_WMMA_OLD=1).
#pragma once

#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {
namespace {

// Flash attention with WMMA: block per (token, KV head). The nh/nkv q heads of the group (12) are the rows of a
// 16-row tile; they see the same cells (selection per token). Per 32 cells: K and V from the int8 cache to f16 in
// the LDS (V transposed), S = Q K^T (8 waves: one cell tile and a quarter of the dimensions each, then summed),
// online softmax, O += P V (32 dimensions per wave). At the end O / l * sigmoid(gate) to f16.
// Two gathers with the same arithmetic in the same order (bit-identical, tests/test_gpu_attn_wmma.hip): direct, and
// pipelined (the rows of the next 32 cells are requested while the current ones compute). Pipelined is 7 % faster with
// the KV cache in VRAM and 3 to 6 % slower in pinned RAM, where PCIe limits
// (research/measurements/2026-10-30-prefill-attn.md).
namespace fa {
constexpr int HD = 256, CC = 32, LQ = HD + 8, LV = CC + 8;
}  // namespace fa

// Per block: the (token, KV head) it works on and the position of the thread in its wave
struct FaCtx {
    const uint8_t* kv;
    int64_t row;
    const int32_t* cells;  // null: all cells up to the token
    int nkv, g, kvh, t;
    int tid, w, r16, kh;
};

// Pointers to the LDS arrays of the kernel. K and V^T share one buffer (K for S, then V for P V): under 32 KB of LDS,
// two blocks per CU.
struct FaLds {
    _Float16 *kvbuf, *ps;
    float (*part)[16][fa::CC];
    float *mrow, *lrow, *arow;
};

// Q of the group (rows >= g zero), with the scale 1/sqrt(hd): wave w needs row r16, dimensions (w >> 1) * 64 + kk + kh
__device__ __forceinline__ void fa_load_q(const FaCtx c, const float* q, int nh, float scale, half8 (&qa)[4]) {
    const int d0 = (c.w >> 1) * 64;
    const float* qr = q + ((int64_t)c.t * nh + c.kvh * c.g + min(c.r16, c.g - 1)) * fa::HD + d0 + c.kh;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) qa[i][j] = c.r16 < c.g ? (_Float16)(qr[i * 16 + j] * scale) : (_Float16)0.0f;
}

// The cell a thread stages: its int8 K row (V follows HD bytes later) and the V scale; kc null past the last cell
struct FaCell {
    const int8_t* kc;
    float vs;
};

// K to f16 in the LDS: thread per (cell, 32 dimensions). Cell fastest: the transposed V writes of a wave sit next to
// each other (with gq fastest, 8 threads hit the same bank).
__device__ __forceinline__ FaCell fa_stage_k(const FaCtx c, _Float16* Ks, int c0, int cn) {
    using namespace fa;
    const int cc = c.tid & 31, gq = c.tid >> 5;
    FaCell cell{nullptr, 0};
    if (cc < cn) {
        const int64_t id = c.cells ? c.cells[c0 + cc] : c0 + cc;
        const uint8_t* r = c.kv + id * c.row;
        cell.kc = (const int8_t*)r + c.kvh * 2 * HD + gq * 32;
        const float* sc = (const float*)(r + (int64_t)c.nkv * 2 * HD) + c.kvh * 16;
        const float ks = sc[gq];
        cell.vs = sc[8 + gq];
        int8_t kb[32];
        *reinterpret_cast<uint4*>(kb) = *reinterpret_cast<const uint4*>(cell.kc);
        *reinterpret_cast<uint4*>(kb + 16) = *reinterpret_cast<const uint4*>(cell.kc + 16);
#pragma unroll
        for (int q8 = 0; q8 < 4; q8++) {
            half8 h;
#pragma unroll
            for (int j = 0; j < 8; j++) h[j] = (_Float16)((float)kb[8 * q8 + j] * ks);
            *reinterpret_cast<half8*>(&Ks[cc * LQ + gq * 32 + 8 * q8]) = h;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 32; j++) Ks[cc * LQ + gq * 32 + j] = (_Float16)0.0f;
    }
    return cell;
}

// S parts: wave w -> cell tile w & 1, dimensions (w >> 1) * 64 ... + 63
__device__ __forceinline__ void fa_scores(const FaCtx c, const FaLds m, const half8 (&qa)[4]) {
    using namespace fa;
    const int ct = c.w & 1, d0 = (c.w >> 1) * 64;
    float8 s = {};
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
        const half8 b = *reinterpret_cast<const half8*>(&m.kvbuf[(ct * 16 + c.r16) * LQ + d0 + kk + c.kh]);
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[kk / 16], b, s);
    }
#pragma unroll
    for (int e = 0; e < 8; e++) m.part[c.w >> 1][c.kh + e][ct * 16 + c.r16] = s[e];
}

// V transposed into the same buffer
__device__ __forceinline__ void fa_stage_v(const FaCtx c, _Float16* Vt, const FaCell cell) {
    using namespace fa;
    const int cc = c.tid & 31, gq = c.tid >> 5;
    if (cell.kc) {
        int8_t vb[32];
        *reinterpret_cast<uint4*>(vb) = *reinterpret_cast<const uint4*>(cell.kc + HD);
        *reinterpret_cast<uint4*>(vb + 16) = *reinterpret_cast<const uint4*>(cell.kc + HD + 16);
#pragma unroll
        for (int j = 0; j < 32; j++) Vt[(gq * 32 + j) * LV + cc] = (_Float16)((float)vb[j] * cell.vs);
    } else {
#pragma unroll
        for (int j = 0; j < 32; j++) Vt[(gq * 32 + j) * LV + cc] = (_Float16)0.0f;
    }
}

// Pipelined gather. Raw bytes of the cell a thread stages: K and V (int8, 32 dimensions each) and their scales
struct FaRaw {
    bool valid;  // false past the last cell
    float ks, vs;
    uint4 k[2], v[2];
};

// Cell id at index ci (from the selection, or the position itself); -1 past the last cell. The address is formed one
// iteration later, so nothing waits for this load.
__device__ __forceinline__ int fa_id(const FaCtx c, int ci, int n_cells) {
    return ci >= n_cells ? -1 : c.cells ? c.cells[ci] : ci;
}

// All global loads of one cell at once (K, V, scales); nothing is consumed here. The loads are unconditional (a
// missing cell reads row 0 and is ignored): inside a branch the compiler waits for them before the join, and with
// byte arrays instead of words it splits them into odd pieces (measured: 1.3x slower, 197 instead of 119 VGPR).
__device__ __forceinline__ FaRaw fa_fetch(const FaCtx c, int id) {
    using namespace fa;
    const int gq = c.tid >> 5;
    const uint8_t* r = c.kv + (int64_t)max(id, 0) * c.row;
    const uint4* kc = reinterpret_cast<const uint4*>(r + (int64_t)c.kvh * 2 * HD + (int64_t)gq * 32);
    const float* sc = (const float*)(r + (int64_t)c.nkv * 2 * HD) + (int64_t)c.kvh * 16;
    FaRaw w;
    w.valid = id >= 0;
    w.k[0] = kc[0], w.k[1] = kc[1];
    w.v[0] = kc[HD / 16], w.v[1] = kc[HD / 16 + 1];
    w.ks = sc[gq];
    w.vs = sc[8 + gq];
    return w;
}

// Dimension j (0 ... 31) of a thread's 32 int8 values
__device__ __forceinline__ float fa_byte(const uint4 (&q)[2], int j) {
    const uint4 u = q[j >> 4];
    const int wd = j >> 2 & 3;
    const uint32_t word = wd == 0 ? u.x : wd == 1 ? u.y : wd == 2 ? u.z : u.w;
    return (float)(int8_t)(word >> (8 * (j & 3)));
}

__device__ __forceinline__ void fa_stage_k_raw(const FaCtx c, _Float16* Ks, const FaRaw w) {
    using namespace fa;
    const int cc = c.tid & 31, gq = c.tid >> 5;
    if (w.valid) {
#pragma unroll
        for (int q8 = 0; q8 < 4; q8++) {
            half8 h;
#pragma unroll
            for (int j = 0; j < 8; j++) h[j] = (_Float16)(fa_byte(w.k, 8 * q8 + j) * w.ks);
            *reinterpret_cast<half8*>(&Ks[cc * LQ + gq * 32 + 8 * q8]) = h;
        }
    } else {
#pragma unroll
        for (int j = 0; j < 32; j++) Ks[cc * LQ + gq * 32 + j] = (_Float16)0.0f;
    }
}

__device__ __forceinline__ void fa_stage_v_raw(const FaCtx c, _Float16* Vt, const FaRaw w) {
    using namespace fa;
    const int cc = c.tid & 31, gq = c.tid >> 5;
    if (w.valid) {
#pragma unroll
        for (int j = 0; j < 32; j++) Vt[(gq * 32 + j) * LV + cc] = (_Float16)(fa_byte(w.v, j) * w.vs);
    } else {
#pragma unroll
        for (int j = 0; j < 32; j++) Vt[(gq * 32 + j) * LV + cc] = (_Float16)0.0f;
    }
}

// Online softmax: 16 threads per row, 2 cells per thread, sums by shuffle within the group of 16
__device__ __forceinline__ void fa_softmax(const FaCtx c, const FaLds m, int cn) {
    using namespace fa;
    const int r = c.tid >> 4, cc = (c.tid & 15) * 2;
    float s2[2];
#pragma unroll
    for (int q = 0; q < 2; q++)
        s2[q] = cc + q < cn ? m.part[0][r][cc + q] + m.part[1][r][cc + q] + m.part[2][r][cc + q] + m.part[3][r][cc + q]
                            : -INFINITY;
    const float mo = m.mrow[r];
    float mx = fmaxf(mo, fmaxf(s2[0], s2[1]));
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 16));
    float l = 0;
#pragma unroll
    for (int q = 0; q < 2; q++) {
        const float e = s2[q] > -INFINITY ? __expf(s2[q] - mx) : 0.0f;
        l += e;
        m.ps[r * LV + cc + q] = (_Float16)e;
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) l += __shfl_xor(l, o, 16);
    lds_sync();  // everyone has read mrow
    if ((c.tid & 15) == 0) {
        const float al = mo > -INFINITY ? __expf(mo - mx) : 0.0f;
        m.mrow[r] = mx, m.lrow[r] = m.lrow[r] * al + l, m.arow[r] = al;
    }
}

// O = alpha * O + P V; wave w: dimensions 32w ... 32w + 31 (two tiles)
__device__ __forceinline__ void fa_accumulate(const FaCtx c, const FaLds m, float8 (&acc)[2]) {
    using namespace fa;
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) acc[j][e] *= m.arow[c.kh + e];
#pragma unroll
    for (int kk = 0; kk < CC; kk += 16) {
        const half8 a = *reinterpret_cast<const half8*>(&m.ps[c.r16 * LV + kk + c.kh]);
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const half8 b = *reinterpret_cast<const half8*>(&m.kvbuf[(c.w * 32 + j * 16 + c.r16) * LV + kk + c.kh]);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc[j]);
        }
    }
}

// One iteration of the pipelined loop on the 32 cells already in registers; same steps and syncs as the direct loop
__device__ __forceinline__ void fa_iterate(const FaCtx c, const FaLds m, const half8 (&qa)[4], float8 (&acc)[2],
                                           const FaRaw w, int cn) {
    fa_stage_k_raw(c, m.kvbuf, w);
    lds_sync();
    fa_scores(c, m, qa);
    lds_sync();  // Ks read: V goes into the same buffer
    fa_stage_v_raw(c, m.kvbuf, w);
    fa_softmax(c, m, cn);
    lds_sync();
    fa_accumulate(c, m, acc);
    lds_sync();
}

// acc[j][e]: row kh + e (q head of the group), dimension w * 32 + j * 16 + r16
__device__ __forceinline__ void fa_write(const FaCtx c, const FaLds m, const float8 (&acc)[2], const AttnWmmaArgs a) {
    using namespace fa;
#pragma unroll
    for (int j = 0; j < 2; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int r = c.kh + e;
            if (r >= c.g) continue;
            const int h = c.kvh * c.g + r, dim = c.w * 32 + j * 16 + c.r16;
            const int64_t i = ((int64_t)c.t * a.nh + h) * HD + dim;
            a.o[i] = (_Float16)(acc[j][e] / m.lrow[r] * sigm(a.gate[i]));
        }
}

template <bool Pipe>
__global__ void __launch_bounds__(TPB) k_attn_wmma(AttnWmmaArgs a) {
    using namespace fa;
    __shared__ __attribute__((aligned(16))) _Float16 KV[CC * LQ > HD * LV ? CC * LQ : HD * LV], Ps[16 * LV];
    __shared__ float part[4][16][CC], mrow[16], lrow[16], arow[16];
    const int t = blockIdx.x / a.nkv, kvh = blockIdx.x % a.nkv, g = a.nh / a.nkv;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int64_t pos = a.pos0 + t;
    const bool sel = qsa_sel(pos, a.blk, a.topk);
    const int32_t* cells = sel ? a.cells_all + (int64_t)t * MAX_CELLS : nullptr;
    const int n_cells = sel ? a.ncells[t] : (int)(pos + 1);
    if (tid < 16) mrow[tid] = -INFINITY, lrow[tid] = 0;
    float8 acc[2] = {};
    const int r16 = lane & 15, kh = (lane >> 4) * 8;
    const FaCtx c{a.kv, a.row, cells, a.nkv, g, kvh, t, tid, w, r16, kh};
    const FaLds m{KV, Ps, part, mrow, lrow, arow};
    half8 qa[4];
    fa_load_q(c, a.q, a.nh, a.scale, qa);
    if constexpr (Pipe) {
        // The rows of iteration i + 1 are requested before iteration i computes, the cell indices of i + 2 right after
        // them (research/34, idea 1b).
        const int cc = tid & 31;
        FaRaw cur = fa_fetch(c, fa_id(c, cc, n_cells));
        int next_id = fa_id(c, CC + cc, n_cells);
#pragma unroll 1
        for (int c0 = 0; c0 < n_cells; c0 += CC) {
            const FaRaw next = fa_fetch(c, next_id);
            next_id = fa_id(c, c0 + 2 * CC + cc, n_cells);
            fa_iterate(c, m, qa, acc, cur, min(CC, n_cells - c0));
            cur = next;
        }
    } else {
        for (int c0 = 0; c0 < n_cells; c0 += CC) {
            const int cn = min(CC, n_cells - c0);
            const FaCell cell = fa_stage_k(c, KV, c0, cn);
            lds_sync();
            fa_scores(c, m, qa);
            lds_sync();  // Ks read: V goes into the same buffer
            fa_stage_v(c, KV, cell);
            fa_softmax(c, m, cn);
            lds_sync();
            fa_accumulate(c, m, acc);
            lds_sync();
        }
    }
    fa_write(c, m, acc, a);
}

}  // namespace
}  // namespace tona::gpu
