// Frozen copy of k_attn_wmma as it was before the gather pipeline (commit 75f85ac). test_gpu_attn_wmma compares
// the production kernel with it byte for byte; do not change this file when the kernel changes.
#pragma once
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu::ref {

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
    const float* qr = q + ((int64_t)c.t * nh + (int64_t)c.kvh * c.g + min(c.r16, c.g - 1)) * fa::HD + d0 + c.kh;
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
        cell.kc = (const int8_t*)r + (int64_t)c.kvh * 2 * HD + (int64_t)gq * 32;
        const float* sc = (const float*)(r + (int64_t)c.nkv * 2 * HD) + (int64_t)c.kvh * 16;
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
    __syncthreads();  // everyone has read mrow
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

__global__ void __launch_bounds__(TPB) k_attn_wmma_ref(AttnWmmaArgs a) {
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
    for (int c0 = 0; c0 < n_cells; c0 += CC) {
        const int cn = min(CC, n_cells - c0);
        const FaCell cell = fa_stage_k(c, KV, c0, cn);
        __syncthreads();
        fa_scores(c, m, qa);
        __syncthreads();  // Ks read: V goes into the same buffer
        fa_stage_v(c, KV, cell);
        fa_softmax(c, m, cn);
        __syncthreads();
        fa_accumulate(c, m, acc);
        __syncthreads();
    }
    fa_write(c, m, acc, a);
}

inline void launch_attn_wmma_ref(const AttnWmmaArgs& a, int n_tokens, hipStream_t st) {
    k_attn_wmma_ref<<<n_tokens * a.nkv, TPB, 0, st>>>(a);
}

}  // namespace tona::gpu::ref
