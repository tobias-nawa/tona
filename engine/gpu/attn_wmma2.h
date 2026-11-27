// Second form of the QSA prefill attention kernel (see attn_wmma1.h for the shape): the same arithmetic in the same
// order, so the output is byte-equal to attn_wmma1.h (tests/test_gpu_attn_wmma.hip), with fewer instructions per 32
// cells:
//  - dequantization of an int8 value x with the f32 scale s as fma(x + 128, s, -128 s): the product is exact in the
//    fma, so it rounds once, to the same f32 as (float)x * s
//    (research/measurements/2026-11-20-coder-sparse-attention.md); the byte comes from the word by v_cvt_f32_ubyteN
//    after flipping its sign bit, no sign extension;
//  - a cell past the last one has scale 0 and so dequantizes to +0, without a branch;
//  - V is staged by pairs of cells (one 32-bit store of two halves instead of two 16-bit stores);
//  - S^T = K Q^T: the accumulator of a lane holds eight consecutive cells of one row, stored as two 16-byte writes;
//  - running maximum and sum live in registers of the softmax threads, no LDS and no barrier for them;
//  - 32-bit offsets into the KV cache (at most 3.7 M rows), so the address is a scalar base plus one register.
// Device code only, included by prefill_attn.hip.
#pragma once

#include "gpu/attn_wmma1.h"

namespace tona::gpu {
namespace {
namespace fa2 {

using namespace fa;

constexpr int PSTR =
    CC + 4;  // row stride of the partial scores, in floats: eight lanes of a 16-byte write spread over the banks

// LDS of the block. K and V^T share one buffer (K for S, then V for P V).
struct Lds {
    _Float16 *kvbuf, *ps;
    float (*part)[16][PSTR];
    float *arow, *lrow;
};

// The raw bytes a thread stages: K of cell (tid & 31), dimensions (tid >> 5) * 32 ... + 31; V of the cells 2p and
// 2p + 1 (p = tid & 15), dimensions (tid >> 4) * 16 ... + 15. Scales are 0 for a cell past the last one.
struct Raw {
    uint4 k[2], v0, v1;
    float ks, vs0, vs1;
};

// Byte offset of the row of cell id (a missing cell reads row 0: the loads are unconditional, the scale zeroes the
// values). One 32-bit offset from the scalar base of the cache lets the compiler use the saddr form of the load.
__device__ __forceinline__ uint32_t row_off(const FaCtx& c, int id) {
    return __umul24((uint32_t)max(id, 0), (uint32_t)c.row);
}

// Cell id at index ci, -1 past the last cell. The load is clamped to the last index instead of branching.
__device__ __forceinline__ int cell_id(const FaCtx& c, int ci, int n_cells) {
    const uint32_t at = (uint32_t)min(ci, max(n_cells - 1, 0));
    const int v = c.cells ? *reinterpret_cast<const int32_t*>(reinterpret_cast<const char*>(c.cells) + (at << 2)) : ci;
    return ci >= n_cells ? -1 : v;
}

template <class T>
__device__ __forceinline__ T load_at(const FaCtx& c, uint32_t off) {
    return *reinterpret_cast<const T*>(c.kv + off);
}

__device__ __forceinline__ void fetch_k(const FaCtx& c, int id, Raw& w) {
    const uint32_t gq = c.tid >> 5, ro = row_off(c, id);
    const uint32_t ko = ro + c.kvh * 2 * HD + gq * 32;
    w.k[0] = load_at<uint4>(c, ko), w.k[1] = load_at<uint4>(c, ko + 16);
    const float ks = load_at<float>(c, ro + c.nkv * 2 * HD + (c.kvh * 16 + gq) * 4);
    w.ks = id >= 0 ? ks : 0.0f;
}

__device__ __forceinline__ void fetch_v(const FaCtx& c, int id0, int id1, Raw& w) {
    const uint32_t gv = c.tid >> 4, r0 = row_off(c, id0), r1 = row_off(c, id1);
    const uint32_t vo = c.kvh * 2 * HD + HD + gv * 16, so = c.nkv * 2 * HD + (c.kvh * 16 + 8 + (gv >> 1)) * 4;
    w.v0 = load_at<uint4>(c, r0 + vo), w.v1 = load_at<uint4>(c, r1 + vo);
    const float s0 = load_at<float>(c, r0 + so), s1 = load_at<float>(c, r1 + so);
    w.vs0 = id0 >= 0 ? s0 : 0.0f, w.vs1 = id1 >= 0 ? s1 : 0.0f;
}

// Byte b (0 ... 3) of the flipped word wf (sign bits toggled, so the byte is x + 128) times s as f16: the result of
// (_Float16)((float)x * s) with the single rounding of an fma against -128 s (exact)
template <int B>
__device__ __forceinline__ _Float16 scaled(uint32_t wf, float s, float ns) {
    const float x = (float)((wf >> (8 * B)) & 0xffu);  // v_cvt_f32_ubyteN
    return (_Float16)__builtin_fmaf(x, s, ns);
}

__device__ __forceinline__ uint32_t flip(uint32_t w) { return w ^ 0x80808080u; }

// K to f16 in the LDS, thread per (cell, 32 dimensions)
__device__ __forceinline__ void stage_k(const FaCtx& c, _Float16* Ks, const Raw& w) {
    const int cc = c.tid & 31, gq = c.tid >> 5;
    const float ns = -128.0f * w.ks;
#pragma unroll
    for (int q = 0; q < 2; q++) {
        const uint4 u = w.k[q];
        const uint32_t x[4] = {flip(u.x), flip(u.y), flip(u.z), flip(u.w)};
#pragma unroll
        for (int h8 = 0; h8 < 2; h8++) {
            half8 h;
            h[0] = scaled<0>(x[2 * h8], w.ks, ns), h[1] = scaled<1>(x[2 * h8], w.ks, ns);
            h[2] = scaled<2>(x[2 * h8], w.ks, ns), h[3] = scaled<3>(x[2 * h8], w.ks, ns);
            h[4] = scaled<0>(x[2 * h8 + 1], w.ks, ns), h[5] = scaled<1>(x[2 * h8 + 1], w.ks, ns);
            h[6] = scaled<2>(x[2 * h8 + 1], w.ks, ns), h[7] = scaled<3>(x[2 * h8 + 1], w.ks, ns);
            *reinterpret_cast<half8*>(&Ks[cc * LQ + gq * 32 + (2 * q + h8) * 8]) = h;
        }
    }
}

using half2v = _Float16 __attribute__((ext_vector_type(2)));

// V transposed into the LDS, two cells (2p, 2p + 1) per 32-bit store; the dimensions of a thread are gv * 16 + j
__device__ __forceinline__ void stage_v(const FaCtx& c, _Float16* Vt, const Raw& w) {
    const int p = c.tid & 15, gv = c.tid >> 4;
    const float n0 = -128.0f * w.vs0, n1 = -128.0f * w.vs1;
    const uint32_t a[4] = {flip(w.v0.x), flip(w.v0.y), flip(w.v0.z), flip(w.v0.w)};
    const uint32_t b[4] = {flip(w.v1.x), flip(w.v1.y), flip(w.v1.z), flip(w.v1.w)};
#pragma unroll
    for (int j = 0; j < 16; j += 4) {
        const int q = j >> 2;
        half2v h[4];
        h[0] = {scaled<0>(a[q], w.vs0, n0), scaled<0>(b[q], w.vs1, n1)};
        h[1] = {scaled<1>(a[q], w.vs0, n0), scaled<1>(b[q], w.vs1, n1)};
        h[2] = {scaled<2>(a[q], w.vs0, n0), scaled<2>(b[q], w.vs1, n1)};
        h[3] = {scaled<3>(a[q], w.vs0, n0), scaled<3>(b[q], w.vs1, n1)};
#pragma unroll
        for (int i = 0; i < 4; i++) *reinterpret_cast<half2v*>(&Vt[(gv * 16 + j + i) * LV + 2 * p]) = h[i];
    }
}

// S^T = K Q^T of the wave's cell tile and quarter of the dimensions; the lane holds row r16, cells kh ... kh + 7
__device__ __forceinline__ void scores(const FaCtx& c, const Lds& m, const half8 (&qa)[4]) {
    const int ct = c.w & 1, d0 = (c.w >> 1) * 64;
    float8 s = {};
#pragma unroll
    for (int kk = 0; kk < 64; kk += 16) {
        const half8 b = *reinterpret_cast<const half8*>(&m.kvbuf[(ct * 16 + c.r16) * LQ + d0 + kk + c.kh]);
        s = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(b, qa[kk / 16], s);
    }
    float4* dst = reinterpret_cast<float4*>(&m.part[c.w >> 1][c.r16][ct * 16 + c.kh]);
    dst[0] = make_float4(s[0], s[1], s[2], s[3]);
    dst[1] = make_float4(s[4], s[5], s[6], s[7]);
}

// Exchange within the 16 threads of a row: the same sums and maxima as __shfl_xor(v, o, 16), without the LDS round trip
template <int O>
__device__ __forceinline__ float xor16(float v) {
    return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), 0x160 + O, 0xf, 0xf, false));
}

// Online softmax: 16 threads per row, 2 cells per thread. mo and lo, the running maximum and sum of the row, are
// the same in all 16 threads. The rescale factor goes to the LDS for the waves that accumulate.
__device__ __forceinline__ void softmax(const FaCtx& c, const Lds& m, int cn, float& mo, float& lo) {
    const int r = c.tid >> 4, cc = (c.tid & 15) * 2;
    const float2 p0 = *reinterpret_cast<const float2*>(&m.part[0][r][cc]);
    const float2 p1 = *reinterpret_cast<const float2*>(&m.part[1][r][cc]);
    const float2 p2 = *reinterpret_cast<const float2*>(&m.part[2][r][cc]);
    const float2 p3 = *reinterpret_cast<const float2*>(&m.part[3][r][cc]);
    float s2[2] = {cc < cn ? p0.x + p1.x + p2.x + p3.x : -INFINITY,
                   cc + 1 < cn ? p0.y + p1.y + p2.y + p3.y : -INFINITY};
    float mx = fmaxf(mo, fmaxf(s2[0], s2[1]));
    mx = fmaxf(mx, xor16<8>(mx));
    mx = fmaxf(mx, xor16<4>(mx));
    mx = fmaxf(mx, xor16<2>(mx));
    mx = fmaxf(mx, xor16<1>(mx));
    float l = 0;
#pragma unroll
    for (int q = 0; q < 2; q++) {
        const float e = s2[q] > -INFINITY ? __expf(s2[q] - mx) : 0.0f;
        l += e;
        m.ps[r * LV + cc + q] = (_Float16)e;
    }
    l += xor16<8>(l);
    l += xor16<4>(l);
    l += xor16<2>(l);
    l += xor16<1>(l);
    const float al = mo > -INFINITY ? __expf(mo - mx) : 0.0f;
    mo = mx, lo = lo * al + l;
    if ((c.tid & 15) == 0) m.arow[r] = al;
}

// O = alpha * O + P V; wave w: dimensions 32w ... 32w + 31 (two tiles)
__device__ __forceinline__ void accumulate(const FaCtx& c, const Lds& m, float8 (&acc)[2]) {
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

// Gather modes: 0 the V rows are requested after S (as in the first form), 1 with the K rows, 2 the K and V rows of
// the next 32 cells are requested before the current ones compute, 3 only the K rows of the next 32 cells
enum Gather { V_LATE = 0, V_WITH_K = 1, AHEAD = 2, K_AHEAD = 3 };

// The three cell ids a thread needs for the 32 cells from c0: K, and the V pair
struct Ids {
    int k, v0, v1;
};

__device__ __forceinline__ Ids ids_at(const FaCtx& c, int c0, int n_cells) {
    const int p2 = (c.tid & 15) * 2;
    return {cell_id(c, c0 + (c.tid & 31), n_cells), cell_id(c, c0 + p2, n_cells), cell_id(c, c0 + p2 + 1, n_cells)};
}

// What a thread carries through the steps: its Q fragments, the accumulator, running maximum and sum of its row
struct Work {
    half8 qa[4];
    float8 acc[2];
    float mo, lo;
};

// One step of 32 cells on the raw bytes in w. v_late: the V rows are requested between S and the V stores.
template <bool VLate>
__device__ __forceinline__ void step(const FaCtx& c, const Lds& m, Work& k, Raw& w, const Ids& ids, int cn) {
    stage_k(c, m.kvbuf, w);
    lds_sync();
    scores(c, m, k.qa);
    if constexpr (VLate) fetch_v(c, ids.v0, ids.v1, w);
    lds_sync();  // K read: V goes into the same buffer
    stage_v(c, m.kvbuf, w);
    softmax(c, m, cn, k.mo, k.lo);
    lds_sync();
    accumulate(c, m, k.acc);
    lds_sync();
}

template <int G>
__device__ __forceinline__ void run(const FaCtx& c, const Lds& m, Work& k, int n_cells) {
    if constexpr (G >= AHEAD) {
        // Two steps per turn, so the raw bytes of the next step are not copied: A and B change roles. The ids of the
        // step after next are read while a step computes.
        constexpr bool V_NOW = G == AHEAD;
        Ids i0 = ids_at(c, 0, n_cells), i1 = ids_at(c, CC, n_cells);
        Raw a_raw;
        fetch_k(c, i0.k, a_raw);
        if constexpr (V_NOW) fetch_v(c, i0.v0, i0.v1, a_raw);
#pragma unroll 1
        for (int c0 = 0; c0 < n_cells; c0 += 2 * CC) {
            Raw b_raw;
            fetch_k(c, i1.k, b_raw);
            if constexpr (V_NOW) fetch_v(c, i1.v0, i1.v1, b_raw);
            const Ids i2 = ids_at(c, c0 + 2 * CC, n_cells);
            step<!V_NOW>(c, m, k, a_raw, i0, min(CC, n_cells - c0));
            if (c0 + CC >= n_cells) break;
            fetch_k(c, i2.k, a_raw);
            if constexpr (V_NOW) fetch_v(c, i2.v0, i2.v1, a_raw);
            const Ids i3 = ids_at(c, c0 + 3 * CC, n_cells);
            step<!V_NOW>(c, m, k, b_raw, i1, min(CC, n_cells - c0 - CC));
            i0 = i2, i1 = i3;
        }
    } else {
#pragma unroll 1
        for (int c0 = 0; c0 < n_cells; c0 += CC) {
            const Ids ids = ids_at(c, c0, n_cells);
            Raw w;
            fetch_k(c, ids.k, w);
            if constexpr (G == V_WITH_K) fetch_v(c, ids.v0, ids.v1, w);
            step<G == V_LATE>(c, m, k, w, ids, min(CC, n_cells - c0));
        }
    }
}

// acc[j][e]: row kh + e (q head of the group), dimension w * 32 + j * 16 + r16
__device__ __forceinline__ void write_out(const FaCtx& c, const Lds& m, const float8 (&acc)[2], const AttnWmmaArgs& a) {
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

}  // namespace fa2

template <int G>
__global__ void __launch_bounds__(TPB) k_attn_wmma2(AttnWmmaArgs a) {
    using namespace fa2;
    __shared__ __attribute__((aligned(16))) _Float16 KV[CC * LQ > HD * LV ? CC * LQ : HD * LV], Ps[16 * LV];
    __shared__ __attribute__((aligned(16))) float part[4][16][PSTR], arow[16], lrow[16];
    const int t = blockIdx.x / a.nkv, kvh = blockIdx.x % a.nkv, g = a.nh / a.nkv;
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int64_t pos = a.pos0 + t;
    const bool sel = qsa_sel(pos, a.blk, a.topk);
    const int32_t* cells = sel ? a.cells_all + (int64_t)t * MAX_CELLS : nullptr;
    const int n_cells = sel ? a.ncells[t] : (int)(pos + 1);
    Work k;
    k.acc[0] = k.acc[1] = float8{};
    k.mo = -INFINITY, k.lo = 0;
    const int r16 = lane & 15, kh = (lane >> 4) * 8;
    const FaCtx c{a.kv, a.row, cells, a.nkv, g, kvh, t, tid, w, r16, kh};
    const Lds m{KV, Ps, part, arow, lrow};
    fa_load_q(c, a.q, a.nh, a.scale, k.qa);
    run<G>(c, m, k, n_cells);
    if ((tid & 15) == 0) lrow[tid >> 4] = k.lo;
    lds_sync();
    write_out(c, m, k.acc, a);
}

}  // namespace
}  // namespace tona::gpu
