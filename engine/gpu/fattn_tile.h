// One 16-row tile of full-context attention on WMMA, shared by the decode step (mega_op_fattn.h) and the prefill kernel
// (prefill_fattn.hip): both read the same int8 KV cache and must round the same way, so they share the tile code. A
// tile is 16 query rows against 16 cells at a time: K and V^T go to the LDS as f16, S = Q K^T per wave over a slice of
// the head dimension, an online softmax per row, O += P V per wave over 16 output dimensions. The rows are the q heads
// of one KV group (decode) or tokens times heads (prefill); the caller maps them. Needs 16 waves of 32 lanes. Device
// code only, no mega_ops.h or prefill_dev.h: both define limits of the same names, this header must fit under either.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "gpu/lds_sync.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"

namespace tona::gpu {
namespace {
namespace fa {

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));

constexpr int WAVES = 16, THREADS = 32 * WAVES;
constexpr int CC = 16;                               // cells per step: the N of the WMMA of S and the K of the one of O
constexpr int SCORE_WAVES = 8;                       // waves that compute S, each over HD / 8 dimensions
constexpr int PT_BYTES = SCORE_WAVES * 16 * CC * 4;  // partial scores [wave][row][cell], f32

// Geometry of one head size. The int8 cache row of a position holds, per KV head, K (HD int8), V (HD int8); behind all
// heads the f32 scales, one per 32 values: NS for K, then NS for V, per head.
template <int HD>
struct Geo {
    static_assert(HD == 128 || HD == 256, "head sizes with a tested tile: 128 and 256");
    static constexpr int NS = HD / 32;           // scales per K or V vector
    static constexpr int LQ = HD + 8;            // row stride of K in the LDS, in halves (bank spread)
    static constexpr int DW = HD / SCORE_WAVES;  // dimensions per scoring wave
    static constexpr int STEPS = DW / 16;        // WMMA steps over those
    static constexpr int OUT_WAVES = HD / 16;    // waves that accumulate O, 16 dimensions each
    static constexpr int GROUPS = HD / 8;        // staging threads per cell: 8 dimensions each
    static constexpr int KS_BYTES = CC * LQ * 2, VT_BYTES = HD * CC * 2, PS_BYTES = 16 * CC * 2;
    static constexpr bool PT_OVER_KS = KS_BYTES >= PT_BYTES;  // S is written after K is read, over its space
    static constexpr int BYTES = KS_BYTES + VT_BYTES + PS_BYTES + 48 * 4 + (PT_OVER_KS ? 0 : PT_BYTES);
};

// Views into the LDS. mrow, lrow, arow: running maximum, sum and rescale factor of each of the 16 rows.
template <int HD>
struct Lds {
    _Float16 *Ks, *Vt, *Ps;
    float *pt, *mrow, *lrow, *arow;
};

template <int HD>
__device__ __forceinline__ Lds<HD> lds_of(float* sm) {
    using G = Geo<HD>;
    Lds<HD> l;
    l.Ks = reinterpret_cast<_Float16*>(sm);
    l.Vt = l.Ks + CC * G::LQ;
    l.Ps = l.Vt + HD * CC;
    l.mrow = reinterpret_cast<float*>(l.Ps + 16 * CC);
    l.lrow = l.mrow + 16;
    l.arow = l.lrow + 16;
    l.pt = G::PT_OVER_KS ? reinterpret_cast<float*>(l.Ks) : l.arow + 16;
    return l;
}

// The int8 KV cache of one KV head: row bytes per position, nkv heads per row. Split by position (kv_place.h): the
// cells [0, hot) are in kv, the cells from hot on in cold (its cell hot first); no cold: all of them in kv.
struct KvHead {
    const uint8_t* kv;
    int64_t row;
    int nkv, kvh;
    const uint8_t* cold = nullptr;
    int64_t hot = INT64_MAX;
};

// The row of a position. cold - hot * row is computed once per tile by the compiler; no cold: hot is INT64_MAX and the
// select folds away where the head is built from constants (the prefill)
__device__ __forceinline__ const uint8_t* kv_row_at(const KvHead& k, int64_t cell) {
    return cell < k.hot ? k.kv + cell * k.row : k.cold + (cell - k.hot) * k.row;
}

// Dimensions [w * DW + 16 k2 + kh, +8) of the q row r16 (null: a row past the group, zero) as f16, with the scale
template <int HD>
__device__ __forceinline__ void load_q(const float* qr, float scale, half8 (&qa)[Geo<HD>::STEPS], int lane) {
    using G = Geo<HD>;
    const int w = threadIdx.x / 32, kh = (lane >> 4) * 8;
    if (w >= SCORE_WAVES) return;
#pragma unroll
    for (int k2 = 0; k2 < G::STEPS; k2++)
#pragma unroll
        for (int j = 0; j < 8; j++)
            qa[k2][j] = qr ? (_Float16)(qr[w * G::DW + 16 * k2 + kh + j] * scale) : (_Float16)0.0f;
}

// K and V^T of the cells [c0, c0 + cn) into the LDS, one thread per (cell, 8 dimensions), dequantized to f16. Cells
// past cn are zero. Cell fastest in the thread index: the transposed V writes of 16 threads go to 16 neighbours.
template <int HD>
__device__ __forceinline__ void stage(const KvHead k, const Lds<HD> l, int64_t c0, int cn) {
    using G = Geo<HD>;
    const int cc = threadIdx.x & 15, gq = threadIdx.x >> 4;
    if (gq >= G::GROUPS) return;
    half8 kh8 = {}, vh8 = {};
    if (cc < cn) {
        const uint8_t* r = kv_row_at(k, c0 + cc);
        const int8_t* kc = (const int8_t*)r + (int64_t)k.kvh * 2 * HD + gq * 8;
        const float* sc = (const float*)(r + (int64_t)k.nkv * 2 * HD) + k.kvh * 2 * G::NS;
        const float ks = sc[gq / 4], vs = sc[G::NS + gq / 4];
        const uint2 k8 = *reinterpret_cast<const uint2*>(kc);
        const uint2 v8 = *reinterpret_cast<const uint2*>(kc + HD);
        const int8_t* kb = (const int8_t*)&k8;
        const int8_t* vb = (const int8_t*)&v8;
#pragma unroll
        for (int j = 0; j < 8; j++) kh8[j] = (_Float16)((float)kb[j] * ks), vh8[j] = (_Float16)((float)vb[j] * vs);
    }
    *reinterpret_cast<half8*>(&l.Ks[cc * G::LQ + gq * 8]) = kh8;
#pragma unroll
    for (int j = 0; j < 8; j++) l.Vt[(gq * 8 + j) * CC + cc] = vh8[j];
}

// Partial scores S = Q K^T of the waves 0-7 (HD / 8 dimensions each) into pt; the barrier lets pt reuse Ks
template <int HD>
__device__ __forceinline__ void scores(const half8 (&qa)[Geo<HD>::STEPS], const Lds<HD> l, int lane) {
    using G = Geo<HD>;
    const int w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    float8 sv = {};
    if (w < SCORE_WAVES) {
#pragma unroll
        for (int k2 = 0; k2 < G::STEPS; k2++) {
            const half8 bb = *reinterpret_cast<const half8*>(&l.Ks[r16 * G::LQ + w * G::DW + 16 * k2 + kh]);
            sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[k2], bb, sv);
        }
    }
    if (G::PT_OVER_KS) lds_sync();  // Ks read, pt overwrites it
    if (w < SCORE_WAVES) {
#pragma unroll
        for (int e = 0; e < 8; e++) l.pt[(w * 16 + kh + e) * CC + r16] = sv[e];
    }
}

// Online softmax, wave w = row w, lane = cell. valid: cells of this chunk the row may see (causal mask), 0 .. cn.
template <int HD>
__device__ __forceinline__ void softmax(const Lds<HD> l, int valid, int lane) {
    const int w = threadIdx.x / 32;
    float v = -INFINITY;
    if (lane < valid) {
        v = 0;
#pragma unroll
        for (int p8 = 0; p8 < SCORE_WAVES; p8++) v += l.pt[(p8 * 16 + w) * CC + lane];
    }
    const float mo = l.mrow[w], mx = fmaxf(mo, wave_max(v));
    const float e = v > -INFINITY ? __expf(v - mx) : 0.0f;
    if (lane < CC) l.Ps[w * CC + lane] = (_Float16)e;
    const float sum = wave_sum(e);
    if (lane == 0) {
        const float al = mo > -INFINITY ? __expf(mo - mx) : 0.0f;
        l.mrow[w] = mx, l.lrow[w] = l.lrow[w] * al + sum, l.arow[w] = al;
    }
}

// O += P V for the dimensions w * 16 ... of the waves < HD / 16, after rescaling by the softmax factors
template <int HD>
__device__ __forceinline__ void accumulate(float8& acc, const Lds<HD> l, int lane) {
    const int w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    if (w >= Geo<HD>::OUT_WAVES) return;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] *= l.arow[kh + e];
    const half8 a = *reinterpret_cast<const half8*>(&l.Ps[r16 * CC + kh]);
    const half8 bb = *reinterpret_cast<const half8*>(&l.Vt[(w * 16 + r16) * CC + kh]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc);
}

// Sets the running state of the 16 rows to "nothing seen"
__device__ __forceinline__ void reset_rows(float* mrow, float* lrow) {
    if (threadIdx.x < 16) mrow[threadIdx.x] = -INFINITY, lrow[threadIdx.x] = 0;
}

// What a tile works on: the cache of its KV head, its LDS and the lane of the thread
template <int HD>
struct Tile {
    KvHead k;
    Lds<HD> l;
    int lane;
};

// One step over the cells [c0, c0 + cn): stage, scores, softmax, accumulate. valid_w: cells of this chunk the row of
// this wave sees. The barriers: LDS free before staging, staged before the scores, scores before the softmax, softmax
// before the accumulation.
template <int HD>
__device__ __forceinline__ void step(const Tile<HD> t, const half8 (&qa)[Geo<HD>::STEPS], float8& acc, int64_t c0,
                                     int cn, int valid_w) {
    lds_sync();
    stage<HD>(t.k, t.l, c0, cn);
    lds_sync();
    scores<HD>(qa, t.l, t.lane);
    lds_sync();
    softmax<HD>(t.l, valid_w, t.lane);
    lds_sync();
    accumulate<HD>(acc, t.l, t.lane);
}

}  // namespace fa
}  // namespace
}  // namespace tona::gpu
