// Prefill attention with one 16-row tile per wave (prefill_fattn.hip): the work of fattn_tile.h without the split of
// the head dimension over waves. The tile computes S^T = K Q^T (cells x rows) instead of S = Q K^T, so the WMMA result
// has one row of the tile per lane (lane % 16) and eight cells per lane: the row maximum and sum need one exchange
// between the two half waves, and the probabilities are already in the layout of the B operand of O^T = V^T P^T. The
// partial scores, the scores, the softmax and the accumulation stay in registers; the waves of a block share only the
// staged K and V^T. Same operations per row in the same order as fattn_tile.h (partial sums over HD / 8 dimensions
// added in turn, 16 cells per softmax step, the same sum tree), so the result is bit-identical to it. Device code only.
#pragma once

#include "gpu/fattn_tile.h"

namespace tona::gpu {
namespace {
namespace fw {

using fa::float8;
using fa::half8;

// Geometry of one head size and chunk size. A chunk is CCB cells (a multiple of 16) staged for all waves of the block.
template <int HD, int CCB>
struct Geo {
    static_assert(HD == 128 || HD == 256, "head sizes with a tested tile: 128 and 256");
    static_assert(CCB % 16 == 0, "the softmax step is 16 cells");
    static constexpr int NQ = HD / 16;                       // WMMA steps over the head dimension
    static constexpr int STEPS = HD / fa::SCORE_WAVES / 16;  // steps of one partial sum of the score
    static constexpr int LQ = HD + 8;                        // row stride of K in the LDS, in halves
    static constexpr int VS = CCB + 8;                       // row stride of V^T in the LDS, in halves
    static constexpr int KS_BYTES = CCB * LQ * 2, VT_BYTES = HD * VS * 2;
    static constexpr int BYTES = KS_BYTES + VT_BYTES;
};

// The registers of one wave: its Q (B operand of S^T), the accumulator O^T per 16 dimensions, running max and sum of
// the row lane % 16
template <int HD>
struct State {
    half8 qa[HD / 16];
    float8 acc[HD / 16];
    float m, l;
};

// Q of the row of this lane (null: a row past the tokens, zeros), all HD dimensions, as f16 with the scale
template <int HD>
__device__ __forceinline__ void load_q(State<HD>& s, const float* qr, float scale, int lane) {
    const int kh = (lane >> 4) * 8;
#pragma unroll
    for (int i = 0; i < HD / 16; i++)
#pragma unroll
        for (int j = 0; j < 8; j++) s.qa[i][j] = qr ? (_Float16)(qr[16 * i + kh + j] * scale) : (_Float16)0.0f;
#pragma unroll
    for (int i = 0; i < HD / 16; i++) s.acc[i] = float8{};
    s.m = -INFINITY, s.l = 0;
}

// K and V^T of the cells [c0, c0 + CCB) into the LDS (as fa::stage), the cells past cn zero
template <int HD, int CCB, int THREADS>
__device__ __forceinline__ void stage(const fa::KvHead k, _Float16* Ks, _Float16* Vt, int64_t c0, int cn) {
    using G = Geo<HD, CCB>;
    constexpr int GROUPS = HD / 8, ITEMS = CCB * GROUPS;
    static_assert(ITEMS % THREADS == 0, "every thread stages the same number of items");
#pragma unroll
    for (int i = 0; i < ITEMS / THREADS; i++) {
        const int it = threadIdx.x + i * THREADS;
        const int cc = it % CCB, gq = it / CCB;
        half8 kh8 = {}, vh8 = {};
        if (cc < cn) {
            const uint8_t* r = fa::kv_row_at(k, c0 + cc);
            const int8_t* kc = (const int8_t*)r + (int64_t)k.kvh * 2 * HD + gq * 8;
            const float* sc = (const float*)(r + (int64_t)k.nkv * 2 * HD) + k.kvh * 2 * (HD / 32);
            const float ks = sc[gq / 4], vs = sc[HD / 32 + gq / 4];
            const uint2 k8 = *reinterpret_cast<const uint2*>(kc);
            const uint2 v8 = *reinterpret_cast<const uint2*>(kc + HD);
            const int8_t* kb = (const int8_t*)&k8;
            const int8_t* vb = (const int8_t*)&v8;
#pragma unroll
            for (int j = 0; j < 8; j++) kh8[j] = (_Float16)((float)kb[j] * ks), vh8[j] = (_Float16)((float)vb[j] * vs);
        }
        *reinterpret_cast<half8*>(&Ks[cc * G::LQ + gq * 8]) = kh8;
#pragma unroll
        for (int j = 0; j < 8; j++) Vt[(gq * 8 + j) * G::VS + cc] = vh8[j];
    }
}

// Scores of the 16 cells at sub (sub-chunk of the staged chunk) for the row of this lane: the partial sums over
// HD / 8 dimensions each, added in turn (the order of fa::softmax)
template <int HD, int CCB>
__device__ __forceinline__ float8 scores(const State<HD>& s, const _Float16* Ks, int sub, int lane) {
    using G = Geo<HD, CCB>;
    const int r16 = lane & 15, kh = (lane >> 4) * 8;
    float8 v = {};
#pragma unroll
    for (int p = 0; p < fa::SCORE_WAVES; p++) {
        float8 sv = {};
#pragma unroll
        for (int k2 = 0; k2 < G::STEPS; k2++) {
            const int i = p * G::STEPS + k2;
            const half8 kb = *reinterpret_cast<const half8*>(&Ks[(sub * 16 + r16) * G::LQ + 16 * i + kh]);
            sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(kb, s.qa[i], sv);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) v[e] += sv[e];
    }
    return v;
}

// v of the lane l ^ 16 (the other half wave): permlanex16, not the shuffle of the language, which goes through
// ds_bpermute (the LDS queue and its latency inside the softmax chain; same idea as mb_xor in mamba_dev.h)
__device__ __forceinline__ float xor16(float v) {
    return __int_as_float(__builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
}

// Online softmax of one step for the row of this lane. v: scores of cells kh + e; valid: cells of the step the row may
// see. Returns the probabilities as the B operand of O^T and the factor that rescales the accumulator.
__device__ __forceinline__ half8 softmax(float8 v, int valid, float& m, float& l, float& al, int lane) {
    const int kh = (lane >> 4) * 8;
    float mx = -INFINITY;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        if (kh + e >= valid) v[e] = -INFINITY;
        mx = fmaxf(mx, v[e]);
    }
    mx = fmaxf(m, fmaxf(mx, xor16(mx)));
    float8 p;
#pragma unroll
    for (int e = 0; e < 8; e++) p[e] = v[e] > -INFINITY ? __expf(v[e] - mx) : 0.0f;
    // the tree of wave_sum over cells 0..15 (lane = cell, xor offsets 8, 4, 2, 1; offset 16 adds zeros)
    float8 t, u, w;
#pragma unroll
    for (int e = 0; e < 8; e++) t[e] = p[e] + xor16(p[e]);
#pragma unroll
    for (int e = 0; e < 8; e++) u[e] = t[e] + t[e ^ 4];
#pragma unroll
    for (int e = 0; e < 8; e++) w[e] = u[e] + u[e ^ 2];
    const float sum = w[0] + w[1];
    al = m > -INFINITY ? __expf(m - mx) : 0.0f;
    m = mx, l = l * al + sum;
    half8 pb;
#pragma unroll
    for (int e = 0; e < 8; e++) pb[e] = (_Float16)p[e];
    return pb;
}

// O^T += V^T P^T over the 16 cells at sub, after the rescale
template <int HD, int CCB>
__device__ __forceinline__ void accumulate(State<HD>& s, const _Float16* Vt, int sub, half8 pb, float al, int lane) {
    using G = Geo<HD, CCB>;
    const int r16 = lane & 15, kh = (lane >> 4) * 8;
    // x * 1.0f == x exactly: a step in which no row's maximum moved skips the multiplies (wave-uniform branch)
    if (__any(al != 1.0f)) {
#pragma unroll
        for (int d = 0; d < HD / 16; d++)
#pragma unroll
            for (int e = 0; e < 8; e++) s.acc[d][e] *= al;
    }
#pragma unroll
    for (int d = 0; d < HD / 16; d++) {
        const half8 va = *reinterpret_cast<const half8*>(&Vt[(d * 16 + r16) * G::VS + sub * 16 + kh]);
        s.acc[d] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(va, pb, s.acc[d]);
    }
}

}  // namespace fw
}  // namespace
}  // namespace tona::gpu
