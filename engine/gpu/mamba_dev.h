// Device code of the Mamba-2 state update shared by the megakernel step (mega_op_mamba.h) and the prefill kernels
// (prefill_mamba.hip): both must compute the same bits for the same token, wherever it sits in a pass or a chunk (the
// decode path continues a prefilled state and a rolled-back pass is replayed from stored inputs).
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"

namespace tona::gpu {
namespace {

// State of a row (head h, row i) is N = 128 values over the lanes of a wave (lane + 32 k); a wave holds MB_R rows
constexpr int MB_N = 128;
constexpr int MB_K = MB_N / 32;
constexpr int MB_R = 8;

// softplus as ggml and the oracle (ref::softplus): linear above 20
__device__ __forceinline__ float mb_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

struct MbRows {
    float s[MB_R][MB_K];
};

__device__ __forceinline__ void mb_load(MbRows& r, const float* row0, int lane) {
#pragma unroll
    for (int i = 0; i < MB_R; i++)
#pragma unroll
        for (int k = 0; k < MB_K; k++) r.s[i][k] = row0[(int64_t)i * MB_N + lane + 32 * k];
}

__device__ __forceinline__ void mb_store(const MbRows& r, float* row0, int lane) {
#pragma unroll
    for (int i = 0; i < MB_R; i++)
#pragma unroll
        for (int k = 0; k < MB_K; k++) row0[(int64_t)i * MB_N + lane + 32 * k] = r.s[i][k];
}

// s = s * decay + B * (x * dt): the one formula of the commit and of a token in a pass
__device__ __forceinline__ void mb_update(MbRows& r, float decay, const float (&xdt)[MB_R], const float (&B)[MB_K]) {
#pragma unroll
    for (int i = 0; i < MB_R; i++)
#pragma unroll
        for (int k = 0; k < MB_K; k++) r.s[i][k] = fmaf(B[k], xdt[i], r.s[i][k] * decay);
}

// Lane l of the wave holds the sum of row l / 4 after mb_read; the row j is read from lane 4 j
__device__ __forceinline__ int mb_row_lane(int j) { return 4 * j; }

// v of the lane l ^ bit, bit in {1, 2, 4, 8, 16}: DPP row_xmask inside a row of 16 lanes, permlanex16 across the rows.
// The shuffle of the language goes through ds_bpermute (LDS), which is the latency of the reduction chain.
template <int bit>
__device__ __forceinline__ float mb_xor(float v) {
    if constexpr (bit == 16) {
        return __int_as_float(
            __builtin_amdgcn_permlanex16(0, __float_as_int(v), 0x76543210u, 0xfedcba98u, false, false));
    } else {
        constexpr int kRowXmask = 0x160;  // DPP_ROW_XMASK0
        return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(v), kRowXmask + bit, 0xf, 0xf, false));
    }
}

// One step of the reduction of 2 * n values over the lanes that differ in `bit`: the lane keeps n of them (the upper
// half if its bit is set) and adds the partner's values for them
template <int n, int bit>
__device__ __forceinline__ void mb_halve(float (&p)[2 * n], float (&q)[n]) {
    const bool hi = (threadIdx.x & bit) != 0;
#pragma unroll
    for (int j = 0; j < n; j++) {
        const float keep = hi ? p[j + n] : p[j], send = hi ? p[j] : p[j + n];
        q[j] = keep + mb_xor<bit>(send);
    }
}

// y_i = sum_n s_i[n] C[n] for the 8 rows. Per lane the K values one after the other, then the lanes in a fixed
// order: the rows are split over the lanes while they are summed (4 + 2 + 1 shuffles, then 2 over the rest) instead of
// one wave_sum per row (40 shuffles in a dependent chain per token). Returns the sum of row lane / 4.
__device__ __forceinline__ float mb_read(const MbRows& r, const float (&C)[MB_K]) {
    float p[MB_R], q4[4], q2[2], q1[1];
#pragma unroll
    for (int i = 0; i < MB_R; i++) {
        p[i] = 0;
#pragma unroll
        for (int k = 0; k < MB_K; k++) p[i] = fmaf(r.s[i][k], C[k], p[i]);
    }
    mb_halve<4, 16>(p, q4);
    mb_halve<2, 8>(q4, q2);
    mb_halve<1, 4>(q2, q1);
    float t = q1[0];
    t += mb_xor<2>(t);
    t += mb_xor<1>(t);
    return t;
}

// Where the pending inputs of the last pass lie: decay per head [2][BMAX][heads], x * dt per row [2][BMAX][inner], B
// per group [2][BMAX][groups * N] (the two halves alternate with the parity of the pass)
struct MbPend {
    const float *dec, *xdt, *b;
    int heads, inner, groups, bmax;
};

// Commit of the m accepted tokens of the last pass (half `half`) into the rows (head h, group g, first row r0 = h * P
// + i0 as an index into x * dt), token by token with the formula of a pass
__device__ __forceinline__ void mb_commit(const MbPend& p, MbRows& r, const int (&at)[3], int m, int half) {
    const int xrow = at[0], h = at[1], g = at[2];
    const int lane = threadIdx.x % 32;
    const float* od = p.dec + (int64_t)half * p.bmax * p.heads;
    const float* ox = p.xdt + (int64_t)half * p.bmax * p.inner;
    const float* ob = p.b + (int64_t)half * p.bmax * p.groups * MB_N;
    for (int i = 0; i < m; i++) {
        float xdt[MB_R], B[MB_K];
#pragma unroll
        for (int j = 0; j < MB_R; j++) xdt[j] = ox[(int64_t)i * p.inner + xrow + j];
#pragma unroll
        for (int k = 0; k < MB_K; k++) B[k] = ob[((int64_t)i * p.groups + g) * MB_N + lane + 32 * k];
        mb_update(r, od[i * p.heads + h], xdt, B);
    }
}

// Per token and wave: decay of the head and the x * dt of the rows (dt_raw already scaled by the tensor scale)
struct MbTok {
    float decay, dt;
};
__device__ __forceinline__ MbTok mb_token(float dt_raw, float dt_bias, float A) {
    const float dt = mb_softplus(dt_raw + dt_bias);
    return {expf(dt * A), dt};
}

}  // namespace
}  // namespace tona::gpu
