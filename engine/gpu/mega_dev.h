// Device code shared by the megakernel (gpu/arch/*/kernel.hip) and prefill (gpu/prefill.hip): activations, RoPE with
// norm, int8 of the KV cache. Both must compute the same here (KV cache and indexer pass from prefill to the decode
// path). Include after gpu/matvec_dev.h (WAVE, wave_sum).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "cpu/detmath.h"
#include "gpu/yarn_dev.h"

namespace tona::gpu {
namespace {

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }

// A logit as a selection sees it: NaN -> -FLT_MAX, +-inf -> +-FLT_MAX, finite values unchanged (host twin:
// backend/nan_guard.h clean_one). Keeps indices valid and avoids inf - inf in the softmax.
__device__ __forceinline__ float logit_clamp(float v) {
    return v != v ? -__FLT_MAX__ : fminf(fmaxf(v, -__FLT_MAX__), __FLT_MAX__);
}

__device__ __forceinline__ float wave_max(float v) {
    for (int off = WAVE / 2; off > 0; off /= 2) v = fmaxf(v, __shfl_xor(v, off, WAVE));
    return v;
}

// NeoX RoPE of a pair (i, i + n_rot/2), angle in double as in ops.hip
__device__ __forceinline__ void rope_pair(float& x0, float& x1, int i, int n_rot, int64_t pos, float base) {
    const double theta = (double)pos * pow((double)base, -2.0 * i / n_rot);
    const float c = (float)cos(theta), s = (float)sin(theta);
    const float a = x0, b = x1;
    x0 = a * c - b * s;
    x1 = a * s + b * c;
}

// The same for the 64 rotated dimensions of the QSA layers and the MTP layer, with the YaRN table (gpu/yarn_dev.h)
__device__ __forceinline__ void rope_pair_yarn(float& x0, float& x1, int i, int64_t pos, float base) {
    float c, s;
    rope_cs(i, 64, pos, base, c, s);
    const float a = x0, b = x1;
    x0 = a * c - b * s;
    x1 = a * s + b * c;
}

// One wave: RMSNorm ⊙ w over NV*32 values, then RoPE on the first 64 (pairs lane, lane + 32), src -> dst
template <int NV>
__device__ __forceinline__ void wave_norm_rope(const float* src, float* dst, const float* w, int64_t pos, float base,
                                               float eps) {
    const int lane = threadIdx.x % WAVE;
    float v[NV], ss = 0;
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] = src[lane + 32 * k], ss += v[k] * v[k];
    const float scale = rsqrtf(wave_sum(ss) / (NV * 32) + eps);
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] *= scale * w[lane + 32 * k];
    rope_pair_yarn(v[0], v[1], lane, pos, base);
#pragma unroll
    for (int k = 0; k < NV; k++) dst[lane + 32 * k] = v[k];
}

// 256 values of a wave (lane l holds l + 32k) -> int8 with a scale per 32 (group k = v[k] of all lanes)
__device__ __forceinline__ void wave_q8_256(const float (&v)[8], int8_t* q, float* scale, int lane) {
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const float m = wave_max(fabsf(v[k]));
        const float sc = m > 0 ? m / 127.0f : 1.0f;
        q[lane + 32 * k] = (int8_t)__float2int_rn(v[k] / sc);
        if (lane == k) scale[k] = sc;
    }
}

// Copies one row of `bytes` bytes with the threads first, first + step, ... Rows of a block type whose size is no
// multiple of 4 (Q6_K with 256 columns: 210 bytes) sit at 2-byte offsets in a packed buffer, so only a row size
// divisible by 4 may use 32-bit accesses (then every row start is 4-aligned too).
__device__ __forceinline__ void copy_row(uint8_t* dst, const uint8_t* src, int bytes, int first, int step) {
    if (bytes % 4 == 0) {
        const auto* s = reinterpret_cast<const uint32_t*>(src);
        auto* d = reinterpret_cast<uint32_t*>(dst);
        for (int k = first; k < bytes / 4; k += step) d[k] = s[k];
    } else {
        for (int k = first; k < bytes; k += step) dst[k] = src[k];
    }
}

}  // namespace
}  // namespace tona::gpu
