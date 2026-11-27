// Row functions of the float path: one wave computes the dot product of one row with x, the result is in all lanes.
// The single kernels and the chain use the same functions so that both compute bit-identically.
// Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_dot_sub.h"
#include "gpu/matvec_tables.h"

namespace tona::gpu {
namespace {

// Without staging: each lane reads its own sub-block byte by byte (few useful bytes per load instruction).
template <uint32_t T>
__device__ __forceinline__ float row_direct(const uint8_t* row, const float* xs, int in, int lane) {
    float s = 0;
    for (int sub = lane; sub < in / 32; sub += WAVE) s += dot_sub<T>(row, sub, xs + sub * 32);
    return wave_sum(s);
}

// Via LDS: the wave copies a piece of the row (whole blocks) contiguously with 16-byte accesses into its LDS buffer
// and unpacks there. The source is rounded down to 16 bytes; reads reach up to 15 bytes before and STAGE_PAD bytes
// behind the piece (weight buffers need kMatvecReadPad at their end).
constexpr int STAGE_BYTES = 1024;
constexpr int STAGE_PAD = 32;
template <uint32_t T>
__device__ __forceinline__ float row_staged(const uint8_t* row, const float* xs, int in, int lane, uint8_t* lds) {
    constexpr int bb = blk_bytes<T>(), be = blk_elems<T>();
    constexpr int per_chunk = STAGE_BYTES / bb;
    const int nblocks = in / be;
    float s = 0;
    for (int b0 = 0; b0 < nblocks; b0 += per_chunk) {
        const int nb = min(per_chunk, nblocks - b0);
        const uint8_t* src = row + (size_t)b0 * bb;
        const uintptr_t a = reinterpret_cast<uintptr_t>(src);
        const uint4* s16 = reinterpret_cast<const uint4*>(a & ~(uintptr_t)15);
        const int off = (int)(a & 15), n16 = (off + nb * bb + 15) / 16;
        uint4* d16 = reinterpret_cast<uint4*>(lds);
        for (int i = lane; i < n16; i += WAVE) d16[i] = s16[i];
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        const float* xb = xs + (size_t)b0 * be;
        for (int sub = lane; sub < nb * be / 32; sub += WAVE) s += dot_sub<T>(lds + off, sub, xb + sub * 32);
        __builtin_amdgcn_wave_barrier();  // overwrite the buffer only after all lanes have read it
    }
    return wave_sum(s);
}

#ifndef TONA_MATVEC_STAGED
#define TONA_MATVEC_STAGED \
    0  // measured 2026-10-23 (research/measurements/2026-10-22-matvec.md): not faster, compute-bound
#endif

// One LDS buffer per wave; the kernel declares it once (STAGE_DECL) and passes it down.
constexpr int STAGE_WAVE = STAGE_BYTES + 16 + STAGE_PAD;
#define STAGE_DECL                                                                         \
    __shared__ __attribute__((aligned(16))) uint8_t stage_all[ROWS_PER_BLOCK][STAGE_WAVE]; \
    uint8_t* const stage = stage_all[threadIdx.x / WAVE]

template <uint32_t T>
__device__ __forceinline__ float row_generic(const uint8_t* row, const float* xs, int in, int lane, uint8_t* stage) {
    if constexpr (TONA_MATVEC_STAGED && T != 0 && T != 1 && T != 30)
        return row_staged<T>(row, xs, in, lane, stage);
    else
        return row_direct<T>(row, xs, in, lane);
}

// BF16: each lane reads 16 bytes (8 values) at a time, a wave reads 512 contiguous bytes.
// Load G chunks per lane first, then compute in the same order (bit-identical): the load counters count in order,
// while loading one by one made every lane wait a full latency per chunk. G = 4: pass -2.1 % (1 token), -0.7 % (4),
// -1.1 % (8); G = 8 is worse (research/measurements/2026-10-25-hc-matvec.md).
#ifndef ROWBF16_G
#define ROWBF16_G 4
#endif
__device__ __forceinline__ float row_bf16(const uint8_t* w, const float* x, int in, int lane) {
    constexpr int G = ROWBF16_G;
    const uint4* row = reinterpret_cast<const uint4*>(w);
    const int n_chunks = in / 8;
    float s = 0;
    for (int c0 = lane; c0 < n_chunks; c0 += G * WAVE) {
        uint4 v[G];
        float4 x0[G], x1[G];
#pragma unroll
        for (int g = 0; g < G; g++)
            if (c0 + g * WAVE < n_chunks) {
                const int c = c0 + g * WAVE;
                v[g] = row[c];
                x0[g] = reinterpret_cast<const float4*>(x)[2 * c],
                x1[g] = reinterpret_cast<const float4*>(x)[2 * c + 1];
            }
#pragma unroll
        for (int g = 0; g < G; g++)
            if (c0 + g * WAVE < n_chunks) {
                const float xc[8] = {x0[g].x, x0[g].y, x0[g].z, x0[g].w, x1[g].x, x1[g].y, x1[g].z, x1[g].w};
                const uint32_t u[4] = {v[g].x, v[g].y, v[g].z, v[g].w};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    s += __uint_as_float(u[k] << 16) * xc[2 * k];
                    s += __uint_as_float(u[k] & 0xffff0000u) * xc[2 * k + 1];
                }
            }
    }
    return wave_sum(s);
}

// Q6_K: 16 lanes per super-block; lane l handles positions 2l, 2l+1 in all four parts of both halves (16 values)
// with 16-bit reads. A wave computes two super-blocks at once.
__device__ __forceinline__ float row_q6k(const uint8_t* row, const float* xs, int in, int lane) {
    const int nb = in / 256;
    const int l = 2 * (lane % 16);
    float s = 0;
    for (int sb = lane / 16; sb < nb; sb += 2) {
        const uint8_t* b = row + sb * 210;
        const uint16_t* b16 = reinterpret_cast<const uint16_t*>(b);
        const float d = h2f(b + 208);
        const float* xb = xs + sb * 256;
#pragma unroll
        for (int n = 0; n < 2; n++) {
            const uint16_t ql0 = b16[(64 * n + l) / 2], ql1 = b16[(64 * n + 32 + l) / 2];
            const uint16_t qh = b16[(128 + 32 * n + l) / 2];
            const int8_t* scb = reinterpret_cast<const int8_t*>(b + 192 + 8 * n);
#pragma unroll
            for (int part = 0; part < 4; part++) {
                const int half = l / 16;
                const float sc = scb[2 * part + half];
                float acc = 0;
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const int lo = part & 1 ? (ql1 >> (8 * k)) & 0xff : (ql0 >> (8 * k)) & 0xff;
                    const int nib = part >= 2 ? lo >> 4 : lo & 0xf;
                    const int hi = ((qh >> (8 * k)) >> (2 * part)) & 3;
                    acc += (float)((nib | (hi << 4)) - 32) * xb[128 * n + 32 * part + l + k];
                }
                s += d * sc * acc;
            }
        }
    }
    return wave_sum(s);
}

}  // namespace
}  // namespace tona::gpu
