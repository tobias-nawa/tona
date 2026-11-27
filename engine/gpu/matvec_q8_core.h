// int8 path, shared parts: dot4 instructions, unpacked sub-block (Dec), grid tables and the multiply-accumulate of
// an unpacked sub-block with x. Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec.h"
#include "gpu/matvec_tables.h"

namespace tona::gpu {
namespace {

#ifndef TONA_Q8_LDS_TABLES
#define TONA_Q8_LDS_TABLES 1
#endif

__device__ __forceinline__ int dot4u(int a_u8, int b_s8, int c) {
    return __builtin_amdgcn_sudot4(false, a_u8, true, b_s8, c, false);
}
__device__ __forceinline__ int dot4s(int a_s8, int b_s8, int c) {
    return __builtin_amdgcn_sudot4(true, a_s8, true, b_s8, c, false);
}
// 4 bytes from an address that is only 2-byte aligned
__device__ __forceinline__ int ld4_a2(const uint8_t* p) {
    const uint16_t* q = reinterpret_cast<const uint16_t*>(p);
    return (int)((uint32_t)q[0] | ((uint32_t)q[1] << 16));
}
__device__ __forceinline__ int ld4(const uint8_t* p) { return *reinterpret_cast<const int*>(p); }

// 4 nibbles (0..15 per byte) to int8 through a 16-entry table held in registers (t0..t3)
__device__ __forceinline__ int lut16(int q4, int t0, int t1, int t2, int t3) {
    const uint32_t sel = (uint32_t)q4 & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm((uint32_t)t1, (uint32_t)t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm((uint32_t)t3, (uint32_t)t2, sel);
    // selector 12 yields 0x00, 13 yields 0xFF: mask from bit 3 of each nibble
    const uint32_t m = __builtin_amdgcn_perm(0u, 0u, (((uint32_t)q4 >> 3) & 0x01010101u) | 0x0C0C0C0Cu);
    return (int)((hi & m) | (lo & ~m));
}

// Sum of +-g*x for 4 unsigned values g with byte mask m (0xFF = negative)
__device__ __forceinline__ int dot4_signed_mask(int g, int m, int x) { return dot4u(g, x, 0) - 2 * dot4u(g & m, x, 0); }

// Tables of the grid formats: in the kernels from LDS (see k_matvec_q8), otherwise from device memory
struct Tabs {
    const uint64_t* iq2s;
    const uint32_t* iq3xxs;
    const uint32_t* iq3s;
    const uint64_t* m8;
    const uint64_t* m3;
};

// Sub-block sub (32 values) unpacked: 8 words of 4 int8 weights (unsigned if q8_unsigned<T>), a scale per 16-value
// half and an offset per half that is multiplied by the sum of the x values of that half:
//   sum w*x = d_x * (s_lo * dot(w[0..3], x[0..3]) + s_hi * dot(w[4..7], x[4..7]) + b_lo * sum(x_lo) + b_hi * sum(x_hi))
// Unpacked once, then multiplied with B vectors x (verify pass of a draft).
struct Dec {
    int w[8];
    float s_lo, s_hi, b_lo = 0, b_hi = 0;
};
template <uint32_t T>
constexpr bool q8_unsigned() {
    return T == 12 || T == 13 || T == 14 || T == 42 || T == 1019;
}
// Apply the sign by byte mask (0xFF = negative); valid for bytes 1..127 (grid values are never 0)
__device__ __forceinline__ int apply_sign(uint32_t g, uint32_t m) { return (int)((g ^ m) + (m & 0x01010101u)); }

// Mask bytes (0xFF where the bit is set) of the four low bits of s, as byte j of d_kmask8[s] for j < 4, without the
// table in LDS: the bits are spread one per byte by a multiplication, then v_perm turns 0 / 1 into 0x00 / 0xFF
// (selector 12 / 13)
__device__ __forceinline__ uint32_t sign_mask4(uint32_t s) {
    const uint32_t spread = ((s & 0xFu) * 0x204081u) & 0x01010101u;
    return __builtin_amdgcn_perm(0u, 0u, spread | 0x0C0C0C0Cu);
}

template <uint32_t T>
__device__ void decode_q8(const uint8_t* row, int sub, const Tabs& tb, Dec& o);

// Software pipeline of row_q8: the raw bytes of the next sub-blocks are requested before the current ones are unpacked
// and multiplied, so twice as many weight bytes are in flight per lane. Raw<T>::kOn: the format has load_raw and
// decode_raw (IQ4_NL only: moe.down -0.3 ms per pass at 4 vectors; IQ2_S and IQ3_S gained in the
// probe but cost +0.4 ms in the engine, research/measurements/2026-11-12-verify-pass.md).
template <uint32_t T>
struct Raw {
    static constexpr bool kOn = false;
};

// Unpacked sub-block times x (int8, 8 words) with the metadata of x, added to s. Explicit fma: the CPU experts
// (cpu/experts.cpp) compute the same sequence bit-identically.
template <uint32_t T>
__device__ __forceinline__ float apply_q8(const Dec& o, const int* xi, const Q8Meta& m, float s) {
    int lo = 0, hi = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        if constexpr (q8_unsigned<T>())
            lo = dot4u(o.w[k], xi[k], lo), hi = dot4u(o.w[k + 4], xi[k + 4], hi);
        else
            lo = dot4s(o.w[k], xi[k], lo), hi = dot4s(o.w[k + 4], xi[k + 4], hi);
    }
    float r = fmaf(o.s_hi, (float)hi, o.s_lo * (float)lo);
    if constexpr (q8_unsigned<T>()) r = fmaf(o.b_hi, (float)m.s1, fmaf(o.b_lo, (float)m.s0, r));
    return fmaf(m.d, r, s);
}

}  // namespace
}  // namespace tona::gpu
