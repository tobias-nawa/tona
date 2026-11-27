// int8 path: unpacking of the grid formats (IQ2_XXS, IQ2_S, IQ3_XXS, IQ3_S) into a Dec. Magnitudes come from the grid
// tables (Tabs), the signs from a byte mask. Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_q8_core.h"

#ifndef TONA_Q8_RAW16
#define TONA_Q8_RAW16 1
#endif

namespace tona::gpu {
namespace {

// IQ2_XXS: sub-block ib of a block of 256 is 8 bytes at 2 + 8 ib: four grid indices, then four 7-bit sign indices and
// a 4-bit scale (bits 28..31). The grid comes from LDS (load_iq2xxs_lds), the sign masks from the IQ3_XXS table. Same
// float steps as the CPU row functions (cpu/experts_decode.h, scales8<16>): d / 8 times the odd integer 1 + 2 s.
// Aligned words and a byte shift (one 12-byte load) instead of four 2-byte loads were measured and gave nothing
// (118 against 127 GB/s at 1 token, research/measurements/2026-11-19-bolt-b2-b3.md)
// The bytes of a sub-block as they lie in the file: the pipelined row loop (Raw<T>::kOn) requests them before it
// unpacks the sub-blocks of the round in hand, so twice as many weight bytes are in flight per lane
template <>
struct Raw<16> {
    static constexpr bool kOn = TONA_Q8_RAW16 != 0;
    uint32_t q4, aux, d;
};
__device__ __forceinline__ void load_raw(const uint8_t* row, int sub, Raw<16>& r) {
    const uint8_t* b = row + (sub / 8) * 66;
    const int ib = sub % 8;
    r.d = u16(b);
    r.q4 = (uint32_t)ld4_a2(b + 2 + 8 * ib);
    r.aux = (uint32_t)ld4_a2(b + 6 + 8 * ib);
}
__device__ __forceinline__ void decode_raw(const Raw<16>& r, const Tabs& tb, Dec& o) {
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const uint64_t g = lds_iq2xxs[(r.q4 >> 8 * l) & 0xff];
        const uint64_t mk = tb.m3[(r.aux >> 7 * l) & 127];
        o.w[2 * l] = apply_sign((uint32_t)g, (uint32_t)mk);
        o.w[2 * l + 1] = apply_sign((uint32_t)(g >> 32), (uint32_t)(mk >> 32));
    }
    o.s_lo = o.s_hi = h2f_bits(r.d) * 0.125f * (float)(1 + 2 * (int)(r.aux >> 28));
}
template <>
__device__ __forceinline__ void decode_q8<16>(const uint8_t* row, int sub, const Tabs& tb, Dec& o) {
    Raw<16> r;
    load_raw(row, sub, r);
    decode_raw(r, tb, o);
}
template <>
__device__ __forceinline__ void decode_q8<22>(const uint8_t* row, int sub, const Tabs& tb, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 82;
    const int ib = sub % 8;
    const int qs4 = ld4_a2(b + 2 + 4 * ib), sg4 = ld4_a2(b + 34 + 4 * ib);
    const int qh = b[66 + ib], sc = b[74 + ib];
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const int idx = ((qs4 >> 8 * l) & 0xff) | ((qh << (8 - 2 * l)) & 0x300);
        const uint64_t g = tb.iq2s[idx];
        const uint32_t sb = (sg4 >> 8 * l) & 0xff;
        o.w[2 * l] = apply_sign((uint32_t)g, sign_mask4(sb));
        o.w[2 * l + 1] = apply_sign((uint32_t)(g >> 32), sign_mask4(sb >> 4));
    }
    const float d = h2f(b) * 0.125f;
    o.s_lo = d * (float)(1 + 2 * (sc & 0xf)), o.s_hi = d * (float)(1 + 2 * (sc >> 4));
}
template <>
__device__ __forceinline__ void decode_q8<18>(const uint8_t* row, int sub, const Tabs& tb, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 98;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint32_t aux = (uint32_t)ld4_a2(b + 66 + 4 * ib);
    const int q0 = ld4_a2(qs), q1 = ld4_a2(qs + 4);
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const int qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const uint32_t g1 = tb.iq3xxs[(qq >> sh) & 0xff], g2 = tb.iq3xxs[(qq >> (sh + 8)) & 0xff];
        const uint64_t mk = tb.m3[(aux >> 7 * l) & 127];
        o.w[2 * l] = apply_sign(g1, (uint32_t)mk);
        o.w[2 * l + 1] = apply_sign(g2, (uint32_t)(mk >> 32));
    }
    o.s_lo = o.s_hi = h2f(b) * 0.25f * (float)(1 + 2 * (int)(aux >> 28));
}
template <>
__device__ __forceinline__ void decode_q8<21>(const uint8_t* row, int sub, const Tabs& tb, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 110;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const int qh = b[66 + ib];
    const int sg4 = ld4_a2(b + 74 + 4 * ib);
    const int sc = (b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf;
    const int q0 = ld4_a2(qs), q1 = ld4_a2(qs + 4);
#pragma unroll
    for (int l = 0; l < 4; l++) {
        const int qq = l < 2 ? q0 : q1, sh = 16 * (l & 1);
        const int i1 = ((qq >> sh) & 0xff) | ((qh << (8 - 2 * l)) & 256);
        const int i2 = ((qq >> (sh + 8)) & 0xff) | ((qh << (7 - 2 * l)) & 256);
        const uint32_t sb = (sg4 >> 8 * l) & 0xff;
        o.w[2 * l] = apply_sign(tb.iq3s[i1], sign_mask4(sb));
        o.w[2 * l + 1] = apply_sign(tb.iq3s[i2], sign_mask4(sb >> 4));
    }
    o.s_lo = o.s_hi = h2f(b) * (float)(1 + 2 * sc);
}

}  // namespace
}  // namespace tona::gpu
