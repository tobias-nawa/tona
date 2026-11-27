// int8 path: unpacking of the byte-oriented and K-quant formats (Q8_0, Q2_0, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS) into
// a Dec. Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_q8_core.h"

namespace tona::gpu {
namespace {

template <>
__device__ __forceinline__ void decode_q8<8>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + sub * 34;
#pragma unroll
    for (int k = 0; k < 8; k++) o.w[k] = ld4_a2(b + 2 + 4 * k);
    o.s_lo = o.s_hi = h2f(b);
}
template <>
__device__ __forceinline__ void decode_q8<42>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    // Q2_0: blocks of 64 in 18 bytes, value = q - 1 with q in 0..3, four values per byte
    const uint8_t* b = row + (sub / 2) * 18;
    const uint8_t* q = b + 2 + (sub & 1) * 8;
    const uint32_t q0 = (uint32_t)ld4_a2(q), q1 = (uint32_t)ld4_a2(q + 4);
#pragma unroll
    for (int k = 0; k < 8; k++) {
        const uint32_t v = ((k < 4 ? q0 : q1) >> (8 * (k & 3))) & 0xff;
        o.w[k] = (int)((v & 3) | ((v >> 2 & 3) << 8) | ((v >> 4 & 3) << 16) | ((v >> 6 & 3) << 24));
    }
    o.s_lo = o.s_hi = h2f(b);
    o.b_lo = o.b_hi = -o.s_lo;
}
template <>
__device__ __forceinline__ void decode_q8<12>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 144;
    const int ib = sub % 8, jj = ib / 2, sh = (ib & 1) * 4;
    int sc, mn;
    scale_min_k4(ib, b + 4, sc, mn);
    const uint4* q = reinterpret_cast<const uint4*>(b + 16 + 32 * jj);
    const uint4 a = q[0], c = q[1];
    const int w[8] = {(int)a.x, (int)a.y, (int)a.z, (int)a.w, (int)c.x, (int)c.y, (int)c.z, (int)c.w};
#pragma unroll
    for (int k = 0; k < 8; k++) o.w[k] = (w[k] >> sh) & 0x0F0F0F0F;
    o.s_lo = o.s_hi = h2f(b) * sc;
    o.b_lo = o.b_hi = -h2f(b + 2) * mn;
}
template <>
__device__ __forceinline__ void decode_q8<13>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 176;
    const int ib = sub % 8, jj = ib / 2, sh = (ib & 1) * 4;
    int sc, mn;
    scale_min_k4(ib, b + 4, sc, mn);
    const uint4* h = reinterpret_cast<const uint4*>(b + 16);
    const uint4* q = reinterpret_cast<const uint4*>(b + 48 + 32 * jj);
    const uint4 h0 = h[0], h1 = h[1], a = q[0], c = q[1];
    const int hh[8] = {(int)h0.x, (int)h0.y, (int)h0.z, (int)h0.w, (int)h1.x, (int)h1.y, (int)h1.z, (int)h1.w};
    const int w[8] = {(int)a.x, (int)a.y, (int)a.z, (int)a.w, (int)c.x, (int)c.y, (int)c.z, (int)c.w};
#pragma unroll
    for (int k = 0; k < 8; k++) o.w[k] = ((w[k] >> sh) & 0x0F0F0F0F) | (((hh[k] >> ib) & 0x01010101) << 4);
    o.s_lo = o.s_hi = h2f(b) * sc;
    o.b_lo = o.b_hi = -h2f(b + 2) * mn;
}
template <>
__device__ __forceinline__ void decode_q8<14>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 210;
    const int ib = sub % 8, n = ib / 4, part = ib % 4;
    const uint8_t* ql = b + 64 * n + (part & 1) * 32;
    const uint8_t* qh = b + 128 + 32 * n;
    const int nsh = part >= 2 ? 4 : 0, hsh = 2 * part;
#pragma unroll
    for (int k = 0; k < 8; k++)
        o.w[k] = ((ld4_a2(ql + 4 * k) >> nsh) & 0x0F0F0F0F) | (((ld4_a2(qh + 4 * k) >> hsh) & 0x03030303) << 4);
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
    const float d = h2f(b + 208);
    o.s_lo = d * sc[2 * part], o.s_hi = d * sc[2 * part + 1];
    o.b_lo = -32 * o.s_lo, o.b_hi = -32 * o.s_hi;
}
template <>
__device__ __forceinline__ void decode_q8<20>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + sub * 18;
    const int* t = reinterpret_cast<const int*>(d_kvalues);
    const int t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int q = ld4_a2(b + 2 + 4 * k);
        o.w[k] = lut16(q & 0x0F0F0F0F, t0, t1, t2, t3);
        o.w[k + 4] = lut16((q >> 4) & 0x0F0F0F0F, t0, t1, t2, t3);
    }
    o.s_lo = o.s_hi = h2f(b);
}
// IQ4_NL: 2 bytes of scale and 16 of values per sub-block
template <>
struct Raw<20> {
    static constexpr bool kOn = true;
    int q[4];
    uint32_t d;
};
__device__ __forceinline__ void load_raw(const uint8_t* row, int sub, Raw<20>& r) {
    const uint8_t* b = row + sub * 18;
    r.d = u16(b);
#pragma unroll
    for (int k = 0; k < 4; k++) r.q[k] = ld4_a2(b + 2 + 4 * k);
}
__device__ __forceinline__ void decode_raw(const Raw<20>& r, const Tabs&, Dec& o) {
    const int* t = reinterpret_cast<const int*>(d_kvalues);
    const int t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        o.w[k] = lut16(r.q[k] & 0x0F0F0F0F, t0, t1, t2, t3);
        o.w[k + 4] = lut16((r.q[k] >> 4) & 0x0F0F0F0F, t0, t1, t2, t3);
    }
    o.s_lo = o.s_hi = h2f_bits(r.d);
}
template <>
__device__ __forceinline__ void decode_q8<23>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 136;
    const int ib = sub % 8;
    const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((u16(b + 2) >> 2 * ib) & 3) << 4);
    const int* t = reinterpret_cast<const int*>(d_kvalues);
    const int t0 = t[0], t1 = t[1], t2 = t[2], t3 = t[3];
    const uint8_t* qs = b + 8 + 16 * ib;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int q = ld4(qs + 4 * k);
        o.w[k] = lut16(q & 0x0F0F0F0F, t0, t1, t2, t3);
        o.w[k + 4] = lut16((q >> 4) & 0x0F0F0F0F, t0, t1, t2, t3);
    }
    o.s_lo = o.s_hi = h2f(b) * (float)(ls - 32);
}

}  // namespace
}  // namespace tona::gpu
