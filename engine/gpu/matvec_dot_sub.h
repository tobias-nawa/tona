// Float path: dot product of one 32-value sub-block of a row with x, per format. The sub-blocks are unpacked the
// way model/quant.cpp does it (from ggml-quants.c), only per sub-block. Part of matvec_dev.h; do not include on its
// own.
#pragma once

#include "gpu/matvec_tables.h"

namespace tona::gpu {
namespace {

// Dot product of sub-block sub (32 values) of a row with x (already pointing at those 32 values)
template <uint32_t T>
__device__ float dot_sub(const uint8_t* row, int sub, const float* x);

template <>
__device__ float dot_sub<0>(const uint8_t* row, int sub, const float* x) {  // F32
    const float* w = reinterpret_cast<const float*>(row) + sub * 32;
    float s = 0;
    for (int j = 0; j < 32; j++) s += w[j] * x[j];
    return s;
}
template <>
__device__ float dot_sub<1>(const uint8_t* row, int sub, const float* x) {  // F16
    const _Float16* w = reinterpret_cast<const _Float16*>(row) + sub * 32;
    float s = 0;
    for (int j = 0; j < 32; j++) s += (float)w[j] * x[j];
    return s;
}
template <>
__device__ float dot_sub<30>(const uint8_t* row, int sub, const float* x) {  // BF16
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row) + sub * 32;
    float s = 0;
    for (int j = 0; j < 32; j++) s += __uint_as_float((uint32_t)w[j] << 16) * x[j];
    return s;
}
template <>
__device__ float dot_sub<8>(const uint8_t* row, int sub, const float* x) {  // Q8_0
    const uint8_t* b = row + sub * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
    float s = 0;
    for (int j = 0; j < 32; j++) s += q[j] * x[j];
    return s * h2f(b);
}
template <>
__device__ float dot_sub<20>(const uint8_t* row, int sub, const float* x) {  // IQ4_NL
    const uint8_t* b = row + sub * 18;
    const uint8_t* q = b + 2;
    float s = 0;
    for (int j = 0; j < 16; j++) s += d_kvalues[q[j] & 0xf] * x[j] + d_kvalues[q[j] >> 4] * x[j + 16];
    return s * h2f(b);
}
template <>
__device__ float dot_sub<42>(const uint8_t* row, int sub, const float* x) {  // Q2_0, Block 64
    const uint8_t* b = row + (sub / 2) * 18;
    const uint8_t* q = b + 2 + (sub & 1) * 8;
    float s = 0;
    for (int j = 0; j < 32; j++) s += (float)(((q[j / 4] >> ((j % 4) * 2)) & 3) - 1) * x[j];
    return s * h2f(b);
}
template <>
__device__ float dot_sub<12>(const uint8_t* row, int sub, const float* x) {  // Q4_K
    const uint8_t* b = row + (sub / 8) * 144;
    const int ib = sub % 8, jj = ib / 2;
    int sc, m;
    scale_min_k4(ib, b + 4, sc, m);
    const uint8_t* q = b + 16 + 32 * jj;
    float s = 0, sx = 0;
    for (int l = 0; l < 32; l++) {
        const int v = (ib & 1) ? (q[l] >> 4) : (q[l] & 0xF);
        s += v * x[l];
        sx += x[l];
    }
    return h2f(b) * sc * s - h2f(b + 2) * m * sx;
}
template <>
__device__ float dot_sub<13>(const uint8_t* row, int sub, const float* x) {  // Q5_K
    const uint8_t* b = row + (sub / 8) * 176;
    const int ib = sub % 8, jj = ib / 2;
    int sc, m;
    scale_min_k4(ib, b + 4, sc, m);
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48 + 32 * jj;
    const uint8_t u = (uint8_t)((ib & 1 ? 2 : 1) << (2 * jj));
    float s = 0, sx = 0;
    for (int l = 0; l < 32; l++) {
        const int v = ((ib & 1) ? (ql[l] >> 4) : (ql[l] & 0xF)) + ((qh[l] & u) ? 16 : 0);
        s += v * x[l];
        sx += x[l];
    }
    return h2f(b) * sc * s - h2f(b + 2) * m * sx;
}
template <>
__device__ float dot_sub<14>(const uint8_t* row, int sub, const float* x) {  // Q6_K
    const uint8_t* b = row + (sub / 8) * 210;
    const int ib = sub % 8, n = ib / 4, part = ib % 4;
    const uint8_t* ql = b + 64 * n;
    const uint8_t* qh = b + 128 + 32 * n;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
    float s0 = 0, s1 = 0;
    for (int l = 0; l < 32; l++) {
        int q;
        switch (part) {
            case 0:
                q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4);
                break;
            case 1:
                q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4);
                break;
            case 2:
                q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4);
                break;
            default:
                q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4);
                break;
        }
        (l < 16 ? s0 : s1) += (q - 32) * x[l];
    }
    return h2f(b + 208) * (sc[2 * part] * s0 + sc[2 * part + 1] * s1);
}
template <>
__device__ float dot_sub<22>(const uint8_t* row, int sub, const float* x) {  // IQ2_S
    const uint8_t* b = row + (sub / 8) * 82;
    const int ib = sub % 8;
    const float d = h2f(b);
    const uint8_t* qs = b + 2 + 4 * ib;
    const uint8_t* signs = b + 2 + 32 + 4 * ib;
    const uint8_t qh = b[66 + ib], sc = b[74 + ib];
    float r = 0;
    for (int l = 0; l < 4; l++) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(d_iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
        float s = 0;
        for (int j = 0; j < 8; j++) s += grid[j] * ((signs[l] >> j) & 1 ? -x[8 * l + j] : x[8 * l + j]);
        r += s * (0.5f + ((l < 2 ? sc : sc >> 4) & 0xf));
    }
    return r * d * 0.25f;
}
template <>
__device__ float dot_sub<18>(const uint8_t* row, int sub, const float* x) {  // IQ3_XXS
    const uint8_t* b = row + (sub / 8) * 98;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t* ss = b + 2 + 64 + 4 * ib;
    const uint32_t aux = (uint32_t)u16(ss) | ((uint32_t)u16(ss + 2) << 16);
    float r = 0;
    for (int l = 0; l < 4; l++) {
        const uint8_t sg = d_ksigns[(aux >> 7 * l) & 127];
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(d_iq3xxs_grid + qs[2 * l]);
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(d_iq3xxs_grid + qs[2 * l + 1]);
        for (int j = 0; j < 4; j++) {
            r += g1[j] * ((sg >> j) & 1 ? -x[8 * l + j] : x[8 * l + j]);
            r += g2[j] * ((sg >> (j + 4)) & 1 ? -x[8 * l + 4 + j] : x[8 * l + 4 + j]);
        }
    }
    return r * h2f(b) * (0.5f + (aux >> 28)) * 0.5f;
}
template <>
__device__ float dot_sub<21>(const uint8_t* row, int sub, const float* x) {  // IQ3_S
    const uint8_t* b = row + (sub / 8) * 110;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t qh = b[66 + ib];
    const uint8_t* signs = b + 74 + 4 * ib;
    const int sc = (b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf;
    float r = 0;
    for (int l = 0; l < 4; l++) {
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(d_iq3s_grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* g2 =
            reinterpret_cast<const uint8_t*>(d_iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; j++) {
            r += g1[j] * ((signs[l] >> j) & 1 ? -x[8 * l + j] : x[8 * l + j]);
            r += g2[j] * ((signs[l] >> (j + 4)) & 1 ? -x[8 * l + 4 + j] : x[8 * l + 4 + j]);
        }
    }
    return r * h2f(b) * (1 + 2 * sc);
}
template <>
__device__ float dot_sub<23>(const uint8_t* row, int sub, const float* x) {  // IQ4_XS
    const uint8_t* b = row + (sub / 8) * 136;
    const int ib = sub % 8;
    const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((u16(b + 2) >> 2 * ib) & 3) << 4);
    const uint8_t* q = b + 8 + 16 * ib;
    float s = 0;
    for (int j = 0; j < 16; j++) s += d_kvalues[q[j] & 0xf] * x[j] + d_kvalues[q[j] >> 4] * x[j + 16];
    return s * h2f(b) * (ls - 32);
}

}  // namespace
}  // namespace tona::gpu
