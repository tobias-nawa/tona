// Unpacking of weights from the GGUF formats to f16 (one thread per matrix row and 32-value sub-block): the kernels,
// launchers and the launch loops. gemm_dequant.hip and gemm_dequant_bolt.hip name the formats they instantiate.
#pragma once

#include <cstdint>
#include <span>

#include "gpu/gemm_deq_dev.h"
#include "gpu/gemm_dev.h"

namespace tona::gpu {

namespace {

// Quantized formats; F32, F16 and BF16 are in gemm_deq_dev.h (gemm_q.hip unpacks the others itself)
template <>
__device__ void deq_sub<8>(const uint8_t* row, int sub, float* o) {  // Q8_0
    const uint8_t* b = row + sub * 34;
    const int8_t* q = reinterpret_cast<const int8_t*>(b + 2);
    const float d = h2f(b);
    for (int j = 0; j < 32; j++) o[j] = q[j] * d;
}
template <>
__device__ void deq_sub<20>(const uint8_t* row, int sub, float* o) {  // IQ4_NL
    const uint8_t* b = row + sub * 18;
    const uint8_t* q = b + 2;
    const float d = h2f(b);
    for (int j = 0; j < 16; j++) o[j] = d_kvalues[q[j] & 0xf] * d, o[j + 16] = d_kvalues[q[j] >> 4] * d;
}
template <>
__device__ void deq_sub<42>(const uint8_t* row, int sub, float* o) {  // Q2_0, 64-value blocks
    const uint8_t* b = row + (sub / 2) * 18;
    const uint8_t* q = b + 2 + (sub & 1) * 8;
    const float d = h2f(b);
    for (int j = 0; j < 32; j++) o[j] = (float)(((q[j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
}
template <>
__device__ void deq_sub<12>(const uint8_t* row, int sub, float* o) {  // Q4_K
    const uint8_t* b = row + (sub / 8) * 144;
    const int ib = sub % 8, jj = ib / 2;
    int sc, m;
    scale_min_k4(ib, b + 4, sc, m);
    const uint8_t* q = b + 16 + 32 * jj;
    const float d = h2f(b) * sc, mn = h2f(b + 2) * m;
    for (int l = 0; l < 32; l++) o[l] = d * ((ib & 1) ? (q[l] >> 4) : (q[l] & 0xF)) - mn;
}
template <>
__device__ void deq_sub<13>(const uint8_t* row, int sub, float* o) {  // Q5_K
    const uint8_t* b = row + (sub / 8) * 176;
    const int ib = sub % 8, jj = ib / 2;
    int sc, m;
    scale_min_k4(ib, b + 4, sc, m);
    const uint8_t* qh = b + 16;
    const uint8_t* ql = b + 48 + 32 * jj;
    const uint8_t u = (uint8_t)((ib & 1 ? 2 : 1) << (2 * jj));
    const float d = h2f(b) * sc, mn = h2f(b + 2) * m;
    for (int l = 0; l < 32; l++) o[l] = d * (((ib & 1) ? (ql[l] >> 4) : (ql[l] & 0xF)) + ((qh[l] & u) ? 16 : 0)) - mn;
}
template <>
__device__ void deq_sub<14>(const uint8_t* row, int sub, float* o) {  // Q6_K
    const uint8_t* b = row + (sub / 8) * 210;
    const int ib = sub % 8, n = ib / 4, part = ib % 4;
    const uint8_t* ql = b + 64 * n;
    const uint8_t* qh = b + 128 + 32 * n;
    const int8_t* sc = reinterpret_cast<const int8_t*>(b + 192) + 8 * n;
    const float d = h2f(b + 208);
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
        o[l] = d * sc[2 * part + (l >= 16)] * (q - 32);
    }
}
template <>
__device__ void deq_sub<22>(const uint8_t* row, int sub, float* o) {  // IQ2_S
    const uint8_t* b = row + (sub / 8) * 82;
    const int ib = sub % 8;
    const float d = h2f(b) * 0.25f;
    const uint8_t* qs = b + 2 + 4 * ib;
    const uint8_t* signs = b + 2 + 32 + 4 * ib;
    const uint8_t qh = b[66 + ib], sc = b[74 + ib];
    for (int l = 0; l < 4; l++) {
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(d_iq2s_grid + (qs[l] | ((qh << (8 - 2 * l)) & 0x300)));
        const float s = d * (0.5f + ((l < 2 ? sc : sc >> 4) & 0xf));
        for (int j = 0; j < 8; j++) o[8 * l + j] = s * ((signs[l] >> j) & 1 ? -(float)grid[j] : (float)grid[j]);
    }
}
template <>
__device__ void deq_sub<18>(const uint8_t* row, int sub, float* o) {  // IQ3_XXS
    const uint8_t* b = row + (sub / 8) * 98;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t* ss = b + 2 + 64 + 4 * ib;
    const uint32_t aux = (uint32_t)u16(ss) | ((uint32_t)u16(ss + 2) << 16);
    const float d = h2f(b) * (0.5f + (aux >> 28)) * 0.5f;
    for (int l = 0; l < 4; l++) {
        const uint8_t sg = d_ksigns[(aux >> 7 * l) & 127];
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(d_iq3xxs_grid + qs[2 * l]);
        const uint8_t* g2 = reinterpret_cast<const uint8_t*>(d_iq3xxs_grid + qs[2 * l + 1]);
        for (int j = 0; j < 4; j++) {
            o[8 * l + j] = d * ((sg >> j) & 1 ? -(float)g1[j] : (float)g1[j]);
            o[8 * l + 4 + j] = d * ((sg >> (j + 4)) & 1 ? -(float)g2[j] : (float)g2[j]);
        }
    }
}
template <>
__device__ void deq_sub<21>(const uint8_t* row, int sub, float* o) {  // IQ3_S
    const uint8_t* b = row + (sub / 8) * 110;
    const int ib = sub % 8;
    const uint8_t* qs = b + 2 + 8 * ib;
    const uint8_t qh = b[66 + ib];
    const uint8_t* signs = b + 74 + 4 * ib;
    const int sc = (b[106 + ib / 2] >> (4 * (ib & 1))) & 0xf;
    const float d = h2f(b) * (1 + 2 * sc);
    for (int l = 0; l < 4; l++) {
        const uint8_t* g1 = reinterpret_cast<const uint8_t*>(d_iq3s_grid + (qs[2 * l] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* g2 =
            reinterpret_cast<const uint8_t*>(d_iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; j++) {
            o[8 * l + j] = d * ((signs[l] >> j) & 1 ? -(float)g1[j] : (float)g1[j]);
            o[8 * l + 4 + j] = d * ((signs[l] >> (j + 4)) & 1 ? -(float)g2[j] : (float)g2[j]);
        }
    }
}
template <>
__device__ void deq_sub<23>(const uint8_t* row, int sub, float* o) {  // IQ4_XS
    const uint8_t* b = row + (sub / 8) * 136;
    const int ib = sub % 8;
    const int ls = ((b[4 + ib / 2] >> 4 * (ib % 2)) & 0xf) | (((u16(b + 2) >> 2 * ib) & 3) << 4);
    const uint8_t* q = b + 8 + 16 * ib;
    const float d = h2f(b) * (ls - 32);
    for (int j = 0; j < 16; j++) o[j] = d_kvalues[q[j] & 0xf] * d, o[j + 16] = d_kvalues[q[j] >> 4] * d;
}

// Sub-block of a row of row length in; the tile type needs the length (matvec_q8_tile.h). Same arithmetic as
// deq_sub<14>: (d * scale) * (q - 32), so the same bits.
template <uint32_t T>
__device__ void deq_sub_in(const uint8_t* row, int64_t in, int sub, float* o) {
    if constexpr (T == kQ6KTile) {
        Dec dec;
        decode_tile(row, (int)in, sub, dec);
        for (int j = 0; j < 32; j++) {
            const int by = (int)((uint32_t)dec.w[j / 4] >> (8 * (j % 4)) & 0xff);
            o[j] = (j < 16 ? dec.s_lo : dec.s_hi) * (float)(by - 32);
        }
    } else {
        deq_sub<T>(row, sub, o);
    }
}

constexpr int DEQ_MAX = 32;
struct DeqBatch {
    const uint8_t* w[DEQ_MAX];
    _Float16* out[DEQ_MAX];
};

// Thread per (matrix, row, sub-block); blockIdx.y = matrix
template <uint32_t T>
__global__ void __launch_bounds__(256) k_dequant(DeqBatch b, int64_t rows, int64_t in, int64_t row_bytes, int64_t ld) {
    const int64_t subs = in / 32, i = (int64_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= rows * subs) return;
    const int64_t r = i / subs;
    const int s = (int)(i % subs);
    float o[32];
    deq_sub_in<T>(row_at<T>(b.w[blockIdx.y], r, row_bytes), in, s, o);
    _Float16* dst = b.out[blockIdx.y] + r * ld + s * 32;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        half8 h;
#pragma unroll
        for (int j = 0; j < 8; j++) h[j] = (_Float16)o[8 * k + j];
        reinterpret_cast<half8*>(dst)[k] = h;
    }
}

// The rows go to E4M3 with a factor per row (absolute maximum / 448) in one pass: a block per row keeps the row as f16
// in the LDS (the same rounding as dequant_f16 followed by to_fp8_rows, so the same bytes) instead of writing and
// reading it twice through memory. Rows up to 8192 values.
constexpr int FP8_MAX_IN = 8192;

template <uint32_t T>
__global__ void __launch_bounds__(256)
    k_dequant_fp8(const uint8_t* w, int64_t in, int64_t row_bytes, uint8_t* out, float* scale) {
    __shared__ __attribute__((aligned(16))) _Float16 row[FP8_MAX_IN];
    __shared__ float red[8];
    const int64_t r = blockIdx.x;
    float mx = 0;
    for (int s = threadIdx.x; s < in / 32; s += 256) {
        float o[32];
        deq_sub_in<T>(row_at<T>(w, r, row_bytes), in, s, o);
#pragma unroll
        for (int j = 0; j < 32; j++) {
            row[s * 32 + j] = (_Float16)o[j];
            mx = fmaxf(mx, fabsf((float)row[s * 32 + j]));
        }
    }
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    lds_sync();
    mx = 0;
    for (int i = 0; i < 8; i++) mx = fmaxf(mx, red[i]);
    const float sc = mx > 0 ? mx / 448.0f : 1.0f, inv = 1.0f / sc;
    if (threadIdx.x == 0) scale[r] = sc;
    uint32_t* dst = reinterpret_cast<uint32_t*>(out + r * in);
    for (int i = threadIdx.x; i < in / 4; i += 256) {
        int p = __builtin_amdgcn_cvt_pk_fp8_f32((float)row[4 * i] * inv, (float)row[4 * i + 1] * inv, 0, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32((float)row[4 * i + 2] * inv, (float)row[4 * i + 3] * inv, p, true);
        dst[i] = (uint32_t)p;
    }
}

using Fp8Launcher = void (*)(const uint8_t*, const DequantShape&, int64_t, uint8_t*, float*, hipStream_t);

template <uint32_t T>
void launch_dequant_fp8(const uint8_t* w, const DequantShape& d, int64_t row_bytes, uint8_t* out, float* scale,
                        hipStream_t stream) {
    k_dequant_fp8<T><<<(unsigned)d.rows, 256, 0, stream>>>(w, d.in, row_bytes, out, scale);
}

using DequantLauncher = void (*)(const DeqBatch&, dim3, const DequantShape&, int64_t, hipStream_t);

template <uint32_t T>
void launch_dequant(const DeqBatch& b, dim3 grid, const DequantShape& d, int64_t row_bytes, hipStream_t stream) {
    k_dequant<T><<<grid, 256, 0, stream>>>(b, d.rows, d.in, row_bytes, d.ld_out);
}

struct DequantFormat {
    uint32_t type;
    DequantLauncher launch;
    Fp8Launcher launch_fp8;
};

// The launch loops of dequant_f16_multi and dequant_fp8 with the table of one unit; false if it has no kernel for the
// type
inline bool dequant_multi_in(std::span<const DequantFormat> table, const DequantShape& d, const void* const* w,
                             _Float16* const* out, int count, hipStream_t stream) {
    const DequantFormat* fmt = nullptr;
    for (const DequantFormat& f : table)
        if (f.type == d.type) fmt = &f;
    if (!fmt) return false;
    const int64_t rb = row_bytes_of(d.type, d.in);
    for (int i0 = 0; i0 < count; i0 += DEQ_MAX) {
        DeqBatch b{};
        const int m = std::min(DEQ_MAX, count - i0);
        for (int i = 0; i < m; i++) b.w[i] = static_cast<const uint8_t*>(w[i0 + i]), b.out[i] = out[i0 + i];
        const dim3 grid((unsigned)((d.rows * (d.in / 32) + 255) / 256), (unsigned)m);
        fmt->launch(b, grid, d, rb, stream);
        HIPCHECK(hipGetLastError());
    }
    return true;
}

inline bool dequant_fp8_in(std::span<const DequantFormat> table, const DequantShape& d, const void* w, uint8_t* out,
                           float* scale, hipStream_t stream) {
    const DequantFormat* fmt = nullptr;
    for (const DequantFormat& f : table)
        if (f.type == d.type) fmt = &f;
    if (!fmt) return false;
    if (d.rows <= 0) return true;
    fmt->launch_fp8(static_cast<const uint8_t*>(w), d, row_bytes_of(d.type, d.in), out, scale, stream);
    HIPCHECK(hipGetLastError());
    return true;
}

}  // namespace
}  // namespace tona::gpu
