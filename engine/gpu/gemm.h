// Building blocks of the prefill: unpack weights from the GGUF format to f16, and the f16 GEMM with WMMA (gfx12).
// Y[T x N] = X[T x K] * W[N x K]^T, W row by row as in the GGUF (one output per row), X one row per token.
// Kernel measurement: research/measurements/2026-10-23-gemm-wmma.md (36-52 TFLOPS at T = 4096).
#pragma once

#include <hip/hip_runtime.h>

#include <climits>
#include <cstdint>

namespace tona::gpu {

// Partial GEMM of one group: own pointers and shapes (e.g. one expert with its tokens). K a multiple of 32, x and w
// 16-byte aligned; T and N arbitrary (edges are masked). ldx, ldw, ldy: row stride in elements.
struct GemmSeg {
    const _Float16* x;
    const _Float16* w;
    float* y;
    int32_t t, n, k;
    int32_t ldx, ldw, ldy;
    // Columns from n2 on go to y2 (stride ldy2): two weights with the same input in one GEMM
    float* y2 = nullptr;
    int32_t n2 = INT32_MAX, ldy2 = 0;
};
constexpr int GEMM_MAX_SEGS = 32;

// All parts in one launch; every part has its own tiles
void gemm_f16(const GemmSeg* segs, int nseg, hipStream_t stream = nullptr);
// Hyper-connections: up GEMM (s.n = hc * d outputs, weight row g * d + k) with the mix in the epilogue instead of
// writing gate: mixed[t][k] = sum_g R[t][g][k] * sc[t][g] * w[g][k] * sigmoid(gate[t][g][k]) / hc, also as f16. s.y
// stays unused. Computes like gemm_f16 followed by k_mix (bit-identical). Only hc = 4, d a multiple of 64.
struct GemmMix {
    const float *R = nullptr, *sc = nullptr, *w = nullptr;
    float* mixed = nullptr;
    _Float16* mixed16 = nullptr;
    int32_t d = 0;
};
void gemm_f16_mix(const GemmSeg& s, const GemmMix& m, hipStream_t stream = nullptr);

// Shape of a dense GEMM with the default strides (ldx = ldw = k, ldy = n)
struct GemmDims {
    int t, n, k;
};
inline void gemm_f16(const _Float16* x, const _Float16* w, float* y, const GemmDims& d, hipStream_t stream = nullptr) {
    const GemmSeg s{x, w, y, d.t, d.n, d.k, d.k, d.k, d.n};
    gemm_f16(&s, 1, stream);
}

// GEMM with the weights in GGUF format (all parts in the same format): per K step the block unpacks the 128 rows of
// its output tile into the LDS, without an f16 copy in memory. Tile 64 tokens x 128 outputs, for experts with few
// tokens per chunk. w: device pointer (mapped host memory too), row_bytes per row.
struct GemmQSeg {
    const _Float16* x;
    const uint8_t* w;
    float* y;
    int32_t t, n, k;
    int32_t ldx, ldy;
    int64_t row_bytes;
};
void gemm_q(uint32_t type, const GemmQSeg* segs, int nseg, hipStream_t stream = nullptr);

// FP8 GEMM (E4M3, WMMA fp8, gfx12): Y[t][n] = sx[t] * sw[n] * sum_k X8[t][k] * W8[n][k]. One factor per row, applied
// at the end. ldx, ldw in bytes, ldy in floats; K a multiple of 64, x and w 16-byte aligned.
// Factors per block of 128 in K (as in DeepSeek-V3) were slower (30 instead of 46 TFLOPS) and not more accurate
// (perplexity 2.520 against 2.510; measurements/2026-10-24-prefill-fp8).
struct Gemm8Seg {
    const uint8_t* x;
    const float* sx;
    const uint8_t* w;
    const float* sw;
    float* y;
    int32_t t, n, k;
    int32_t ldx, ldw, ldy;
};
void gemm_fp8(const Gemm8Seg* segs, int nseg, hipStream_t stream = nullptr);
// Rows f16 -> E4M3 with a factor per row (absolute maximum / 448); scale[r] = factor
struct Fp8Rows {
    const _Float16* in;
    int64_t ld_in, rows, k;
    uint8_t* out;
    int64_t ld_out;
    float* scale;
};
void to_fp8_rows(const Fp8Rows& r, hipStream_t stream = nullptr);

bool dequant_f16_supported(uint32_t type);
// rows rows of `in` elements in ggml format `type` (device pointer, mapped host memory too) -> f16 with row stride
// ld_out. in a multiple of 32.
struct DequantShape {
    uint32_t type;
    int64_t rows, in, ld_out;
};
void dequant_f16(const DequantShape& d, const void* w, _Float16* out, hipStream_t stream = nullptr);
// rows rows of `in` elements in ggml format `type` -> E4M3 with a factor per row (as dequant_f16 followed by
// to_fp8_rows, in one pass). out: rows x in bytes, scale[r] = factor. in a multiple of 64, at most 8192; ld_out must be
// in.
void dequant_fp8(const DequantShape& d, const void* w, uint8_t* out, float* scale, hipStream_t stream = nullptr);
// Several matrices of the same format and shape in one launch (e.g. the same matrix of several experts)
void dequant_f16_multi(const DequantShape& d, const void* const* w, _Float16* const* out, int n,
                       hipStream_t stream = nullptr);

}  // namespace tona::gpu
