// Device-side pieces shared by the GEMM kernels (gemm_f16.hip, gemm_q.hip, gemm_fp8.hip): vector types, tile
// geometry, mapping of a block to its tile, the WMMA step and the epilogue that writes an accumulator tile.
// Everything is inlined into the kernels; there is no -fgpu-rdc, so a header is the only way to share it.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <format>
#include <stdexcept>

#include "gpu/gemm.h"
#include "gpu/lds_sync.h"

#define HIPCHECK(x)                                                                                           \
    do {                                                                                                      \
        hipError_t err_ = (x);                                                                                \
        if (err_ != hipSuccess) throw std::runtime_error(std::format("{}: {}", #x, hipGetErrorString(err_))); \
    } while (0)

namespace tona::gpu {
namespace {

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));
using int4v = int __attribute__((ext_vector_type(4)));

// Tile per block: BT tokens x BN outputs, K in steps of BK. 8 waves as 2 (T) x 4 (N), 64 x 32 per wave.
constexpr int BT = 128, BN = 128, BK = 32, WT = 64, WN = 32;
constexpr int LDK = BK + 8;  // LDS row stride in halves (80 bytes), against bank conflicts
constexpr int GT = 8;        // token tiles per group (blocks running together share X and W in the cache)
constexpr int BK2 = 64, LDK2 = BK2 + 8;

struct TileOrigin {
    int t0, n0;
};
// Tile order: per group of GT token tiles all output tiles, token tile fastest. nt, nn: tile counts of the segment.
__device__ __forceinline__ TileOrigin tile_origin(int loc, int nt, int nn, int tile_t, int tile_n) {
    const int g = loc / (GT * nn), gt = min(GT, nt - g * GT), in = loc % (GT * nn);
    return {(g * GT + in % gt) * tile_t, (in / gt) * tile_n};
}

// Position of a lane in its wave's WMMA fragments: wave origin in the tile (wt, wn), row r, K offset kh
struct WaveFrag {
    int wt, wn, r, kh;
};

// One K step of 16: every (i, j) fragment pair of the wave. gfx12 layout: lane l holds row (A) or column (B) l % 16
// and the K values (l / 16) * 8 ... + 7. Result: lane l holds column l % 16, rows (l / 16) * 8 ... + 7.
template <int NI, int NJ, int LD>
__device__ __forceinline__ void mma_kstep(const _Float16* sa, const _Float16* sb, const WaveFrag f, int kk,
                                          float8 (&acc)[NI][NJ]) {
    half8 av[NI], bv[NJ];
#pragma unroll
    for (int i = 0; i < NI; i++) av[i] = *reinterpret_cast<const half8*>(&sa[(f.wt + i * 16 + f.r) * LD + kk + f.kh]);
#pragma unroll
    for (int j = 0; j < NJ; j++) bv[j] = *reinterpret_cast<const half8*>(&sb[(f.wn + j * 16 + f.r) * LD + kk + f.kh]);
#pragma unroll
    for (int i = 0; i < NI; i++)
#pragma unroll
        for (int j = 0; j < NJ; j++)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av[i], bv[j], acc[i][j]);
}

// acc[i][j]: token rows wt + i*16 + kh + e, output column wn + j*16 + r. Columns from s.n2 on go to y2.
template <int NI, int NJ>
__device__ __forceinline__ void store_tile(const GemmSeg* s, const TileOrigin o, const WaveFrag f,
                                           const float8 (&acc)[NI][NJ]) {
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        const int n = o.n0 + f.wn + j * 16 + f.r;
        if (n >= s->n) continue;
        float* const y = n < s->n2 ? s->y + n : s->y2 + (n - s->n2);
        const int64_t ldy = n < s->n2 ? s->ldy : s->ldy2;
#pragma unroll
        for (int i = 0; i < NI; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = o.t0 + f.wt + i * 16 + f.kh + e;
                if (t < s->t) y[t * ldy] = acc[i][j][e];
            }
    }
}

// The same for a quantized-weight segment: no second output, y is dense
template <int NI, int NJ>
__device__ __forceinline__ void store_tile_q(const GemmQSeg* s, const TileOrigin o, const WaveFrag f,
                                             const float8 (&acc)[NI][NJ]) {
#pragma unroll
    for (int j = 0; j < NJ; j++) {
        const int n = o.n0 + f.wn + j * 16 + f.r;
        if (n >= s->n) continue;
#pragma unroll
        for (int i = 0; i < NI; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = o.t0 + f.wt + i * 16 + f.kh + e;
                if (t < s->t) s->y[(int64_t)t * s->ldy + n] = acc[i][j][e];
            }
    }
}

}  // namespace
}  // namespace tona::gpu
