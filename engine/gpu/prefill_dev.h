// Constants and block-level helpers shared by the prefill kernels (prefill_*.hip). Included by every translation unit
// that holds prefill kernels; there is no -fgpu-rdc, so this is the only way to share device code.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>

#include "gpu/lds_sync.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_types.h"

namespace tona::gpu {
namespace {

constexpr int DS = 128;  // DeltaNet head size
constexpr int BMAX = MEGA_BMAX, RAW_RING = MEGA_RAW_RING, MAX_CELLS = MEGA_MAX_CELLS;
constexpr int TPB = 256;  // threads per block of the simple kernels

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));

inline unsigned blocks_for(int64_t n) { return (unsigned)((n + TPB - 1) / TPB); }

// Does the token at pos select pool blocks (it sees more blocks than top_k holds)?
__device__ __forceinline__ bool qsa_sel(int64_t pos, int blk, int topk) {
    const int64_t n_pool = (pos + 1) / blk;
    return n_pool > std::min<int64_t>(n_pool, topk / blk);
}

__device__ __forceinline__ float block_sum256(float v, float* red) {
    v = wave_sum(v);
    __syncthreads();
    if (threadIdx.x % WAVE == 0) red[threadIdx.x / WAVE] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < TPB / WAVE; i++) t += red[i];
    return t;
}
__device__ __forceinline__ float block_max256(float v, float* red) {
    v = wave_max(v);
    __syncthreads();
    if (threadIdx.x % WAVE == 0) red[threadIdx.x / WAVE] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < TPB / WAVE; i++) t = fmaxf(t, red[i]);
    return t;
}

}  // namespace
}  // namespace tona::gpu
