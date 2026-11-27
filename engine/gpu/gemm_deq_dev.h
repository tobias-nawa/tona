// Dequantization of one 32-value sub-block of a weight row to float, here for the unquantized formats (the rest is
// in gemm_dequant.hip). Same formulas as dot_sub in matvec_dev.h, without x. Shared by gemm_dequant.hip and gemm_q.hip.
#pragma once

#include "gpu/matvec_dev.h"

namespace tona::gpu {
namespace {

// Sub-block sub (32 values) of a row to o
template <uint32_t T>
__device__ void deq_sub(const uint8_t* row, int sub, float* o);

template <>
__device__ void deq_sub<0>(const uint8_t* row, int sub, float* o) {  // F32
    const float* w = reinterpret_cast<const float*>(row) + sub * 32;
    for (int j = 0; j < 32; j++) o[j] = w[j];
}
template <>
__device__ void deq_sub<1>(const uint8_t* row, int sub, float* o) {  // F16
    const _Float16* w = reinterpret_cast<const _Float16*>(row) + sub * 32;
    for (int j = 0; j < 32; j++) o[j] = (float)w[j];
}
template <>
__device__ void deq_sub<30>(const uint8_t* row, int sub, float* o) {  // BF16
    const uint16_t* w = reinterpret_cast<const uint16_t*>(row) + sub * 32;
    for (int j = 0; j < 32; j++) o[j] = __uint_as_float((uint32_t)w[j] << 16);
}

}  // namespace
}  // namespace tona::gpu
