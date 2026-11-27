// GEMM with NVFP4 and IQ2_XXS weights, the part of gemm_q (gpu/gemm.h) that lives in its own unit (gemm_q_bolt.hip).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "gpu/gemm.h"

namespace tona::gpu {

bool gemm_q_bolt_type(uint32_t type);
void gemm_q_bolt(uint32_t type, const GemmQSeg* segs, int nseg, hipStream_t stream);
// dequant_f16_multi and dequant_fp8 (gpu/gemm.h) for these types (gemm_dequant_bolt.hip)
void dequant_bolt_multi(const DequantShape& d, const void* const* w, _Float16* const* out, int n, hipStream_t stream);
void dequant_bolt_fp8(const DequantShape& d, const void* w, uint8_t* out, float* scale, hipStream_t stream);

}  // namespace tona::gpu
