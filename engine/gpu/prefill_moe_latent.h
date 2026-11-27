// Prefill kernels of the latent MoE (Bolt): squared-ReLU activation of the ungated experts, the weighted sum of the
// experts' outputs in the latent space, and two small element-wise steps around the latent projections. The routing
// is launch_route_sig (prefill_route_sig.h), the scatter and the expert GEMMs are the shared MoE streaming's.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tona::gpu {

// act = f16(relu(u)^2), rows rows of ff from row0 (as the decode step OP_ACT_RELU2 computes it)
struct Relu2Args {
    const float* u;
    _Float16* act;
    int64_t row0, rows;
    int ff;
};
void launch_relu2_16(const Relu2Args& a, hipStream_t st);

// y[t][k] = f16(sum over the ne chosen experts of sw[t][r] * out[rowmap[t][r]][k]), k < width, in rank order
struct LatentSumArgs {
    const float* out;
    const int32_t* rowmap;
    const float* sw;
    _Float16* y;
    int n, width, ne;
};
void launch_moe_sum_latent(const LatentSumArgs& a, hipStream_t st);

// dst = f16(src), n values
void launch_f32_to_f16(const float* src, _Float16* dst, int64_t n, hipStream_t st);

// y += a, n values (the shared expert's output joins the projected routed experts)
void launch_add_into(float* y, const float* a, int64_t n, hipStream_t st);

}  // namespace tona::gpu
