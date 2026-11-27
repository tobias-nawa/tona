// Prefill residual step of models with one plain residual stream (Ornith, Bolt): R += y, xn = RMSNorm(R) * w as f16 for
// the next GEMM. The counterpart of OP_RES_NORM of the megakernel (mega_op_resnorm.h); the Coder's k_combine_norm16 has
// four gated streams. Host-visible arguments and launcher; the kernel is in prefill_res.hip.
#pragma once

#include <hip/hip_runtime.h>

namespace tona::gpu {

// Block per token. R: [n][d], updated in place when y is given (y: [n][d], null: R stays); w: gain of the norm (null:
// no norm, only the sum); xn: [n][d] f16 (null: none). d a multiple of 4.
struct ResNorm16Args {
    float* R;
    const float* y;
    const float* w;
    _Float16* xn;
    int d;
    float eps;
};
void launch_res_norm16(const ResNorm16Args& a, int n_tokens, hipStream_t st);

}  // namespace tona::gpu
