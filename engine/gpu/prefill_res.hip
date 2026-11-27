// Prefill residual step of the plain residual stream (prefill_res.h).
#include "gpu/prefill_dev.h"
#include "gpu/prefill_res.h"

namespace tona::gpu {

namespace {

__global__ void __launch_bounds__(TPB) k_res_norm16(ResNorm16Args a) {
    __shared__ float red[TPB / WAVE];
    const int64_t t = blockIdx.x, base = t * a.d;
    float ss = 0;
    for (int i = threadIdx.x; i < a.d; i += TPB) {
        const float v = a.R[base + i] + (a.y ? a.y[base + i] : 0.0f);
        if (a.y) a.R[base + i] = v;
        ss += v * v;
    }
    if (!a.w) return;
    const float scale = rsqrtf(block_sum256(ss, red) / a.d + a.eps);
    // Written by this thread above (same index), so no barrier is needed; the value is still in the cache
    for (int i = threadIdx.x; i < a.d; i += TPB) {
        const float o = a.R[base + i] * scale * a.w[i];
        if (a.xn) a.xn[base + i] = (_Float16)o;
    }
}

}  // namespace

void launch_res_norm16(const ResNorm16Args& a, int n_tokens, hipStream_t st) {
    k_res_norm16<<<n_tokens, TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
