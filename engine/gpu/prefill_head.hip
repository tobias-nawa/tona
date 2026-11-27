// Prefill kernels of the LM head check (negative log-likelihood of the next token) and a fill helper.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

namespace {

__global__ void __launch_bounds__(TPB) k_lse(LseArgs a) {
    const auto& [L, rs, r0, tgt, mrun, srun, tlog] = a;
    __shared__ float red[TPB / WAVE];
    const int t = blockIdx.x;
    const float* l = L + (int64_t)t * rs;
    float mx = -INFINITY;
    for (int i = threadIdx.x; i < rs; i += TPB) mx = fmaxf(mx, l[i]);
    mx = block_max256(mx, red);
    float s = 0;
    for (int i = threadIdx.x; i < rs; i += TPB) s += __expf(l[i] - mx);
    s = block_sum256(s, red);
    if (threadIdx.x == 0) {
        const float m0 = mrun[t], m1 = fmaxf(m0, mx);
        srun[t] = (m0 > -INFINITY ? srun[t] * __expf(m0 - m1) : 0.0f) + s * __expf(mx - m1);
        mrun[t] = m1;
        const int g = tgt[t];
        if (g >= r0 && g < r0 + rs) tlog[t] = l[g - r0];
    }
}

__global__ void k_fill(FillArgs a) {
    const auto& [p, n, v] = a;
    const int i = blockIdx.x * TPB + threadIdx.x;
    if (i < n) p[i] = v;
}

}  // namespace

void launch_lse(const LseArgs& a, int n_tokens, hipStream_t st) { k_lse<<<n_tokens, TPB, 0, st>>>(a); }
void launch_fill(const FillArgs& a, hipStream_t st) { k_fill<<<blocks_for(a.n), TPB, 0, st>>>(a); }

}  // namespace tona::gpu
