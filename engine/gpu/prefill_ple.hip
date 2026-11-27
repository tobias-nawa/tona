// Prefill kernels of the per-layer embedding (PLE): gating of the streams and the dilated causal conv.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

namespace {

__global__ void __launch_bounds__(TPB) k_ple_gate(PleGateArgs a) {
    const auto& [R, key, val, w_key, w_query, w_conv, gated, normed, hc, d, eps] = a;
    __shared__ float red[TPB / WAVE];
    const int b = blockIdx.x / hc, g = blockIdx.x % hc;
    const int64_t n = (int64_t)hc * d;
    const float* kc = key + b * n + (int64_t)g * d;
    const float* rc = R + b * n + (int64_t)g * d;
    const float* vb = val + (int64_t)b * d;
    float* gt_ = gated + b * n + (int64_t)g * d;
    float* nm = normed + b * n + (int64_t)g * d;
    float sk = 0, sq = 0;
    for (int i = threadIdx.x; i < d; i += TPB) sk += kc[i] * kc[i], sq += rc[i] * rc[i];
    const float fk = rsqrtf(block_sum256(sk, red) / d + eps), fq = rsqrtf(block_sum256(sq, red) / d + eps);
    float dot = 0;
    for (int i = threadIdx.x; i < d; i += TPB)
        dot += (kc[i] * fk * w_key[g * d + i]) * (rc[i] * fq * w_query[g * d + i]);
    const float s = block_sum256(dot, red) * rsqrtf((float)d);
    const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
    const float sgn = (float)((s > 0) - (s < 0));
    const float gt = sigm(sgn * mag);
    float sg = 0;
    for (int i = threadIdx.x; i < d; i += TPB) {
        const float v = vb[i] * gt;
        gt_[i] = v;
        sg += v * v;
    }
    const float fg = rsqrtf(block_sum256(sg, red) / d + eps);
    for (int i = threadIdx.x; i < d; i += TPB) nm[i] = gt_[i] * fg * w_conv[g * d + i];
}

__global__ void k_ple_conv(PleConvArgs a) {
    const auto& [R, gated, normed, w, hist, n, pos0, cols, K, dil] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * cols) return;
    const int64_t t = i / cols, ch = i % cols, pos = pos0 + t, RG = (int64_t)(K - 1) * dil + BMAX;
    float acc = 0;
    for (int k = 0; k < K; k++) {
        const int64_t p = pos - (int64_t)(K - 1 - k) * dil;
        const float x = p >= pos0 ? normed[(p - pos0) * cols + ch] : p >= 0 ? hist[(p % RG) * cols + ch] : 0.0f;
        acc += w[ch * K + k] * x;
    }
    R[i] += gated[i] + silu_f(acc);
}

}  // namespace

void launch_ple_gate(const PleGateArgs& a, int n_tokens, hipStream_t st) {
    k_ple_gate<<<n_tokens * a.hc, TPB, 0, st>>>(a);
}
void launch_ple_conv(const PleConvArgs& a, hipStream_t st) {
    k_ple_conv<<<blocks_for((int64_t)a.n * a.cols), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
