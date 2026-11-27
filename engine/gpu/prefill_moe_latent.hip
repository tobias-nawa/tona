// Prefill kernels of the latent MoE (prefill_moe_latent.h).
#include "gpu/prefill_dev.h"
#include "gpu/prefill_moe_latent.h"
#include "gpu/prefill_split.h"

namespace tona::gpu {

namespace {

__global__ void k_relu2_16(Relu2Args a) {
    const auto& [u, act, row0, rows, ff] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= rows * ff) return;
    const int64_t j = row0 * ff + i;
    const float v = u[j];
    act[j] = (_Float16)(v > 0.0f ? v * v : 0.0f);
}

__global__ void k_moe_sum_latent(LatentSumArgs a) {
    const auto& [out, rowmap, sw, y, n, width, ne] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * width) return;
    const int64_t t = i / width, k = i % width;
    float s = 0;
    for (int r = 0; r < ne; r++) s += sw[t * ne + r] * out[(int64_t)rowmap[t * ne + r] * width + k];
    y[i] = (_Float16)s;
}

__global__ void k_f32_to_f16(const float* src, _Float16* dst, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i < n) dst[i] = (_Float16)src[i];
}

__global__ void k_add_into(float* y, const float* a, int64_t n) {
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i < n) y[i] += a[i];
}

}  // namespace

void launch_relu2_16(const Relu2Args& a, hipStream_t st) { k_relu2_16<<<blocks_for(a.rows * a.ff), TPB, 0, st>>>(a); }

void launch_moe_sum_latent(const LatentSumArgs& a, hipStream_t st) {
    for_row_pieces(a.n, [&](int t0, int m) {
        LatentSumArgs b = a;  // out stays: rowmap holds absolute rows of it
        b.rowmap += (int64_t)t0 * a.ne, b.sw += (int64_t)t0 * a.ne, b.y += (int64_t)t0 * a.width, b.n = m;
        k_moe_sum_latent<<<blocks_for((int64_t)m * a.width), TPB, 0, st>>>(b);
    });
}

void launch_f32_to_f16(const float* src, _Float16* dst, int64_t n, hipStream_t st) {
    k_f32_to_f16<<<blocks_for(n), TPB, 0, st>>>(src, dst, n);
}

void launch_add_into(float* y, const float* a, int64_t n, hipStream_t st) {
    k_add_into<<<blocks_for(n), TPB, 0, st>>>(y, a, n);
}

}  // namespace tona::gpu
