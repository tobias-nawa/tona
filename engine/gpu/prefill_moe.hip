// Prefill kernels of the MoE layers: routing, grouping of the tokens by expert, SwiGLU and the weighted sum.
#include "gpu/mega_mailbox_fwd.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"
#include "gpu/prefill_split.h"

namespace tona::gpu {

namespace {

// Best candidate of the lane, then over the wave (larger value, on a tie the smaller index)
template <int PER>
__device__ __forceinline__ void route_best(const float (&l)[PER], int lane, float& bv, int& be) {
    bv = -INFINITY;
    be = 1 << 30;
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int e = lane + WAVE * k;
        if (l[k] > bv || (l[k] == bv && e < be)) bv = l[k], be = e;
    }
    for (int off = WAVE / 2; off > 0; off /= 2) {
        const float ov = __shfl_xor(bv, off, WAVE);
        const int oe = __shfl_xor(be, off, WAVE);
        if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
    }
}

// Counts the row if a logit is NaN (cnt[route_nan_word]), then takes NaN as -inf
template <int PER>
__device__ __forceinline__ void route_clean(float (&l)[PER], int lane, int32_t* cnt) {
    float nan_lanes = 0;
#pragma unroll
    for (int k = 0; k < PER; k++) nan_lanes += l[k] != l[k] ? 1.0f : 0.0f;
    nan_lanes = wave_sum(nan_lanes);
    if (lane == 0 && nan_lanes > 0) atomicAdd(&cnt[route_nan_word(PER * WAVE)], 1);
#pragma unroll
    for (int k = 0; k < PER; k++) l[k] = pf_clean_logit(l[k]);
}

// MaxE: size class of the experts per layer; the Coder's instantiation is the kernel it always had
template <int MaxE>
__global__ void k_route(RouteArgs a) {
    const auto& [logits, sel, sw, cnt, n, E, ne, max_e] = a;
    (void)max_e;  // the size class is the template parameter
    const int64_t t = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (t >= n) return;
    constexpr int PER = MaxE / WAVE;  // logits per lane
    float l[PER];
    float mx = -INFINITY;
#pragma unroll
    for (int k = 0; k < PER; k++) {
        const int e = lane + WAVE * k;
        l[k] = e < E ? logits[t * E + e] : -INFINITY;
        mx = fmaxf(mx, l[k]);
    }
    route_clean<PER>(l, lane, cnt);  // fmaxf skips NaN, so mx is the same before and after
    mx = wave_max(mx);
    float sum = 0;
#pragma unroll
    for (int k = 0; k < PER; k++) sum += l[k] > -INFINITY ? __expf(l[k] - mx) : 0.0f;
    sum = wave_sum(sum);
    float wsum = 0, pch[16];
    int ech[16];
    for (int r = 0; r < ne; r++) {
        float bv;
        int be;
        route_best<PER>(l, lane, bv, be);
#pragma unroll
        for (int k = 0; k < PER; k++)
            if (lane + WAVE * k == be) l[k] = -INFINITY;
        const float p = __expf(bv - mx) / sum;
        pch[r] = p, ech[r] = be;
        wsum += p;
    }
    if (lane == 0)
        for (int r = 0; r < ne; r++) {
            sel[t * ne + r] = ech[r];
            sw[t * ne + r] = pch[r] / wsum;
            atomicAdd(&cnt[ech[r]], 1);
        }
}

__global__ void k_scatter(ScatterArgs args) {
    const auto& [sel, off, fill, rowmap, x, xg, n, ne, d] = args;
    const int64_t a = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (a >= (int64_t)n * ne) return;
    const int e = sel[a];
    int r = 0;
    if (lane == 0) r = off[e] + atomicAdd(&fill[e], 1);
    r = __shfl(r, 0, WAVE);
    if (lane == 0) rowmap[a] = r;
    const uint4* src = reinterpret_cast<const uint4*>(x + a / ne * d);
    uint4* dst = reinterpret_cast<uint4*>(xg + (int64_t)r * d);
    for (int k = lane; k < d / 8; k += WAVE) dst[k] = src[k];
}

__global__ void k_swiglu16(Swiglu16Args a) {
    const auto& [g, u, act, row0, rows, ff] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= rows * ff) return;
    const int64_t j = row0 * ff + i;
    act[j] = (_Float16)(silu_f(g[j]) * u[j]);
}

__global__ void k_moe_sum(MoeSumArgs a) {
    const auto& [out, rowmap, sw, sho, shi, y, n, d, ne] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * d) return;
    const int64_t t = i / d, k = i % d;
    float s = 0;
    for (int r = 0; r < ne; r++) s += sw[t * ne + r] * out[(int64_t)rowmap[t * ne + r] * d + k];
    y[i] = s + sigm(shi[t]) * sho[i];
}

__global__ void k_moe_check(MoeCheckArgs a) {
    const auto& [sel, rowmap, cnt, off, flag, n, ne, E, stage] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * ne) return;
    const int e = sel[i];
    const int r = stage == 0 ? 0 : rowmap[i];
    const bool ok = stage == 0 ? e >= 0 && e < E : pf_row_ok(r, e, E, (int64_t)n * ne, off, cnt);
    if (ok) return;
    if (atomicAdd(&flag[0], 1) == 0) flag[1] = (int)i, flag[2] = e, flag[3] = r;
}

}  // namespace

void launch_moe_check(const MoeCheckArgs& a, hipStream_t st) {
    k_moe_check<<<blocks_for((int64_t)a.n * a.ne), TPB, 0, st>>>(a);
}
void launch_route(const RouteArgs& a, hipStream_t st) {
    if (a.max_e > MEGA_MAX_EXPERTS)
        k_route<MEGA_MAX_EXPERTS_WIDE><<<blocks_for((int64_t)a.n * WAVE), TPB, 0, st>>>(a);
    else
        k_route<MEGA_MAX_EXPERTS><<<blocks_for((int64_t)a.n * WAVE), TPB, 0, st>>>(a);
}
void launch_scatter(const ScatterArgs& a, hipStream_t st) {
    k_scatter<<<blocks_for((int64_t)a.n * a.ne * WAVE), TPB, 0, st>>>(a);
}
void launch_swiglu16(const Swiglu16Args& a, hipStream_t st) {
    k_swiglu16<<<blocks_for(a.rows * a.ff), TPB, 0, st>>>(a);
}
void launch_moe_sum(const MoeSumArgs& a, hipStream_t st) {
    for_row_pieces(a.n, [&](int t0, int m) {
        MoeSumArgs b = a;  // out stays: rowmap holds absolute rows of it
        b.rowmap += (int64_t)t0 * a.ne, b.sw += (int64_t)t0 * a.ne, b.shi += t0, b.sho += (int64_t)t0 * a.d;
        b.y += (int64_t)t0 * a.d, b.n = m;
        k_moe_sum<<<blocks_for((int64_t)m * a.d), TPB, 0, st>>>(b);
    });
}

}  // namespace tona::gpu
