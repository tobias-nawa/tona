// Step functions of the MoE layers: activation, expert sum, routing on the GPU.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// g, u, act, q: n = UMAX * ff + sff per token (i[0], i[1] = ff); only the n_u occupied slots and the shared expert
__device__ __noinline__ void op_act_quant(const MegaOp& op, const Ctx& c) {
    const float* g = (const float*)op.p[0];
    const float* u = (const float*)op.p[1];
    float* out = (float*)op.p[2];
    int8_t* q = (int8_t*)op.p[3];
    Q8Meta* meta = (Q8Meta*)op.p[4];
    const int n = op.i[0], ff = op.i[1];
    const int64_t used = (int64_t)c.P->n_u * ff / 32, per = used + (n - UMAX * ff) / 32;
    for (int64_t w = c.gw; w < c.nb * per; w += c.GW) {
        const int64_t b = w / per, r = w % per;
        const int64_t i = b * n + (r < used ? r * 32 : (int64_t)UMAX * ff + (r - used) * 32) + c.lane;
        const float a = tona::det::silu(g[i]) * u[i];  // as the CPU experts do, for bit-identical sums
        out[i] = a;
        wave_q8(a, i, q, meta);
    }
}

// LDS of op_route: top-k per token, the union of the experts and their weights
struct RouteLds {
    int* top;      // [BMAX][16]
    float* tw;     // [BMAX][16]
    int* u_of;     // E
    int* slot;     // UMAX
    uint32_t* um;  // UMAX
    float* ew;     // BMAX * UMAX
};
static_assert(1024 + BMAX * UMAX <= SM_FLOATS, "op_route must fit in sm");

// Softmax and top-k of the logits r of one token (E <= 512, E/32 values per lane in registers); ties go to the smaller
// index, the weights are renormalized later
constexpr int ROUTE_PER = 16;  // E <= 512

// Softmax of the logits r over E values spread over the lanes (j-th value of lane l at j * WAVE + l); unused slots
// get -1. Returns the sum of the exponentials. The logits are clamped (NaN = -FLT_MAX, +-inf = +-FLT_MAX): every
// exponential is a number between 0 and 1, the sum is at least 1, and a row of NaN selects the experts 0 ... K - 1
// instead of an index outside the experts. nan_n: NaN logits of this lane.
__device__ __forceinline__ float route_softmax(float (&v)[ROUTE_PER], const float* r, int E, int lane, int& nan_n) {
    const int per = E / WAVE;
    float mx = -INFINITY;
    nan_n = 0;
#pragma unroll
    for (int j = 0; j < ROUTE_PER; j++) {
        const float raw = j < per ? r[j * WAVE + lane] : -INFINITY;
        nan_n += raw != raw;
        v[j] = j < per ? logit_clamp(raw) : raw;
        mx = fmaxf(mx, v[j]);
    }
    mx = wave_max(mx);
    float s = 0;
#pragma unroll
    for (int j = 0; j < ROUTE_PER; j++) v[j] = j < per ? expf(v[j] - mx) : -1.0f, s += j < per ? v[j] : 0.0f;
    return wave_sum(s);
}

// Returns the NaN logits of the token's row
__device__ __forceinline__ int route_wave_topk(const RouteLds l, const float* r, int E, int K, int w, int lane) {
    constexpr int PER = ROUTE_PER;
    float v[PER];
    int nan_n;
    const float s = route_softmax(v, r, E, lane, nan_n);
    for (int k = 0; k < K; k++) {
        float bv = -2;
        int be = E;
#pragma unroll
        for (int j = 0; j < PER; j++)
            if (v[j] > bv) bv = v[j], be = j * WAVE + lane;  // j rises: on a tie the smaller index stays
        for (int off = WAVE / 2; off > 0; off /= 2) {
            const float ov = __shfl_xor(bv, off, WAVE);
            const int oe = __shfl_xor(be, off, WAVE);
            if (ov > bv || (ov == bv && oe < be)) bv = ov, be = oe;
        }
#pragma unroll
        for (int j = 0; j < PER; j++)
            if (j * WAVE + lane == be) v[j] = -2;
        if (lane == 0) l.top[w * 16 + k] = be, l.tw[w * 16 + k] = bv / s;
    }
    return (int)wave_sum((float)nan_n);
}

// Union of the selected experts over the tokens (thread 0 only): slot, token mask and renormalized weights.
// Returns the number of distinct experts.
__device__ __forceinline__ int route_union(const RouteLds l, const Ctx& c, int K, bool all) {
    int nu = 0;
    for (int b = all ? 0 : c.nb - 1; b < c.nb; b++) {
        float ws = 0;
        for (int k = 0; k < K; k++) ws += l.tw[b * 16 + k];
        for (int k = 0; k < K; k++) {
            const int e = l.top[b * 16 + k];
            if (l.u_of[e] < 0) l.u_of[e] = nu, l.slot[nu] = e, nu++;
            l.ew[b * UMAX + l.u_of[e]] = l.tw[b * 16 + k] / ws;
            l.um[l.u_of[e]] |= 1u << b;
        }
    }
    return nu;
}

// Routing on the GPU (MTP, all experts in VRAM, slot = expert): like Mega::select, softmax, top-k, weights
// renormalized, union over the tokens. Block 0, one wave per token. p[0]: router logits [nb][E]; i: E, k
__device__ __noinline__ void op_route(const MegaOp& op, const Ctx& c) {
    if (blockIdx.x != 0) return;
    const int E = op.i[0], K = op.i[1], w = threadIdx.x / WAVE, lane = c.lane;
    const RouteLds l{(int*)c.sm,         c.sm + BMAX * 16,        (int*)(c.sm + 256),
                     (int*)(c.sm + 768), (uint32_t*)(c.sm + 896), c.sm + 1024};
    for (int i = threadIdx.x; i < UMAX; i += THREADS) l.um[i] = 0, l.slot[i] = 0;
    for (int i = threadIdx.x; i < BMAX * UMAX; i += THREADS) l.ew[i] = 0;
    for (int e = threadIdx.x; e < E; e += THREADS) l.u_of[e] = -1;
    // Only the last token: nobody needs the output of the others (just their KV), and routing them would select up
    // to 80 experts. With several sequences (mtp_step_multi, one row each) all rows.
    const bool all = c.P->nseq > 0;
    int nan_row = 0;
    if (w == c.nb - 1 || (all && w < c.nb))
        nan_row = route_wave_topk(l, (const float*)op.p[0] + (int64_t)w * E, E, K, w, lane);
    // Every wave writes its slot, so no slot keeps the count of an earlier step; the CPU sums them
    if (op.p[1] && w < BMAX && lane == 0) ((int*)op.p[1])[w] = nan_row;
    lds_sync();
    // Union in LDS by thread 0 (<= 80 entries), then all threads copy it to P: writing every entry from thread 0
    // straight to device memory (umask |= ...) was slower.
    __shared__ int nu_s;
    if (threadIdx.x == 0) nu_s = route_union(l, c, K, all);
    lds_sync();
    for (int i = threadIdx.x; i < UMAX; i += THREADS) c.P->umask[i] = l.um[i], c.P->slot[i] = l.slot[i];
    for (int i = threadIdx.x; i < BMAX * UMAX; i += THREADS) c.P->ew[i] = l.ew[i];
    if (threadIdx.x == 0) c.P->n_u = nu_s, c.P->n_miss = 0, c.P->n_cpu = 0;
}

// ed: UMAX*d per token; sh, y: d per token; sh_inp: 32 per token. Experts that were not selected (weight 0) have no
// valid output and are skipped. The sum over the experts is fixed-point (det::to_fix); the CPU's experts are added as
// a fixed-point sum: bit-identical whichever side computes which expert.
__device__ __noinline__ void op_moe_sum(const MegaOp& op, const Ctx& c) {
    const float* ed = (const float*)op.p[0];
    const float* sh = (const float*)op.p[1];
    const float* sh_inp = (const float*)op.p[2];
    float* y = (float*)op.p[3];
    const int d = op.i[1], nu = c.P->n_u;
    // the weights once per block into LDS
    float* w_all = c.sm;
    for (int k = threadIdx.x; k < c.nb * UMAX; k += THREADS) w_all[k] = c.P->ew[k];
    lds_sync();
    for (int64_t t = c.gt; t < (int64_t)c.nb * d; t += c.GT) {
        const int b = (int)(t / d);
        const int64_t i = t % d;
        const float* w = w_all + b * UMAX;
        int64_t a = c.P->n_cpu ? ((const int64_t*)op.p[4])[t] : 0;  // experts of the CPU, already weighted
        for (int e = 0; e < nu; e++)
            if (w[e] != 0.0f) a += tona::det::to_fix(w[e] * ed[((int64_t)b * UMAX + e) * d + i]);
        y[t] = tona::det::from_fix(a) + sigm(sh_inp[b * 32]) * sh[t];
    }
}

// Runs the step if it is a MoE step; false for the other groups.
__device__ __forceinline__ bool dispatch_moe(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_ACT_QUANT:
            op_act_quant(op, c);
            return true;
        case OP_MOE_SUM:
            op_moe_sum(op, c);
            return true;
        case OP_ROUTE:
            op_route(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
