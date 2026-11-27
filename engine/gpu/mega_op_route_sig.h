// Routing on the GPU with the sigmoid policy (backend/route_policy.h): sigmoid of the router logits, the n_used largest
// score + bias, weights = score / sum of the chosen scores * scale, union over the tokens into MegaParams. The rules
// come from route_policy.h, the same functions the CPU side (backend/expert_route.h) calls, so both sides compute the
// same bits. The step of the softmax policy (op_route in mega_op_moe.h) is the Coder's and stays as it is; this one
// is part of the kernels of architectures that route by sigmoid. E up to 512, any E (not only multiples of 32).
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc).
#pragma once

#include "backend/route_policy.h"
#include "gpu/mega_op_moe.h"
#include "gpu/route_sig_dev.h"

namespace tona::gpu {
namespace {

// Top-K of one token by key into l.top, the scores of the chosen into l.tw. Returns the unusable logits of the row.
__device__ __forceinline__ int route_sig_wave_topk(const RouteLds l, SigRow& row, int K, int w) {
    for (int k = 0; k < K; k++) {
        const SigBest b = route_sig_next(row);
        if (threadIdx.x % WAVE == 0) l.top[w * 16 + k] = b.expert, l.tw[w * 16 + k] = b.score;
    }
    return (int)wave_sum((float)row.bad);
}

// Union of the selected experts over the tokens (thread 0 only): as route_union, with the weight of the sigmoid policy.
// The sum of the scores runs in rank order from 0, as on the CPU.
__device__ __forceinline__ int route_sig_union(const RouteLds l, const Ctx& c, int K, bool all, float scale) {
    int nu = 0;
    for (int b = all ? 0 : c.nb - 1; b < c.nb; b++) {
        float ws = 0;
        for (int k = 0; k < K; k++) ws += l.tw[b * 16 + k];
        for (int k = 0; k < K; k++) {
            const int e = l.top[b * 16 + k];
            if (l.u_of[e] < 0) l.u_of[e] = nu, l.slot[nu] = e, nu++;
            l.ew[b * UMAX + l.u_of[e]] = tona::backend::sigmoid_route::weight(l.tw[b * 16 + k], ws, scale);
            l.um[l.u_of[e]] |= 1u << b;
        }
    }
    return nu;
}

// Block 0, one wave per token, like op_route. p[0]: router logits [nb][E], p[1]: unusable logits per token (or null),
// p[2]: selection bias [E] (or null); i[0] = E, i[1] = K; f[0] = scale of the weights.
__device__ __noinline__ void op_route_sigmoid(const MegaOp& op, const Ctx& c) {
    if (blockIdx.x != 0) return;
    const int E = op.i[0], K = op.i[1], w = threadIdx.x / WAVE;
    const RouteLds l{(int*)c.sm,         c.sm + BMAX * 16,        (int*)(c.sm + 256),
                     (int*)(c.sm + 768), (uint32_t*)(c.sm + 896), c.sm + 1024};
    for (int i = threadIdx.x; i < UMAX; i += THREADS) l.um[i] = 0, l.slot[i] = 0;
    for (int i = threadIdx.x; i < BMAX * UMAX; i += THREADS) l.ew[i] = 0;
    for (int e = threadIdx.x; e < E; e += THREADS) l.u_of[e] = -1;
    // Only the last token unless several sequences share the pass (their rows all), as op_route
    const bool all = c.P->nseq > 0;
    int bad = 0;
    if (w == c.nb - 1 || (all && w < c.nb)) {
        SigRow row;
        route_sig_row(row, (const float*)op.p[0] + (int64_t)w * E, (const float*)op.p[2], E);
        bad = route_sig_wave_topk(l, row, K, w);
    }
    if (op.p[1] && w < BMAX && c.lane == 0) ((int*)op.p[1])[w] = bad;
    lds_sync();
    __shared__ int nu_s;
    if (threadIdx.x == 0) nu_s = route_sig_union(l, c, K, all, op.f[0]);
    lds_sync();
    for (int i = threadIdx.x; i < UMAX; i += THREADS) c.P->umask[i] = l.um[i], c.P->slot[i] = l.slot[i];
    for (int i = threadIdx.x; i < BMAX * UMAX; i += THREADS) c.P->ew[i] = l.ew[i];
    if (threadIdx.x == 0) c.P->n_u = nu_s, c.P->n_miss = 0, c.P->n_cpu = 0;
}

// Runs the step if it is the sigmoid routing step; false for the other groups.
__device__ __forceinline__ bool dispatch_route_sigmoid(const MegaOp& op, const Ctx& c) {
    if (op.kind != OP_ROUTE_SIG) return false;
    op_route_sigmoid(op, c);
    return true;
}

}  // namespace
}  // namespace tona::gpu
