// The per-wave part of the sigmoid routing policy (backend/route_policy.h), shared by the megakernel step
// (mega_op_route_sig.h) and the prefill kernel (prefill_route_sig.hip): scores and choice keys of one token's logits in
// the registers of a wave and the pick of the best remaining expert. The rules are route_policy.h's, the functions
// the CPU side calls too. Needs no other header of the kernels (both families define limits of the same names).
#pragma once

#include <hip/hip_runtime.h>

#include "backend/route_policy.h"
#include "gpu/matvec_dev.h"

namespace tona::gpu {
namespace {

constexpr int SIG_PER = 16;  // slots per lane: experts up to 16 * 32 = 512

// Scores and choice keys of one token's logits; expert e = j * WAVE + lane sits in slot j of its lane
struct SigRow {
    float key[SIG_PER], score[SIG_PER];
    int bad;  // NaN and +inf logits of this lane (counted like backend::route_sigmoid)
};

// Any E up to 512, not only multiples of 32. bias null: none.
__device__ __forceinline__ void route_sig_row(SigRow& row, const float* r, const float* bias, int E) {
    const int lane = threadIdx.x % WAVE;
    row.bad = 0;
#pragma unroll
    for (int j = 0; j < SIG_PER; j++) {
        const int e = j * WAVE + lane;
        float s = 0.0f, k = -INFINITY;
        if (e < E) {
            row.bad += r[e] != r[e] || r[e] == INFINITY;
            s = tona::backend::sigmoid_route::score(r[e]);
            k = tona::backend::sigmoid_route::key(s, bias ? bias[e] : 0.0f);
        }
        row.score[j] = s, row.key[j] = k;
    }
}

// The best remaining expert of the wave: key, score and expert (larger key, on a tie the smaller index)
struct SigBest {
    float key, score;
    int expert;
};

// Returns the best remaining expert of the wave (the same in every lane) and takes it out of the row. The keys are
// finite (route_policy.h cleans them), so -inf marks a slot that is free or taken.
__device__ __forceinline__ SigBest route_sig_next(SigRow& row) {
    const int lane = threadIdx.x % WAVE;
    SigBest b{-INFINITY, 0.0f, 1 << 30};
#pragma unroll
    for (int j = 0; j < SIG_PER; j++)
        if (row.key[j] > b.key) b = {row.key[j], row.score[j], j * WAVE + lane};  // j rises: the smaller index stays
    for (int off = WAVE / 2; off > 0; off /= 2) {
        const float ok = __shfl_xor(b.key, off, WAVE), os = __shfl_xor(b.score, off, WAVE);
        const int oe = __shfl_xor(b.expert, off, WAVE);
        if (ok > b.key || (ok == b.key && oe < b.expert)) b = {ok, os, oe};
    }
#pragma unroll
    for (int j = 0; j < SIG_PER; j++)
        if (j * WAVE + lane == b.expert) row.key[j] = -INFINITY;
    return b;
}

}  // namespace
}  // namespace tona::gpu
