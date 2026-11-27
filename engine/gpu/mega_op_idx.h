// Step function of the QSA indexer score (OP_IDX_SCORE). A header of its own so that tests/test_gpu_topk.hip can run it
// without the rest of the QSA steps. Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not
// include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Sums of the four heads' partial dots across the wave, the same additions in the same tree as four wave_sum calls
// (xor butterfly 16, 8, 4, 2, 1; an addition gives the same bits whichever operand is first), but the first two stages
// exchange half and a quarter of the values: afterwards lane l holds the sum of head 2 * bit4(l) + bit3(l) in every
// lane of its group of 8. 10 lane exchanges for 4 heads instead of 20.
__device__ __forceinline__ void heads4_sum(const float (&d)[4], float (&out)[4]) {
    const int lane = threadIdx.x % WAVE;
    const bool hi16 = lane & 16, hi8 = lane & 8;
    float keep0 = hi16 ? d[2] : d[0], keep1 = hi16 ? d[3] : d[1];
    const float send0 = hi16 ? d[0] : d[2], send1 = hi16 ? d[1] : d[3];
    keep0 += __shfl_xor(send0, 16, WAVE);
    keep1 += __shfl_xor(send1, 16, WAVE);
    float s = hi8 ? keep1 : keep0;
    s += __shfl_xor(hi8 ? keep0 : keep1, 8, WAVE);
    for (int off = 4; off > 0; off >>= 1) s += __shfl_xor(s, off, WAVE);
    for (int h = 0; h < 4; h++) out[h] = __shfl(s, 16 * (h >> 1) + 8 * (h & 1), WAVE);
}

// The pool keys of the pass are scored four at a time: a wave keeps the q of one token (4 heads, lane l holds the
// dimensions l + 32 k) in registers for all of them, and has their pool loads in flight together; one wave on its own
// has too few loads outstanding to cover the latency of the pool (research/measurements/2026-10-29-mega-topk.md)
constexpr int IDX_GROUP = 4, IDX_HEADS = 4, IDX_DIM = 128;

// Pool blocks each token of the pass scores: all it sees, or none if it does not select (qsa_select). Divided once per
// step: 64-bit divisions per (token, block) cost more than the pool read
// (research/measurements/2026-10-29-mega-topk.md)
struct IdxLimits {
    int n[BMAX];
};
__device__ __forceinline__ IdxLimits idx_limits(const MegaOp& op, const Ctx& c) {
    const int blk = op.i[2], n_top = op.i[7] / blk;
    IdxLimits lim{};
#pragma unroll
    for (int b = 0; b < BMAX; b++) {
        const int n_pool = b < c.nb ? (int)(c.pos + b + 1) / blk : 0;
        lim.n[b] = n_pool > n_top ? n_pool : 0;
    }
    return lim;
}

// Score of one (token, pool block): per head the dot product of q and the pool key, ReLU, summed over the heads in
// order. qv: q of the token [head][k], pv: the key in the lanes [k]
__device__ __forceinline__ float idx_score_token(const float (&qv)[IDX_HEADS][4], const float (&pv)[4]) {
    float dot[IDX_HEADS], sum[IDX_HEADS], total = 0;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; h++) {
        dot[h] = 0;
#pragma unroll
        for (int k = 0; k < 4; k++) dot[h] += qv[h][k] * pv[k];
    }
    heads4_sum(dot, sum);
#pragma unroll
    for (int h = 0; h < IDX_HEADS; h++) total += fmaxf(sum[h], 0.0f);
    return total * rsqrtf((float)IDX_DIM);
}

// Scores of the pool blocks pb0 + j * GW (j < IDX_GROUP, those below the limit of the token) for every token
__device__ __forceinline__ void idx_score_group(const MegaOp& op, const Ctx& c, const IdxLimits& lim, int pb0,
                                                const float (&pv)[IDX_GROUP][4]) {
    const float* q = (const float*)op.p[0];
    float* score = (float*)op.p[2];
    const int sstride = op.i[3];
#pragma unroll
    for (int b = 0; b < BMAX; b++) {
        if (b >= c.nb || pb0 >= lim.n[b]) continue;
        float qv[IDX_HEADS][4];
#pragma unroll
        for (int h = 0; h < IDX_HEADS; h++)
#pragma unroll
            for (int k = 0; k < 4; k++) qv[h][k] = q[((int64_t)b * IDX_HEADS + h) * IDX_DIM + c.lane + WAVE * k];
#pragma unroll
        for (int j = 0; j < IDX_GROUP; j++) {
            const int pb = pb0 + j * (int)c.GW;
            if (pb >= lim.n[b]) continue;
            const float total = idx_score_token(qv, pv[j]);
            if (c.lane == 0) score[(int64_t)b * sstride + pb] = total;
        }
    }
}

// q: 4 heads * 128 per token; score: i[3] values per token. One wave per pool block for all tokens of the pass: the
// pool (f32, 32 MB per layer at 250K) is read once, not once per token; every score is the same sum in the same order
// as when each token walked the pool alone. The heads and the dimension are those of the model (checked at load).
__device__ __noinline__ void op_idx_score(const MegaOp& op, const Ctx& c) {
    const float* pool = (const float*)op.p[1];
    const IdxLimits lim = idx_limits(op, c);
    const int per =
        c.nb > 0 ? lim.n[c.nb - 1] : 0;  // the last token sees the most blocks, and selects first if any does
    for (int pb0 = (int)c.gw; pb0 < per; pb0 += (int)c.GW * IDX_GROUP) {
        float pv[IDX_GROUP][4];
#pragma unroll
        for (int j = 0; j < IDX_GROUP; j++) {
            const int pb = std::min(pb0 + j * (int)c.GW, per - 1);  // beyond the end: a block read twice, not scored
#pragma unroll
            for (int k = 0; k < 4; k++) pv[j][k] = pool[(int64_t)pb * IDX_DIM + c.lane + WAVE * k];
        }
        idx_score_group(op, c, lim, pb0, pv);
    }
}

}  // namespace
}  // namespace tona::gpu
