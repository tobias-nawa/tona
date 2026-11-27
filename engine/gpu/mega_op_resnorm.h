// Residual step of models with one plain residual stream (Ornith, Bolt): R' = R + y, xn = RMSNorm(R') * w, int8 for the
// next matvec. The Coder's steps (OP_COMBINE_NORM, OP_HC_MIX_QUANT in mega_op_norm.h) assume four gated streams and a
// per-stream sigmoid; this one has none of that. Without y it is the norm of the embedding or of the last residual
// (R' = R). Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); only the profiles of
// architectures with a plain residual include it.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Waves per token for the sum of squares: derived from PNB, not from the width of the pass, so the partial sums of a
// token (and its rounding) do not depend on how many tokens the pass has. Above PNB tokens a wave does several tokens.
constexpr int RES_WPT = MWAVES / PNB;
static_assert(BMAX * RES_WPT <= 64 && BMAX <= 32, "the partial sums and the scales of all tokens fit the front of sm");

struct ResOperands {
    const float *R, *y, *w;  // R: input [nb][d]; y: added to it (null: none); w: gain of the norm (null: no norm)
    float *Rout, *xn;        // R' (null: not written; may be R itself), norm output as float (null: none)
    int8_t* q;               // norm output as int8 with its scales (null: none)
    Q8Meta* meta;
    int d;
};

// 1/RMS of R + y per token. Every block computes the sums itself, which saves a barrier (see combine_scales: block sums
// one after the other cost 50 % more at 4 tokens). Result in scale[b]; part: the partial sums.
__device__ __forceinline__ void res_scales(const ResOperands& a, const Ctx& c, float eps, float* part, float* scale) {
    const int wv = threadIdx.x / WAVE, sub = wv % RES_WPT;
    for (int b = wv / RES_WPT; b < c.nb; b += MWAVES / RES_WPT) {
        const float4* R4 = reinterpret_cast<const float4*>(a.R + (int64_t)b * a.d);
        const float4* y4 = a.y ? reinterpret_cast<const float4*>(a.y + (int64_t)b * a.d) : nullptr;
        float ss = 0;
#pragma unroll 4
        for (int i = sub * WAVE + c.lane; i < a.d / 4; i += RES_WPT * WAVE) {
            float4 v = R4[i];
            if (y4) {
                const float4 u = y4[i];
                v.x += u.x, v.y += u.y, v.z += u.z, v.w += u.w;
            }
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = wave_sum(ss);
        if (c.lane == 0) part[b * RES_WPT + sub] = ss;
    }
    lds_sync();
    if ((int)threadIdx.x < c.nb) {
        float t = 0;
        for (int i = 0; i < RES_WPT; i++) t += part[threadIdx.x * RES_WPT + i];
        scale[threadIdx.x] = rsqrtf(t / a.d + eps);
    }
    lds_sync();
}

// op: p[0] R, p[1] y, p[2] w, p[3] Rout, p[4] xn, p[5] int8, p[6] meta; i[0] = d (a multiple of 128); f[0] = eps.
// Rout may be R: all sums are complete (barrier) before the first write.
__device__ __noinline__ void op_res_norm(const MegaOp& op, const Ctx& c) {
    const ResOperands a{(const float*)op.p[0], (const float*)op.p[1], (const float*)op.p[2], (float*)op.p[3],
                        (float*)op.p[4],       (int8_t*)op.p[5],      (Q8Meta*)op.p[6],      op.i[0]};
    float* scale = c.sm + 64;
    if (a.w) res_scales(a, c, op.f[0], c.sm, scale);
    for (int64_t t = c.gt; t < (int64_t)c.nb * a.d; t += c.GT) {  // whole waves: nb * d and GT are multiples of 32
        const float v = a.R[t] + (a.y ? a.y[t] : 0.0f);
        if (a.Rout) a.Rout[t] = v;
        if (!a.w) continue;
        const float o = v * scale[t / a.d] * a.w[t % a.d];
        if (a.xn) a.xn[t] = o;
        if (a.q) wave_q8(o, t, a.q, a.meta);
    }
}

// Runs the step if it is a plain residual step; false for the other groups.
__device__ __forceinline__ bool dispatch_resnorm(const MegaOp& op, const Ctx& c) {
    if (op.kind != OP_RES_NORM) return false;
    op_res_norm(op, c);
    return true;
}

}  // namespace
}  // namespace tona::gpu
