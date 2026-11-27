// Step functions of per-layer embeddings (PLE): gate and dilated conv.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// One block per (token, stream). R, key, gated, normed: hc*d per token; val: d per token
__device__ __noinline__ void op_ple_gate(const MegaOp& op, const Ctx& c) {
    const float* R = (const float*)op.p[0];
    const float* key = (const float*)op.p[1];
    const float* val = (const float*)op.p[2];
    const float* w_key = (const float*)op.p[3];
    const float* w_query = (const float*)op.p[4];
    const float* w_conv = (const float*)op.p[5];
    float* gated = (float*)op.p[6];
    float* normed = (float*)op.p[7];
    const int hc = op.i[0], d = op.i[1];
    const int64_t n = (int64_t)hc * d;
    const float eps = op.f[0];
    for (int gb = blockIdx.x; gb < hc * c.nb; gb += gridDim.x) {
        const int b = gb / hc, g = gb % hc;
        const float* kc = key + b * n + (int64_t)g * d;
        const float* rc = R + b * n + (int64_t)g * d;
        const float* vb = val + (int64_t)b * d;
        float* gt_ = gated + b * n + (int64_t)g * d;
        float* nm = normed + b * n + (int64_t)g * d;
        float sk = 0, sq = 0;
        for (int i = threadIdx.x; i < d; i += THREADS) sk += kc[i] * kc[i], sq += rc[i] * rc[i];
        const float fk = rsqrtf(block_sum(sk, c.sm) / d + eps), fq = rsqrtf(block_sum(sq, c.sm) / d + eps);
        float dot = 0;
        for (int i = threadIdx.x; i < d; i += THREADS)
            dot += (kc[i] * fk * w_key[g * d + i]) * (rc[i] * fq * w_query[g * d + i]);
        const float s = block_sum(dot, c.sm) * rsqrtf((float)d);
        const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
        const float sgn = (float)((s > 0) - (s < 0));
        const float gt = sigm(sgn * mag);
        float sg = 0;
        for (int i = threadIdx.x; i < d; i += THREADS) {
            const float v = vb[i] * gt;
            gt_[i] = v;
            sg += v * v;
        }
        const float fg = rsqrtf(block_sum(sg, c.sm) / d + eps);
        for (int i = threadIdx.x; i < d; i += THREADS) nm[i] = gt_[i] * fg * w_conv[g * d + i];
    }
}

// One thread per channel, all tokens one after the other; ring with (K − 1) · dil + BMAX places (see dn_conv1).
// R, gated, normed, Rout: n per token
__device__ __noinline__ void op_ple_conv(const MegaOp& op, const Ctx& c) {
    const float* R = (const float*)op.p[0];
    const float* gated = (const float*)op.p[1];
    const float* normed = (const float*)op.p[2];
    const float* w = (const float*)op.p[3];
    float* hist = (float*)op.p[4];
    float* Rout = (float*)op.p[5];
    const int n = op.i[0], K = op.i[1], dil = op.i[2];
    const int RG = (K - 1) * dil + BMAX;
    for (int64_t ch = c.gt; ch < n; ch += c.GT) {
        for (int b = 0; b < c.nb; b++) {
            const int64_t pos = c.pos + b, o = (int64_t)b * n + ch;
            float acc = 0;
            for (int k = 0; k < K; k++) {
                const int back = (K - 1 - k) * dil;
                float x;
                if (back == 0)
                    x = normed[o];
                else {
                    const int64_t p = pos - back;
                    x = p >= 0 ? hist[(p % RG) * (int64_t)n + ch] : 0.0f;
                }
                acc += w[ch * K + k] * x;
            }
            hist[(pos % RG) * (int64_t)n + ch] = normed[o];
            Rout[o] = R[o] + gated[o] + silu_f(acc);
        }
    }
}

// Runs the step if it is a PLE step; false for the other groups.
__device__ __forceinline__ bool dispatch_ple(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_PLE_GATE:
            op_ple_gate(op, c);
            return true;
        case OP_PLE_CONV:
            op_ple_conv(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
