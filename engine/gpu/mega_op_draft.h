// Drafter step functions around the attention: dynamic convolution and norm + RoPE of q and k.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Convolution over the tokens of the block with kernel 2 and groups of i[1] channels (side i[2]): the weight of tap t
// for channel c = dyn[g + G·(t + 2·side)] + base[c + d·(t + 2·side)], g = c / group. in, out, q: d = i[0] per token;
// dyn: 4·G per token. i[3] = 1: out += convolution (residual)
__device__ __noinline__ void op_dconv(const MegaOp& op, const Ctx& c) {
    const float* in = (const float*)op.p[0];
    const float* dyn = (const float*)op.p[1];
    const float* base = (const float*)op.p[2];
    float* out = (float*)op.p[3];
    int8_t* q = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int d = op.i[0], grp = op.i[1], side = op.i[2], G = d / grp, P = 4 * G;
    for (int64_t t = c.gt; t < (int64_t)c.nb * d; t += c.GT) {
        const int b = (int)(t / d), ch = (int)(t % d), g = ch / grp;
        float acc = 0;
        for (int tap = 0; tap < 2 && tap <= b; tap++)
            acc += (dyn[(int64_t)b * P + g + G * (tap + 2 * side)] + base[ch + d * (tap + 2 * side)]) *
                   in[(int64_t)(b - tap) * d + ch];
        const float o = op.i[3] ? out[t] + acc : acc;
        out[t] = o;
        if (q) wave_q8(o, t, q, meta);
    }
}

// One wave: RMSNorm ⊙ w over 128 values, then NeoX RoPE over all 128 (pairs i, i + 64)
template <typename T>
__device__ __forceinline__ void wave_norm_rope128(const float* src, T* dst, const float* w, int64_t pos, float base,
                                                  float eps) {
    const int lane = threadIdx.x % WAVE;
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] = src[lane + 32 * k], ss += v[k] * v[k];
    const float scale = rsqrtf(wave_sum(ss) / 128 + eps);
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] *= scale * w[lane + 32 * k];
    rope_pair(v[0], v[2], lane, 128, pos, base);
    rope_pair(v[1], v[3], lane + 32, 128, pos, base);
#pragma unroll
    for (int k = 0; k < 4; k++) dst[lane + 32 * k] = (T)v[k];
}

// i[2] = 0: block at pos, q heads in place, K/V at pos + b; i[2] = 1: only K/V at pos − commit + b (context).
// q: nh*128 per token; k, v: nkv*128 per token; Kc, Vc by position, float or (T = _Float16, i[4] = 1) half
template <typename T>
__device__ void op_dqk_t(const MegaOp& op, const Ctx& c) {
    float* q = (float*)op.p[0];
    const float* q_norm = (const float*)op.p[1];
    const float* k = (const float*)op.p[2];
    const float* v = (const float*)op.p[3];
    const float* k_norm = (const float*)op.p[4];
    T* Kc = (T*)op.p[5];
    T* Vc = (T*)op.p[6];
    const int nkv = op.i[1], ctx = op.i[2], nh = ctx ? 0 : op.i[0], hd = 128;
    const int64_t ring = op.i[3];  // places in the KV ring (window + buffer)
    const int64_t p0 = ctx ? c.pos - c.P->commit : c.pos;
    const int per = nh + nkv;
    for (int64_t u0 = c.gw; u0 < (int64_t)c.nb * per; u0 += c.GW) {
        const int b = (int)(u0 / per), u = (int)(u0 % per);
        const int64_t pos = p0 + b;
        if (u < nh) {
            float* qh = q + ((int64_t)b * op.i[0] + u) * hd;
            wave_norm_rope128(qh, qh, q_norm, pos, op.f[1], op.f[0]);
        } else {
            const int h = u - nh;
            const int64_t slot = pos % ring;
            wave_norm_rope128(k + ((int64_t)b * nkv + h) * hd, Kc + (slot * nkv + h) * hd, k_norm, pos, op.f[1],
                              op.f[0]);
            for (int i = c.lane; i < hd; i += WAVE)
                Vc[(slot * nkv + h) * hd + i] = (T)v[((int64_t)b * nkv + h) * hd + i];
        }
    }
}

__device__ __noinline__ void op_dqk(const MegaOp& op, const Ctx& c) {
    if (op.i[4])
        op_dqk_t<_Float16>(op, c);
    else
        op_dqk_t<float>(op, c);
}

}  // namespace
}  // namespace tona::gpu
