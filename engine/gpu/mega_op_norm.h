// Step functions for norms, hyper-connection streams and plain element-wise int8 steps.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// e: 2*d per token; R, xn, q: hc*d per token
__device__ __noinline__ void op_bcast_norm(const MegaOp& op, const Ctx& c) {
    const float* e = (const float*)op.p[0];
    float* R = (float*)op.p[1];
    float* xn = (float*)op.p[2];
    const float* w = (const float*)op.p[3];
    int8_t* q = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int hc = op.i[0], d = op.i[1];
    const int64_t n = (int64_t)hc * d;
    for (int b = 0; b < c.nb; b++) {
        const float* eb = e + (int64_t)b * 2 * d;
        float ss = 0;
        for (int i = threadIdx.x; i < d; i += THREADS) ss += eb[i] * eb[i];
        const float scale = rsqrtf(block_sum(ss, c.sm) / d + op.f[0]);
        for (int64_t j = c.gt; j < n; j += c.GT) {  // waves cover whole blocks of 32 (hc*d and GT are multiples)
            const float v = eb[j % d];
            R[b * n + j] = v;
            const float o = v * scale * w[j];
            xn[b * n + j] = o;
            if (q) wave_q8(o, j, q + b * n, meta + b * n / 32);
        }
    }
}

// Per (token, stream) values of op_combine_norm in LDS: coef = 2 sigma(inject/hc), scale = 1/RMS, part = partial sums.
struct CombineLds {
    float* coef;
    float* scale;
    float* part;
};
static_assert(BMAX * 4 <= 32, "coef and scale per (token, stream) for hc <= 4");

// 1/RMS of R + y * coef per (token, stream). wpc waves per (token, stream), all sums at the same time; every block
// computes them itself, which saves a barrier. Block sums one after the other took 2.94 instead of 1.92 ms per pass at
// 4 tokens (research/measurements/2026-10-23-verify-pass.md). wpc is derived from PNB, not nb, so the partial sums of
// a token do not depend on the width of the pass (bit-identical). Above PNB tokens a wave computes several sums.
struct CombineSums {
    const float* R;
    const float* y;
    CombineLds lds;
    int hc, d;
};
__device__ __forceinline__ void combine_scales(const CombineSums s, const MegaOp& op, const Ctx c) {
    const int64_t n = (int64_t)s.hc * s.d;
    const int nk = c.nb * s.hc, wpc = MWAVES / (PNB * s.hc), wv = threadIdx.x / WAVE, sub = wv % wpc;
    for (int k = wv / wpc; k < nk; k += MWAVES / wpc) {
        const int b = k / s.hc, g = k % s.hc;
        // float4 and unrolled: at 4 tokens one wave per sum; scalar, that was 80 loads awaited one after the other
        const float4* R4 = reinterpret_cast<const float4*>(s.R + b * n + (int64_t)g * s.d);
        const float4* y4 = s.y ? reinterpret_cast<const float4*>(s.y + (int64_t)b * s.d) : nullptr;
        const float cf = s.lds.coef[k];
        float ss = 0;
#pragma unroll 4
        for (int i = sub * WAVE + c.lane; i < s.d / 4; i += wpc * WAVE) {
            float4 v = R4[i];
            if (y4) {
                const float4 u = y4[i];
                v.x += u.x * cf, v.y += u.y * cf, v.z += u.z * cf, v.w += u.w * cf;
            }
            ss += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        ss = wave_sum(ss);
        if (c.lane == 0) s.lds.part[k * wpc + sub] = ss;
    }
    lds_sync();
    if ((int)threadIdx.x < nk) {
        float t = 0;
        for (int i = 0; i < wpc; i++) t += s.lds.part[threadIdx.x * wpc + i];
        s.lds.scale[threadIdx.x] = rsqrtf(t / s.d + op.f[0]);
    }
    lds_sync();
}

// Operands of op_combine_norm besides the sums
struct CombineOperands {
    const float *R, *y, *w;
    float *Rout, *xn;
    int8_t* q;
    Q8Meta* meta;
    int hc, d;
};

// Values of the first element of this thread (R, y and the norm gain), requested before the coefficients and the norm
// are computed so that their latency overlaps theirs (the same values go through the same operations as before:
// bit-identical)
struct CombineFirst {
    float r = 0.0f, y = 0.0f, w = 0.0f;
};
__device__ __forceinline__ CombineFirst combine_request_first(const CombineOperands& a, int64_t first, int64_t n) {
    CombineFirst f;
    f.r = a.R[first];
    if (a.w) f.w = a.w[first % n];
    if (a.y) f.y = a.y[(first / n) * a.d + first % n % a.d];
    return f;
}

__device__ __forceinline__ void combine_apply(const CombineOperands& a, const CombineLds& lds, const CombineFirst& f,
                                              const Ctx& c) {
    const int64_t n = (int64_t)a.hc * a.d, first = c.gt;
    for (int64_t t = first; t < (int64_t)c.nb * n; t += c.GT) {
        const int b = (int)(t / n);
        const int64_t j = t % n;
        const int k = b * a.hc + (int)(j / a.d);
        const bool pre = t == first;
        const float rv = pre ? f.r : a.R[t];
        const float v = rv + (a.y ? (pre ? f.y : a.y[(int64_t)b * a.d + j % a.d]) * lds.coef[k] : 0.0f);
        if (a.Rout) a.Rout[t] = v;
        if (a.w) {
            const float o = v * lds.scale[k] * (pre ? f.w : a.w[j]);
            a.xn[t] = o;
            if (a.q) wave_q8(o, t, a.q, a.meta);
        }
    }
}

// R, Rout, xn, q: hc*d per token; y: d per token; inject: 32 per token
__device__ __noinline__ void op_combine_norm(const MegaOp& op, const Ctx& c) {
    const CombineOperands a{(const float*)op.p[0],
                            (const float*)op.p[1],
                            (const float*)op.p[4],
                            (float*)op.p[3],
                            (float*)op.p[5],
                            (int8_t*)op.p[6],
                            (Q8Meta*)op.p[7],
                            op.i[0],
                            op.i[1]};
    const float* inject = (const float*)op.p[2];
    const int hc = a.hc, d = a.d;
    const CombineLds lds{c.sm, c.sm + 32, c.sm + 64};
    CombineFirst first;
    if (c.gt < (int64_t)c.nb * hc * d) first = combine_request_first(a, c.gt, (int64_t)hc * d);
    lds_sync();
    if ((int)threadIdx.x < c.nb * hc) {
        const int b = threadIdx.x / hc, g = threadIdx.x % hc;
        lds.coef[threadIdx.x] = a.y ? 2.0f * sigm(inject[b * 32 + g] / hc) : 0.0f;
    }
    lds_sync();
    if (a.w) combine_scales({a.R, a.y, lds, hc, d}, op, c);
    combine_apply(a, lds, first, c);
}

// xn, gate: hc*d per token; mixed, q: d per token
__device__ __noinline__ void op_hc_mix_quant(const MegaOp& op, const Ctx& c) {
    const float* xn = (const float*)op.p[0];
    const float* gate = (const float*)op.p[1];
    float* mixed = (float*)op.p[2];
    int8_t* q = (int8_t*)op.p[3];
    Q8Meta* meta = (Q8Meta*)op.p[4];
    const int hc = op.i[0], d = op.i[1];
    const int64_t n = (int64_t)hc * d;
    for (int64_t u = c.gw; u < (int64_t)c.nb * (d / 32); u += c.GW) {
        const int b = (int)(u / (d / 32));
        const int64_t i = (u % (d / 32)) * 32 + c.lane;
        float s = 0;
        for (int g = 0; g < hc; g++) s += xn[b * n + g * d + i] * sigm(gate[b * n + g * d + i]);
        const float m = s / hc;
        mixed[(int64_t)b * d + i] = m;
        wave_q8(m, (int64_t)b * d + i, q, meta);
    }
}

// x -> int8, n = i[0] per token (tokens lie n apart)
__device__ __noinline__ void op_quant(const MegaOp& op, const Ctx& c) {
    const float* x = (const float*)op.p[0];
    int8_t* q = (int8_t*)op.p[1];
    Q8Meta* meta = (Q8Meta*)op.p[2];
    const int n = op.i[0];
    for (int64_t u = c.gw; u < (int64_t)c.nb * (n / 32); u += c.GW) {
        const int64_t i = u * 32 + c.lane;
        wave_q8(x[i], i, q, meta);
    }
}

// R: hc*d per token; out: i[2] values per token, from column i[3]
__device__ __noinline__ void op_hc_mean(const MegaOp& op, const Ctx& c) {
    const float* R = (const float*)op.p[0];
    float* out = (float*)op.p[1];
    const int hc = op.i[0], d = op.i[1], os = op.i[2], o0 = op.i[3];
    for (int64_t t = c.gt; t < (int64_t)c.nb * d; t += c.GT) {
        const int64_t b = t / d, i = t % d;
        float s = 0;
        for (int g = 0; g < hc; g++) s += R[b * hc * d + g * d + i];
        out[b * os + o0 + i] = s * (1.0f / hc);
    }
}

// x, out, q: n = i[0] per token; every block computes the sum per token itself (out != x)
__device__ __noinline__ void op_norm(const MegaOp& op, const Ctx& c) {
    const float* x = (const float*)op.p[0];
    const float* w = (const float*)op.p[1];
    float* out = (float*)op.p[2];
    int8_t* q = (int8_t*)op.p[3];
    Q8Meta* meta = (Q8Meta*)op.p[4];
    const int n = op.i[0];
    for (int b = 0; b < c.nb; b++) {
        const float* xb = x + (int64_t)b * n;
        float ss = 0;
        for (int i = threadIdx.x; i < n; i += THREADS) ss += xb[i] * xb[i];
        const float scale = rsqrtf(block_sum(ss, c.sm) / n + op.f[0]);
        for (int64_t j = c.gt; j < n; j += c.GT) {
            const float o = xb[j] * scale * w[j];
            out[(int64_t)b * n + j] = o;
            if (q) wave_q8(o, (int64_t)b * n + j, q, meta);
        }
    }
}

// g, u, act, q: n = i[0] per token
__device__ __noinline__ void op_swiglu(const MegaOp& op, const Ctx& c) {
    const float* g = (const float*)op.p[0];
    const float* u = (const float*)op.p[1];
    float* out = (float*)op.p[2];
    int8_t* q = (int8_t*)op.p[3];
    Q8Meta* meta = (Q8Meta*)op.p[4];
    for (int64_t w = c.gw; w < (int64_t)c.nb * (op.i[0] / 32); w += c.GW) {
        const int64_t i = w * 32 + c.lane;
        const float a = silu_f(g[i]) * u[i];
        out[i] = a;
        wave_q8(a, i, q, meta);
    }
}

// Runs the step if it is a norm or element-wise int8 step; false for the other groups.
__device__ __forceinline__ bool dispatch_norm(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_BCAST_NORM:
            op_bcast_norm(op, c);
            return true;
        case OP_COMBINE_NORM:
            op_combine_norm(op, c);
            return true;
        case OP_HC_MIX_QUANT:
            op_hc_mix_quant(op, c);
            return true;
        case OP_QUANT:
            op_quant(op, c);
            return true;
        case OP_HC_MEAN:
            op_hc_mean(op, c);
            return true;
        case OP_NORM:
            op_norm(op, c);
            return true;
        case OP_SWIGLU:
            op_swiglu(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
