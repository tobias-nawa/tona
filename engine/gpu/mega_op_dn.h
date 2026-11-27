// Step functions of the DeltaNet layers: causal conv, delta rule, gated norm.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Causal conv over K positions. The ring holds H + BMAX positions (index position mod R, R = K - 1 + BMAX), so tokens
// of a pass that are later discarded cannot overwrite inputs the next pass still reads. b is the token in the pass:
// earlier tokens of the same pass come from qkv instead of the ring, so the tokens run independently of each other.
struct DnConvIn {
    const float* qkv;
    float* ring;
    const float* w;
    int ch, K;
};
__device__ __forceinline__ float dn_conv1(const DnConvIn in, int c, int64_t pos, int b) {
    const float *qkv = in.qkv, *w = in.w;
    float* ring = in.ring;
    const int ch = in.ch, K = in.K;
    const int H = K - 1, R = H + BMAX;
    float acc = 0;
    for (int k = 0; k < H; k++) {
        const int64_t p = pos - (H - k);
        const int back = H - k;  // this many positions before the token
        acc += w[c * K + k] * (back <= b ? qkv[c - (int64_t)back * ch] : p >= 0 ? ring[(p % R) * ch + c] : 0.0f);
    }
    acc += w[c * K + H] * qkv[c];
    ring[(pos % R) * ch + c] = qkv[c];
    return silu_f(acc);
}

// qkv, out: ch = 2*kd + vd per token; ring see dn_conv1. Unit = (q/k head or 32 v channels, token). The q/k heads
// come first: they are the long units, and one wave doing all tokens one after the other was the bottleneck.
__device__ __noinline__ void op_dn_conv(const MegaOp& op, const Ctx& c) {
    const float* qkv = (const float*)op.p[0];
    float* ring = (float*)op.p[1];
    const float* w = (const float*)op.p[2];
    float* out = (float*)op.p[3];
    const int kd = op.i[0], vd = op.i[1], K = op.i[2], nk = op.i[4];
    const int ch = 2 * kd + vd;
    const int64_t nqk = (int64_t)2 * nk * c.nb, n = nqk + (int64_t)(vd / 32) * c.nb;
    auto unit = [&](int64_t g) {
        const bool qk = g < nqk;
        const int64_t u = qk ? g / c.nb : 2 * nk + (g - nqk) / c.nb;
        const int b = (int)((qk ? g : g - nqk) % c.nb);
        const float* xb = qkv + (int64_t)b * ch;
        float* ob = out + (int64_t)b * ch;
        if (qk) {
            float o[DS / 32], ss = 0;
#pragma unroll
            for (int k = 0; k < DS / 32; k++) {
                o[k] = dn_conv1({xb, ring, w, ch, K}, (int)u * DS + c.lane + 32 * k, c.pos + b, b);
                ss += o[k] * o[k];
            }
            // L2 norm as rms_norm(x, eps/n) * 1/sqrt(n)
            const float scale = rsqrtf(wave_sum(ss) / DS + op.f[0] / DS) * rsqrtf((float)DS);
#pragma unroll
            for (int k = 0; k < DS / 32; k++) ob[u * DS + c.lane + 32 * k] = o[k] * scale;
        } else {
            const int ci = 2 * kd + (int)(u - 2 * nk) * 32 + c.lane;
            ob[ci] = dn_conv1({xb, ring, w, ch, K}, ci, c.pos + b, b);
        }
    };
    // p[12]: assignment of the units to the waves (a dependency instead of a barrier behind dn.in, MegaOp::nobar = 2):
    // per width [GW + 1 starts][units], stride i[5]; every unit sits on a block that computes its rows itself
    if (const int* tb = (const int*)op.p[12]) {
        tb += (int64_t)(c.nb - 1) * op.i[5];
        for (int k = tb[c.gw]; k < tb[c.gw + 1]; k++) unit(tb[c.GW + 1 + k]);
        return;
    }
    for (int64_t g = c.gw; g < n; g += c.GW) unit(g);
}

// Delta rule, one wave per (V head j, DN_CW columns); lane = column v of the state S (128 x 128 per head), the rows
// spread over DN_PARTS lanes. Per token: δ_b = β_b (v_b − α_b S_{b−1}ᵀk_b), S_b = α_b S_{b−1} + k_b δ_bᵀ,
// o_b = S_bᵀ q_b / √128. The stored state lags one pass behind: this pass first commits the `commit` accepted tokens
// of the previous pass (from their δ, k, α), token by token as in that pass. δ, α, k of this pass go into the half
// `parity` of the intermediate buffers (pd: vd per token, pa: nv per token, pk: kd per token); the other half is read.
// cv: ch per token; beta/alpha: nv per token; raw: vd per token.
//
// DN_CW columns per wave, 32 / DN_CW lanes per column with a part of the rows each: with 32 columns per wave only
// 192 of 512 waves were busy, each working through 128 rows one after the other. The waves of a block share DN_HPB
// heads; their k, q and old k lie in LDS instead of being loaded per row from the L2.
// DN_CW = 8 (TONA_DN_CW): every lane holds DS / DN_PARTS = 32 rows of its column in registers; with 16 it would be 64,
// which spill (scratch of k_mega<4> 200 -> 480 bytes per lane, server pass +1.6 ms,
// research/measurements/2026-10-25-bit-exact.md).
#ifndef TONA_DN_CW
#define TONA_DN_CW 8
#endif
constexpr int DN_CW = TONA_DN_CW, DN_PARTS = 32 / DN_CW, DN_CH = DS / DN_CW, DN_HPB = MWAVES / DN_CH;
constexpr int DN_RW = DS / DN_PARTS;  // rows of a column per lane

// softplus(x), linear above 20 where log1p(e^x) equals x in float
__device__ __forceinline__ float dn_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// Dot product over the rows of the column: 4 partial sums per lane in a fixed order, then across the DN_PARTS lanes
__device__ __forceinline__ float dn_dot(const float (&s)[DN_RW], const float* x) {
    float t[4] = {0, 0, 0, 0};
#pragma unroll
    for (int u = 0; u < DN_RW; u++) t[u % 4] = fmaf(s[u], x[u], t[u % 4]);
    float r = (t[0] + t[1]) + (t[2] + t[3]);
    for (int off = DN_CW; off < WAVE; off *= 2) r += __shfl_xor(r, off, WAVE);
    return r;
}

// The state column back to memory
__device__ __forceinline__ void dn_store(float* st, const float (&s)[DN_RW], int r0) {
#pragma unroll
    for (int u = 0; u < DN_RW; u++) st[(int64_t)(r0 + u) * DS] = s[u];
}

// k, q and the old k (the commit's k_i) of the DN_HPB heads of the block go to LDS, [DN_HPB][BMAX][DS] each
struct DnStage {
    const float *cv, *ok;
    float *lk, *lq, *lo;
    int nk, nv, kd, ch, commit, nb;
};

__device__ __forceinline__ void dn_stage(const DnStage g, int64_t base) {
    for (int t = threadIdx.x; t < DN_HPB * BMAX * DS; t += THREADS) {
        const int hh = t / (BMAX * DS), b = (t / DS) % BMAX, x = t % DS;
        const int64_t jh = base / DN_CH + hh;
        if (jh >= g.nv) continue;
        const int kh = (int)(jh % g.nk);
        if (b < g.nb)
            g.lk[t] = g.cv[(int64_t)b * g.ch + g.kd + kh * DS + x], g.lq[t] = g.cv[(int64_t)b * g.ch + kh * DS + x];
        if (b < g.commit) g.lo[t] = g.ok[(int64_t)b * g.kd + kh * DS + x];
    }
}

// What token b of a column writes: o and δ for the first part of the rows, α and the k of the head for the first
// column chunk. nd, na, nk are the buffer half of this pass.
struct DnOut {
    float *raw, *nd, *na, *nk;
    int vd, nv, kd, n_k;
};

// Column of a wave: V head j, K head kh, column vi of chunk `chunk`, rows from part * DN_RW
struct DnCol {
    int j, vi, chunk, part, kh;
};

// Results of token b for the column
struct DnTok {
    float o, dl, alpha;
};

__device__ __forceinline__ void dn_emit(const DnOut g, const Ctx& c, const DnCol col, const float* hk, int b,
                                        const DnTok t) {
    const int j = col.j, vi = col.vi;
    if (col.part == 0) {
        g.raw[(int64_t)b * g.vd + j * DS + vi] = t.o * rsqrtf((float)DS);
        g.nd[(int64_t)b * g.vd + j * DS + vi] = t.dl;
    }
    if (col.chunk == 0 && c.lane == 0) g.na[b * g.nv + j] = t.alpha;
    if (col.chunk == 0 && j < g.n_k) {
#pragma unroll
        for (int x = 0; x < DS / 32; x++)
            g.nk[(int64_t)b * g.kd + col.kh * DS + c.lane + 32 * x] = hk[b * DS + c.lane + 32 * x];
    }
}

// Delta rule, one wave per (V head j, DN_CW columns); lane = column v of the state S (128 x 128 per head), the rows
// spread over DN_PARTS lanes. Per token: δ_b = β_b (v_b − α_b S_{b−1}ᵀk_b), S_b = α_b S_{b−1} + k_b δ_bᵀ,
// o_b = S_bᵀ q_b / √128. The stored state lags one pass behind: this pass first commits the `commit` accepted tokens
// of the previous pass (from their δ, k, α), token by token as in that pass. δ, α, k of this pass go into the half
// `parity` of the intermediate buffers (pd: vd per token, pa: nv per token, pk: kd per token); the other half is read.
// cv: ch per token; beta/alpha: nv per token; raw: vd per token.
//
// Every lane holds DN_RW = 32 rows of its column in registers (see DN_CW above). Every token takes the same steps
// wherever it sits in a pass and however many tokens were committed before it: bit-identical across the width of the
// verify pass.
template <int NB>
__device__ __noinline__ void op_dn_recur(const MegaOp& op, const Ctx& c) {
    const float* cv = (const float*)op.p[0];
    const float* bt = (const float*)op.p[1];
    const float* al = (const float*)op.p[2];
    const float* dt_bias = (const float*)op.p[3];
    const float* A = (const float*)op.p[4];
    float* pd = (float*)op.p[5];
    float* pa = (float*)op.p[6];
    float* state = (float*)op.p[7];
    float* raw = (float*)op.p[8];
    float* pk = (float*)op.p[9];
    const int nk = op.i[0], nv = op.i[1], kd = op.i[3], vd = nv * DS, ch = 2 * kd + vd;
    const int m = c.commit, par = c.parity, nb = c.nb;
    const float* ok = pk + (int64_t)(par ^ 1) * BMAX * kd;
    const float* od = pd + (int64_t)(par ^ 1) * BMAX * vd;
    const float* oa = pa + (int64_t)(par ^ 1) * BMAX * nv;
    float* nk_ = pk + (int64_t)par * BMAX * kd;
    float* nd = pd + (int64_t)par * BMAX * vd;
    float* na = pa + (int64_t)par * BMAX * nv;
    const int part = c.lane / DN_CW, wv = threadIdx.x / WAVE;
    float* lk = c.sm;                     // [DN_HPB][BMAX][DS]
    float* lq = lk + DN_HPB * BMAX * DS;  // [DN_HPB][BMAX][DS]
    float* lo = lq + DN_HPB * BMAX * DS;  // [DN_HPB][BMAX][DS]
    const DnStage stage{cv, ok, lk, lq, lo, nk, nv, kd, ch, m, nb};
    const DnOut out{raw, nd, na, nk_, vd, nv, kd, nk};
    const int64_t total = (int64_t)nv * DN_CH;
    for (int64_t base = (int64_t)blockIdx.x * MWAVES; base < total; base += c.GW) {
        lds_sync();
        dn_stage(stage, base);
        lds_sync();
        const int64_t u = base + wv;
        if (u >= total) continue;
        const int j = (int)(u / DN_CH), chunk = (int)(u % DN_CH), vi = chunk * DN_CW + c.lane % DN_CW, hh = wv / DN_CH;
        const int kh = j % nk;  // GGUF: V head j uses K head j mod nk
        const float* hk = lk + hh * BMAX * DS;
        const float* hq = lq + hh * BMAX * DS;
        const int r0 = part * DN_RW;
        float* st = state + (int64_t)j * DS * DS + vi;
        float s[DN_RW];
#pragma unroll
        for (int u = 0; u < DN_RW; u++) s[u] = st[(int64_t)(r0 + u) * DS];
        const float* lold = lo + hh * BMAX * DS + r0;
#pragma unroll
        for (int i = 0; i < BMAX; i++) {
            if (i >= m) break;
            const float a = oa[i * nv + j], d = od[(int64_t)i * vd + j * DS + vi];
#pragma unroll
            for (int u = 0; u < DN_RW; u++) s[u] = fmaf(lold[i * DS + u], d, a * s[u]);
        }
        if (m > 0) dn_store(st, s, r0);
        // Above 4 tokens a loop: unrolled 8 times the compiler held the loads of all tokens at once and spilled
        constexpr int kUnroll = NB > 4 ? 1 : NB;
#pragma unroll kUnroll
        for (int b = 0; b < NB; b++) {
            if (b >= nb) continue;
            const float beta = sigm(bt[b * nv + j]);
            const float xa = al[b * nv + j] + dt_bias[j];
            const float sp = dn_softplus(xa);
            const float alpha = expf(sp * A[j]);
            const float* kb = hk + b * DS + r0;
            const float sk = dn_dot(s, kb);  // S_{b−1}ᵀk_b
            const float dl = beta * fmaf(-alpha, sk, cv[(int64_t)b * ch + 2 * kd + j * DS + vi]);
#pragma unroll
            for (int u = 0; u < DN_RW; u++) s[u] = fmaf(kb[u], dl, alpha * s[u]);
            const float o = dn_dot(s, hq + b * DS + r0);  // S_bᵀq_b
            dn_emit(out, c, DnCol{j, vi, chunk, part, kh}, hk, b, DnTok{o, dl, alpha});
        }
    }
}

// What multiplies the normed head: σ(z) (the Coder) or silu(z) (Ornith). A template parameter, so a model's kernel
// holds only its own gate and the Coder's machine code does not change.
enum class DnGate { kSigmoid, kSilu };

template <DnGate Gate>
__device__ __forceinline__ float dn_gate(float z) {
    if constexpr (Gate == DnGate::kSilu)
        return silu_f(z);
    else
        return sigm(z);
}

// One wave per (token, V head, 32 values): norm over the whole head (reads the 128 raw values), ⊙ w ⊙ gate(z), int8.
// raw, z: vd per token; o, q: i[1] per token
template <DnGate Gate>
__device__ __noinline__ void op_dn_norm(const MegaOp& op, const Ctx& c) {
    const float* raw = (const float*)op.p[0];
    const float* w_norm = (const float*)op.p[1];
    const float* z = (const float*)op.p[2];
    float* o = (float*)op.p[3];
    int8_t* q8 = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int nv = op.i[0], vd = nv * DS;
    const int64_t os = op.i[1];
    for (int64_t u = c.gw; u < (int64_t)c.nb * nv * (DS / 32); u += c.GW) {
        const int b = (int)(u / (nv * (DS / 32)));
        const int64_t r = u % (nv * (DS / 32));
        const int j = (int)(r / (DS / 32)), vi = (int)(r % (DS / 32)) * 32 + c.lane;
        const float* rb = raw + (int64_t)b * vd;
        float ss = 0;
#pragma unroll
        for (int k = 0; k < DS / 32; k++) {
            const float v = rb[j * DS + c.lane + 32 * k];
            ss += v * v;
        }
        const float ov = rb[j * DS + vi] * rsqrtf(wave_sum(ss) / DS + op.f[0]) * w_norm[vi] *
                         dn_gate<Gate>(z[(int64_t)b * vd + j * DS + vi]);
        o[b * os + j * DS + vi] = ov;
        wave_q8(ov, b * os + j * DS + vi, q8, meta);
    }
}

// Runs the step if it is a DeltaNet step; false for the other groups.
template <int NB, DnGate Gate = DnGate::kSigmoid>
__device__ __forceinline__ bool dispatch_dn(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_DN_CONV:
            op_dn_conv(op, c);
            return true;
        case OP_DN_RECUR:
            op_dn_recur<NB>(op, c);
            return true;
        case OP_DN_NORM:
            op_dn_norm<Gate>(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
