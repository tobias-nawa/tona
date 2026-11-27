// Full-context attention of the megakernel (Ornith, Bolt): every cached position is attended, no indexer, no top-k.
// Three steps: prepare (split q and gate, optional QK norm and RoPE, K and V into the int8 cache), split-K partial
// results per (token, KV head, chunk of positions) on WMMA, merge of the partials with the optional sigmoid gate. The
// head size, the gate, the QK norm and the RoPE are template parameters (FattnCfg), so a model's kernel contains only
// what it runs. The tile is shared with the prefill kernel (fattn_tile.h). Part of the kernel translation unit
// (arch/<name>/kernel.hip, no -fgpu-rdc); only the profiles of architectures with full attention include it.
#pragma once

#include "gpu/fattn_tile.h"
#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

struct FattnCfg {
    int hd;        // head size: 128 or 256
    bool gate;     // attn_q holds [q, gate] per head; the output is multiplied by sigmoid(gate)
    bool qk_norm;  // RMSNorm of the q and k heads with their weights, before RoPE
    bool rope;     // NeoX RoPE on the first 64 dimensions (the YaRN table of the unit), q and k
};

// Chunks per (token, head) and their smallest size. The size depends on the number of positions only, never on the
// tokens in the pass, so a token's rounding does not depend on the width of the pass. FA_SPL bounds the partial
// buffer: nb * nh * FA_SPL * (hd + 2) floats whatever the context (262,144 positions: 4,096 per chunk).
constexpr int FA_SPL = 64, FA_CH_MIN = 64;

__device__ __forceinline__ int fa_chunk(int64_t n_cells) {
    const int64_t per = (n_cells + FA_SPL - 1) / FA_SPL;
    return (int)std::max<int64_t>(FA_CH_MIN, (per + fa::CC - 1) / fa::CC * fa::CC);
}

// Values of the lane: v[k] = src[lane + 32 k], then optionally RMSNorm ⊙ w and RoPE (pairs lane, lane + 32)
template <int NV, bool NORM, bool ROPE>
__device__ __forceinline__ void prep_vec(const float* src, const float* w, int64_t pos, const MegaOp& op,
                                         float (&v)[NV]) {
    const int lane = threadIdx.x % WAVE;
    float ss = 0;
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] = src[lane + 32 * k], ss += v[k] * v[k];
    if constexpr (NORM) {
        const float scale = rsqrtf(wave_sum(ss) / (NV * 32) + op.f[0]);
#pragma unroll
        for (int k = 0; k < NV; k++) v[k] *= scale * w[lane + 32 * k];
    }
    if constexpr (ROPE) rope_pair_yarn(v[0], v[1], lane, pos, op.f[1]);
}

// 32 x NV values of a wave -> int8 with a scale per 32 (group k = v[k] of all lanes), as wave_q8_256
template <int NV>
__device__ __forceinline__ void prep_q8(const float (&v)[NV], int8_t* q, float* scale) {
    const int lane = threadIdx.x % WAVE;
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const float m = wave_max(fabsf(v[k]));
        const float sc = m > 0 ? m / 127.0f : 1.0f;
        q[lane + 32 * k] = (int8_t)__float2int_rn(v[k] / sc);
        if (lane == k) scale[k] = sc;
    }
}

// op: p[0] attn_q output per token (nh * hd, or nh * 2 hd with gate), p[1] q out [nb][nh][hd], p[2] gate out, p[3] q
// norm weights, p[4] k [nb][nkv][hd], p[5] k norm weights, p[11] int8 cache, p[12] v [nb][nkv][hd]; i[0] nh, i[1] nkv,
// i[6] cache row bytes; f[0] eps, f[1] RoPE base. A cache split by position (kv_place.h): p[13] the pinned part, i[7]
// the cells before it (p[11]). One wave per q head, K head and V head of every token.
template <FattnCfg Cfg>
__device__ __forceinline__ void fattn_prep_q(const MegaOp& op, const Ctx& c, int b, int u, int64_t pos) {
    constexpr int HD = Cfg.hd, NV = HD / 32;
    const int nh = op.i[0];
    const int64_t stride = (Cfg.gate ? 2 : 1) * HD;
    const float* qf = (const float*)op.p[0] + ((int64_t)b * nh + u) * stride;
    float v[NV];
    prep_vec<NV, Cfg.qk_norm, Cfg.rope>(qf, (const float*)op.p[3], pos, op, v);
    float* qn = (float*)op.p[1] + ((int64_t)b * nh + u) * HD;
#pragma unroll
    for (int k = 0; k < NV; k++) qn[c.lane + 32 * k] = v[k];
    if constexpr (Cfg.gate) {
        float* g = (float*)op.p[2] + ((int64_t)b * nh + u) * HD;
        for (int i = c.lane; i < HD; i += WAVE) g[i] = qf[HD + i];
    }
}

template <FattnCfg Cfg>
__device__ __forceinline__ void fattn_prep_kv(const MegaOp& op, int b, int u, int64_t pos) {
    constexpr int HD = Cfg.hd, NV = HD / 32, NS = HD / 32;
    const int nh = op.i[0], nkv = op.i[1];
    const bool is_v = u >= nh + nkv;
    const int h = u - nh - (is_v ? nkv : 0);
    const float* src = (const float*)(is_v ? op.p[12] : op.p[4]) + ((int64_t)b * nkv + h) * HD;
    float v[NV];
    if (is_v)
        prep_vec<NV, false, false>(src, nullptr, pos, op, v);
    else
        prep_vec<NV, Cfg.qk_norm, Cfg.rope>(src, (const float*)op.p[5], pos, op, v);
    const bool cold = op.p[13] && pos >= op.i[7];
    uint8_t* r = cold ? (uint8_t*)op.p[13] + (pos - op.i[7]) * op.i[6] : (uint8_t*)op.p[11] + pos * op.i[6];
    prep_q8<NV>(v, (int8_t*)r + (int64_t)h * 2 * HD + (is_v ? HD : 0),
                (float*)(r + (int64_t)nkv * 2 * HD) + h * 2 * NS + (is_v ? NS : 0));
}

template <FattnCfg Cfg>
__device__ __noinline__ void op_fattn_prep(const MegaOp& op, const Ctx& c) {
    const int nh = op.i[0], per = nh + 2 * op.i[1];
    for (int64_t u0 = c.gw; u0 < (int64_t)c.nb * per; u0 += c.GW) {
        const int b = (int)(u0 / per), u = (int)(u0 % per);
        if (u < nh)
            fattn_prep_q<Cfg>(op, c, b, u, c.pos + b);
        else
            fattn_prep_kv<Cfg>(op, b, u, c.pos + b);
    }
}

// Partial result of one task: q rows of the KV group, one chunk. part: hd + 2 floats per (token, head, chunk): the
// unnormalized sum of p V, then the maximum and the sum of p.
struct FattnTask {
    int b, kvh, nh, nkv, s;
};

template <int HD>
__device__ __forceinline__ void fattn_store(const FattnTask t, const fa::Lds<HD> l, const fa::float8& acc, float* part,
                                            int lane) {
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8, g = t.nh / t.nkv;
    if (w < fa::Geo<HD>::OUT_WAVES) {
#pragma unroll
        for (int e = 0; e < 8; e++)
            if (kh + e < g)
                part[(((int64_t)t.b * t.nh + t.kvh * g + kh + e) * FA_SPL + t.s) * (HD + 2) + w * 16 + r16] = acc[e];
    }
    if ((int)threadIdx.x < g) {
        float* dst = part + (((int64_t)t.b * t.nh + t.kvh * g + threadIdx.x) * FA_SPL + t.s) * (HD + 2);
        dst[HD] = l.mrow[threadIdx.x], dst[HD + 1] = l.lrow[threadIdx.x];
    }
}

// op: p[0] q [nb][nh][hd], p[1] int8 cache, p[2] partials, p[3] its pinned part from the cell i[5] on (null: not
// split); i[0] nh, i[1] nkv, i[4] cache row bytes; f[0] scale. The group of q heads of a KV head (up to 16) fills the
// rows of the tile: K and V are read once per group.
template <FattnCfg Cfg>
__device__ __noinline__ void op_fattn_split(const MegaOp& op, const Ctx& c) {
    constexpr int HD = Cfg.hd;
    const int nh = op.i[0], nkv = op.i[1], g = nh / nkv;
    const fa::Lds<HD> l = fa::lds_of<HD>(c.sm);
    const int lane = c.lane, r16 = lane & 15;
    for (int it = blockIdx.x; it < c.nb * nkv * FA_SPL; it += gridDim.x) {
        const FattnTask t{it / FA_SPL / nkv, it / FA_SPL % nkv, nh, nkv, it % FA_SPL};
        const int64_t n_cells = c.pos + t.b + 1;
        const int ch = fa_chunk(n_cells);
        const int64_t cs = (int64_t)t.s * ch;
        if (cs >= n_cells) continue;
        const int64_t ce = std::min<int64_t>(n_cells, cs + ch);
        fa::half8 qa[fa::Geo<HD>::STEPS] = {};
        const float* qr = r16 < g ? (const float*)op.p[0] + ((int64_t)t.b * nh + t.kvh * g + r16) * HD : nullptr;
        fa::load_q<HD>(qr, op.f[0], qa, lane);
        lds_sync();  // the LDS of the previous task is free
        fa::reset_rows(l.mrow, l.lrow);
        fa::float8 acc = {};
        const fa::KvHead kvh{(const uint8_t*)op.p[1],
                             op.i[4],
                             nkv,
                             t.kvh,
                             (const uint8_t*)op.p[3],
                             op.p[3] ? (int64_t)op.i[5] : INT64_MAX};
        const fa::Tile<HD> tile{kvh, l, lane};
        for (int64_t c0 = cs; c0 < ce; c0 += fa::CC) {
            const int cn = (int)std::min<int64_t>(fa::CC, ce - c0);
            fa::step<HD>(tile, qa, acc, c0, cn, cn);
        }
        fattn_store<HD>(t, l, acc, (float*)op.p[2], lane);
    }
}

// One wave per (token, head, 32 values): the chunks weighted by e^(m_s - M), divided by the total sum, optionally *
// sigma(gate), int8. op: p[2] partials, p[4] gate [nb][nh][hd], p[5] o float, p[6] int8, p[7] meta; i[0] nh, i[3]
// values per token in o.
template <FattnCfg Cfg>
__device__ __noinline__ void op_fattn_merge(const MegaOp& op, const Ctx& c) {
    constexpr int HD = Cfg.hd, NS = HD / 32;
    const float* part = (const float*)op.p[2];
    const int nh = op.i[0];
    const int64_t os = op.i[3];
    for (int64_t g = c.gw; g < (int64_t)c.nb * nh * NS; g += c.GW) {
        const int k = (int)(g % NS), h = (int)(g / NS % nh), b = (int)(g / NS / nh);
        const int64_t n_cells = c.pos + b + 1;
        const int ns = (int)((n_cells + fa_chunk(n_cells) - 1) / fa_chunk(n_cells));
        const float* src = part + ((int64_t)b * nh + h) * FA_SPL * (HD + 2);
        float M = -INFINITY;
        for (int s = 0; s < ns; s++) M = fmaxf(M, src[s * (HD + 2) + HD]);
        const int e = c.lane + 32 * k;
        float L = 0, a = 0;
        for (int s = 0; s < ns; s++) {
            const float w = __expf(src[s * (HD + 2) + HD] - M);
            L += w * src[s * (HD + 2) + HD + 1];
            a += w * src[s * (HD + 2) + e];
        }
        float ov = a / L;
        if constexpr (Cfg.gate) ov *= sigm(((const float*)op.p[4])[((int64_t)b * nh + h) * HD + e]);
        const int64_t i = b * os + (int64_t)h * HD + e;
        ((float*)op.p[5])[i] = ov;
        wave_q8(ov, i, (int8_t*)op.p[6], (Q8Meta*)op.p[7]);
    }
}

// Runs the step if it is a full-attention step; false for the other groups.
template <FattnCfg Cfg>
__device__ __forceinline__ bool dispatch_fattn(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_FATTN_PREP:
            op_fattn_prep<Cfg>(op, c);
            return true;
        case OP_FATTN_SPLIT:
            op_fattn_split<Cfg>(op, c);
            return true;
        case OP_FATTN_MERGE:
            op_fattn_merge<Cfg>(op, c);
            return true;
        default:
            return false;
    }
}

static_assert(fa::THREADS == THREADS, "the tile needs the 16 waves of the megakernel");
static_assert(fa::Geo<256>::BYTES <= SM_FLOATS * 4 && fa::Geo<128>::BYTES <= SM_FLOATS * 4,
              "the tile must fit in sm, which must not grow");

}  // namespace
}  // namespace tona::gpu
