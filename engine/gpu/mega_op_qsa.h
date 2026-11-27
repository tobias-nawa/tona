// Step functions of the QSA layers: prepare (norm, RoPE, KV cache, indexer pool), attention of one block (the block
// score is in mega_op_idx.h, the top-k in mega_op_topk.h). Part of the kernel translation unit (arch/<name>/kernel.hip,
// no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"
#include "gpu/mega_op_idx.h"
#include "gpu/mega_op_topk.h"

namespace tona::gpu {
namespace {

// KV cache int8: one row of KV_ROW bytes per position: per KV head K (hd int8) and V (hd int8), behind them the
// scales (float) per 32 values: 8 for K and 8 for V per head. A block of 4 positions lies contiguous (for the RAM).

// Buffers and sizes of op_qsa_prep. qfull: 2*nh*hd per token; qn, g, tk, tv: per token; idx_q: ih*idim per token;
// trk: raw indexer key per token; kv: int8 KV cache by position (row bytes per position); ring: raw indexer keys
// (RAW_RING); pool: by block.
struct QsaPrep {
    const float *qfull, *q_norm, *k_norm, *trk, *idx_k_norm, *idx_q_norm, *tv;
    float *qn, *g, *tk, *pool, *idx_q, *ring;
    uint8_t* kv;
    int nh, nkv, hd, idim, ih, blk;
    int64_t row;
    float eps, base;
};

__device__ __forceinline__ QsaPrep qsa_prep_args(const MegaOp& op) {
    QsaPrep a;
    a.qfull = (const float*)op.p[0];
    a.qn = (float*)op.p[1];
    a.g = (float*)op.p[2];
    a.q_norm = (const float*)op.p[3];
    a.tk = (float*)op.p[4];
    a.k_norm = (const float*)op.p[5];
    a.trk = (const float*)op.p[6];
    a.pool = (float*)op.p[7];
    a.idx_k_norm = (const float*)op.p[8];
    a.idx_q = (float*)op.p[9];
    a.idx_q_norm = (const float*)op.p[10];
    a.kv = (uint8_t*)op.p[11];
    a.tv = (const float*)op.p[12];
    a.ring = (float*)op.p[13];
    a.nh = op.i[0], a.nkv = op.i[1], a.hd = op.i[2], a.idim = op.i[3], a.ih = op.i[4], a.blk = op.i[5];
    a.row = op.i[6];
    a.eps = op.f[0], a.base = op.f[1];
    return a;
}

// Unit u of token b, positions pos: q head u (norm + RoPE) and its gate
__device__ __forceinline__ void qsa_prep_q(const QsaPrep a, const Ctx c, int b, int64_t u, int64_t pos) {
    const float* qf = a.qfull + (int64_t)b * 2 * a.nh * a.hd;
    wave_norm_rope<8>(qf + u * 2 * a.hd, a.qn + (int64_t)b * a.nh * a.hd + u * a.hd, a.q_norm, pos, a.base, a.eps);
    for (int i = c.lane; i < a.hd; i += WAVE)
        a.g[(int64_t)b * a.nh * a.hd + u * a.hd + i] = qf[u * 2 * a.hd + a.hd + i];
}

// K (norm + RoPE) or V of one head -> int8 into the row of the position
__device__ __forceinline__ void qsa_prep_kv(const QsaPrep a, const Ctx c, int b, int64_t u, int64_t pos) {
    const bool is_v = u >= a.nh + a.nkv;
    const int h = (int)(u - a.nh - (is_v ? a.nkv : 0));
    float* src = (is_v ? (float*)a.tv : a.tk) + ((int64_t)b * a.nkv + h) * a.hd;
    if (!is_v) wave_norm_rope<8>(src, src, a.k_norm, pos, a.base, a.eps);
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = src[c.lane + 32 * k];
    uint8_t* r = a.kv + pos * a.row;
    wave_q8_256(v, (int8_t*)r + h * 2 * a.hd + (is_v ? a.hd : 0),
                (float*)(r + (int64_t)a.nkv * 2 * a.hd) + h * 16 + (is_v ? 8 : 0), c.lane);
}

// Stores the raw indexer key of the token in the ring; at the end of a block the pool entry of that block.
__device__ __forceinline__ void qsa_prep_pool(const QsaPrep a, const Ctx c, int b, int64_t pos) {
    const int idim = a.idim, blk = a.blk;
    float* dst_ring = a.ring + (pos % RAW_RING) * idim;
    for (int i = c.lane; i < idim; i += WAVE) dst_ring[i] = a.trk[(int64_t)b * idim + i];
    if ((pos + 1) % blk != 0) return;
    // raw key at position p: still in the per-token buffer if from this pass, otherwise in the ring
    auto raw_at = [&](int64_t p) { return p >= c.pos ? a.trk + (p - c.pos) * idim : a.ring + (p % RAW_RING) * idim; };
    // mean of blk raw keys from `first` -> RMSNorm ⊙ w -> RoPE at first
    const int64_t first = pos + 1 - blk;
    float* dst = a.pool + (pos / blk) * idim;
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float m = 0;
        for (int r = 0; r < blk; r++) m += raw_at(first + r)[c.lane + 32 * k];
        v[k] = m * (1.0f / blk);
        ss += v[k] * v[k];
    }
    const float scale = rsqrtf(wave_sum(ss) / idim + a.eps);
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] *= scale * a.idx_k_norm[c.lane + 32 * k];
    rope_pair_yarn(v[0], v[1], c.lane, first, a.base);
#pragma unroll
    for (int k = 0; k < 4; k++) dst[c.lane + 32 * k] = v[k];
}

__device__ __noinline__ void op_qsa_prep(const MegaOp& op, const Ctx& c) {
    const QsaPrep a = qsa_prep_args(op);
    const int64_t per = a.nh + 2 * a.nkv + 1 + a.ih;
    for (int64_t u0 = c.gw; u0 < c.nb * per; u0 += c.GW) {
        const int b = (int)(u0 / per);
        const int64_t u = u0 % per, pos = c.pos + b;
        if (u < a.nh) {
            qsa_prep_q(a, c, b, u, pos);
        } else if (u < a.nh + 2 * a.nkv) {
            qsa_prep_kv(a, c, b, u, pos);
        } else if (u == a.nh + 2 * a.nkv) {
            qsa_prep_pool(a, c, b, pos);
        } else if (qsa_select(op, pos)) {
            const int64_t h = u - a.nh - 2 * a.nkv - 1;
            float* iq = a.idx_q + (int64_t)b * a.ih * a.idim + h * a.idim;
            wave_norm_rope<4>(iq, iq, a.idx_q_norm, pos, a.base, a.eps);
        }
    }
}

// The row of one (token, head) task of op_attn: the cells it attends to, its q in LDS and the score row sc
struct AttnRow {
    const uint8_t* kv;     // int8 KV cache, row bytes per position
    const int32_t* cells;  // null: the positions 0 .. n_cells
    float *qh, *sc;
    int n_cells, kvh, nkv, hd;
    int64_t row;
};

// Scores of the cells: wave per cell, q·k scaled
__device__ __forceinline__ void attn_scores(const AttnRow a, const MegaOp& op, const Ctx& c) {
    const int wv = threadIdx.x / WAVE;
    for (int ci = wv; ci < a.n_cells; ci += MWAVES) {
        const int64_t cell = a.cells ? a.cells[ci] : ci;
        const uint8_t* r = a.kv + cell * a.row;
        const int8_t* kc = (const int8_t*)r + a.kvh * 2 * a.hd;
        const float* ks = (const float*)(r + (int64_t)a.nkv * 2 * a.hd) + a.kvh * 16;
        float dot = 0;
        for (int k = 0; k < a.hd / WAVE; k++) dot += a.qh[c.lane + WAVE * k] * (float)kc[c.lane + WAVE * k] * ks[k];
        dot = wave_sum(dot);
        if (c.lane == 0) a.sc[ci] = dot * op.f[0];
    }
}

// Softmax numerators e^(score − max) in place; returns their sum
__device__ __forceinline__ float attn_softmax(const AttnRow a, float* red) {
    float mx = -INFINITY;
    for (int ci = threadIdx.x; ci < a.n_cells; ci += THREADS) mx = fmaxf(mx, a.sc[ci]);
    mx = block_max(mx, red);
    float sum = 0;
    for (int ci = threadIdx.x; ci < a.n_cells; ci += THREADS) {
        const float e = __expf(a.sc[ci] - mx);
        a.sc[ci] = e;
        sum += e;
    }
    return block_sum(sum, red);
}

// Where op_attn writes: o (float) and q8 (int8) with i[3] values per token
struct AttnOut {
    const float* gate;
    float* o;
    int8_t* q8;
    Q8Meta* meta;
    int64_t os;
};

// Thread i < hd: its output dimension = Σ p·V / sum · σ(gate), whole waves because hd = 256
__device__ __forceinline__ void attn_output(const AttnRow a, const AttnOut out, float sum, int b, int h, int nh) {
    const int i = threadIdx.x;
    if (i >= a.hd) return;
    float acc = 0;
    for (int ci = 0; ci < a.n_cells; ci++) {
        const int64_t cell = a.cells ? a.cells[ci] : ci;
        const uint8_t* r = a.kv + cell * a.row;
        acc += a.sc[ci] * (float)((const int8_t*)r)[a.kvh * 2 * a.hd + a.hd + i] *
               ((const float*)(r + (int64_t)a.nkv * 2 * a.hd))[a.kvh * 16 + 8 + i / 32];
    }
    const float ov = acc / sum * sigm(out.gate[((int64_t)b * nh + h) * a.hd + i]);
    out.o[b * out.os + (int64_t)h * a.hd + i] = ov;
    wave_q8(ov, b * out.os + (int64_t)h * a.hd + i, out.q8, out.meta);
}

// One block per (token, head). q, gate: nh*hd per token; o, q8: i[3] per token; cells: MAX_CELLS per token
__device__ __noinline__ void op_attn(const MegaOp& op, const Ctx& c) {
    const float* q = (const float*)op.p[0];
    const uint8_t* kv = (const uint8_t*)op.p[1];  // int8 KV cache, i[4] bytes per position
    const int32_t* cells_buf = (const int32_t*)op.p[3];
    const AttnOut out{(const float*)op.p[4], (float*)op.p[5], (int8_t*)op.p[6], (Q8Meta*)op.p[7], op.i[3]};
    const int nh = op.i[0], nkv = op.i[1], hd = op.i[2];
    const int64_t row = op.i[4];
    float* red = c.sm;
    float* qh = c.sm + 64;
    float* sc = qh + 256;
    for (int hb = blockIdx.x; hb < nh * c.nb; hb += gridDim.x) {
        const int b = hb / nh, h = hb % nh;
        const int64_t pos = c.pos + b;
        const bool sel = qsa_select(op, pos);
        const int32_t* cells = sel ? cells_buf + (int64_t)b * MAX_CELLS : nullptr;
        const int n_cells = sel ? c.P->n_cells[c.r0 + b] : (int)(pos + 1);
        const AttnRow a{kv, cells, qh, sc, n_cells, h / (nh / nkv), nkv, hd, row};
        lds_sync();
        for (int i = threadIdx.x; i < hd; i += THREADS) qh[i] = q[((int64_t)b * nh + h) * hd + i];
        lds_sync();
        attn_scores(a, op, c);
        lds_sync();
        const float sum = attn_softmax(a, red);
        lds_sync();
        attn_output(a, out, sum, b, h, nh);
    }
}

}  // namespace
}  // namespace tona::gpu
