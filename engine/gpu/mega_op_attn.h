// QSA attention in split-K form: partial results per chunk (WMMA or wave per chunk) and their merge.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"
#include "gpu/mega_op_qsa.h"

namespace tona::gpu {
namespace {

// Split-K of the QSA attention (flash decoding): one wave per (token, head, chunk of ATT_CH cells), lane = 8
// dimensions. After 13K of context every token has ~2,051 cells; op_attn went through them one after the other in one
// block per (token, head) (96 tasks per layer). Partial result per chunk: unnormalized sum Σ p·V, max and Σ p.
static_assert(MWAVES * ATT_CH <= MAX_CELLS, "the score rows of a wave must fit into the sc area");

// Cells per chunk: small enough for all chunks together to give about 2 tasks per wave (short context: many small
// chunks instead of one wave per head), at most ATT_CH, at most ATT_SPL chunks. nb is passed as PNB, not as the width
// of the pass: the chunks of a token (and so its rounding) must not depend on how many tokens the pass has
// (bit-identical for B = 1 ... 4).
__device__ __forceinline__ int att_chunk(int n_cells, int nb, int nh, int64_t GW) {
    const int want = (int)std::min<int64_t>(ATT_SPL, std::max<int64_t>(1, 2 * GW / ((int64_t)nb * nh)));
    const int ch = (n_cells + want - 1) / want;
    return std::min(ATT_CH, std::max(16, (ch + 3) & ~3));
}

__device__ __forceinline__ void attn_cells(const MegaOp& op, const Ctx& c, int b, const int32_t*& cells, int& n_cells) {
    const int64_t pos = c.pos + b;
    const bool sel = qsa_select(op, pos);
    cells = sel ? (const int32_t*)op.p[3] + (int64_t)b * MAX_CELLS : nullptr;
    n_cells = sel ? c.P->n_cells[c.r0 + b] : (int)(pos + 1);
}

// Block variant (op_attn_wmma): chunks per (token, KV head), about 2 tasks per block, multiples of aw::CC
__device__ __forceinline__ int att_chunk_blk(int n_cells, int nb, int nkv, int nblk) {
    const int want = std::min(ATT_SPL, std::max(1, 2 * nblk / (nb * nkv)));
    const int ch = (n_cells + want - 1) / want;
    return std::max(aw::CC, (ch + aw::CC - 1) / aw::CC * aw::CC);
}

// After the pattern of the prefill kernel k_attn_wmma and of aiter PR #6670 (QSA decode, gfx950): one block computes
// all g = nh/nkv heads of a KV group together, K/V are read once and unpacked to f16. Per 16 cells: K and V^T into LDS,
// S = Q·K^T (waves 0-7, 32 dimensions each, Q straight from memory), online softmax (wave per row), O += P·V (wave per
// 16 dimensions). Partial results as op_attn_split, merged by op_attn_merge (i[6] = 1).

// Views into the LDS (layout see aw::BYTES)
struct AwLds {
    _Float16 *Ks, *Vt, *Ps;
    float *pt, *mrow, *lrow, *arow;
};

__device__ __forceinline__ AwLds aw_lds(float* sm) {
    using namespace aw;
    AwLds l;
    l.Ks = reinterpret_cast<_Float16*>(sm);
    l.pt = reinterpret_cast<float*>(l.Ks);  // partial scores [8][16][CC], after Ks has been read
    l.Vt = l.Ks + CC * LQ;
    l.Ps = l.Vt + HD * CC;
    l.mrow = reinterpret_cast<float*>(l.Ps + 16 * CC);
    l.lrow = l.mrow + 16;
    l.arow = l.lrow + 16;
    return l;
}

// One task: KV head kvh of token b, cells [cs, ce) of its cell list (null: the cells 0 .. n_cells)
struct AwTask {
    const float* q;
    const uint8_t* kv;
    const int32_t* cells;
    int b, kvh, nh, nkv, cs, ce;
    int64_t row;
    float scale;
};

// A parts of Q (waves 0-7): row r16 = head of the group, dimensions w*32 + 16*k2 + kh ... + 7, with the scale
__device__ __forceinline__ void aw_load_q(const AwTask t, aw::half8 (&qa)[2], int lane) {
    using namespace aw;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int g = t.nh / t.nkv;
    if (w < 8 && r16 < g) {
        const float* qr = t.q + ((int64_t)t.b * t.nh + t.kvh * g + r16) * HD + w * 32 + kh;
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++)
#pragma unroll
            for (int j = 0; j < 8; j++) qa[k2][j] = (_Float16)(qr[16 * k2 + j] * t.scale);
    }
}

// K and V^T of the cells [c0, c0 + cn) into LDS, one thread per (cell, 8 dimensions), dequantized to f16
__device__ __forceinline__ void aw_stage(const AwTask t, const AwLds l, int c0, int cn) {
    using namespace aw;
    const int tid = threadIdx.x, cc = tid & 15, gq = tid >> 4;
    half8 kh8 = {}, vh8 = {};
    if (cc < cn) {
        const int64_t cell = t.cells ? t.cells[c0 + cc] : c0 + cc;
        const uint8_t* r = t.kv + cell * t.row;
        const int8_t* kc = (const int8_t*)r + t.kvh * 2 * HD + gq * 8;
        const float* sc = (const float*)(r + (int64_t)t.nkv * 2 * HD) + t.kvh * 16;
        const float ks = sc[gq / 4], vs = sc[8 + gq / 4];
        const uint2 k8 = *reinterpret_cast<const uint2*>(kc);
        const uint2 v8 = *reinterpret_cast<const uint2*>(kc + HD);
        const int8_t* kb = (const int8_t*)&k8;
        const int8_t* vb = (const int8_t*)&v8;
#pragma unroll
        for (int j = 0; j < 8; j++) kh8[j] = (_Float16)((float)kb[j] * ks), vh8[j] = (_Float16)((float)vb[j] * vs);
    }
    *reinterpret_cast<half8*>(&l.Ks[cc * LQ + gq * 8]) = kh8;
#pragma unroll
    for (int j = 0; j < 8; j++) l.Vt[(gq * 8 + j) * CC + cc] = vh8[j];
}

// Partial scores S = Q·K^T per wave 0-7 (32 dimensions each) into pt; the barrier in between lets pt reuse Ks
__device__ __forceinline__ void aw_scores(const aw::half8 (&qa)[2], const AwLds l, int lane) {
    using namespace aw;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    float8 sv = {};
    if (w < 8) {
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const half8 bb = *reinterpret_cast<const half8*>(&l.Ks[r16 * LQ + w * 32 + 16 * k2 + kh]);
            sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[k2], bb, sv);
        }
    }
    lds_sync();  // Ks read, pt overwrites it
    if (w < 8) {
#pragma unroll
        for (int e = 0; e < 8; e++) l.pt[(w * 16 + kh + e) * CC + r16] = sv[e];
    }
}

// Online softmax, wave w: row w, lane < 16 = cell
__device__ __forceinline__ void aw_softmax(const AwLds l, int cn, int lane) {
    using namespace aw;
    const int w = threadIdx.x / WAVE;
    float v = -INFINITY;
    if (lane < cn) {
        v = 0;
#pragma unroll
        for (int p8 = 0; p8 < 8; p8++) v += l.pt[(p8 * 16 + w) * CC + lane];
    }
    const float mo = l.mrow[w], mx = fmaxf(mo, wave_max(v));
    const float e = v > -INFINITY ? __expf(v - mx) : 0.0f;
    if (lane < CC) l.Ps[w * CC + lane] = (_Float16)e;
    const float sum = wave_sum(e);
    if (lane == 0) {
        const float al = mo > -INFINITY ? __expf(mo - mx) : 0.0f;
        l.mrow[w] = mx, l.lrow[w] = l.lrow[w] * al + sum, l.arow[w] = al;
    }
}

// O += P·V for the dimensions w*16 ... of this wave, after rescaling by the softmax factors
__device__ __forceinline__ void aw_accumulate(aw::float8& acc, const AwLds l, int lane) {
    using namespace aw;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] *= l.arow[kh + e];
    const half8 a = *reinterpret_cast<const half8*>(&l.Ps[r16 * CC + kh]);
    const half8 bb = *reinterpret_cast<const half8*>(&l.Vt[(w * 16 + r16) * CC + kh]);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc);
}

// acc[e]: head kh + e of the group, dimension w*16 + r16; max and sum go behind the HD values
__device__ __forceinline__ void aw_store(const AwTask t, const AwLds l, const aw::float8 acc, int s, float* part,
                                         int lane) {
    using namespace aw;
    const int tid = threadIdx.x, w = tid / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int g = t.nh / t.nkv;
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int r = kh + e;
        if (r < g) part[(((int64_t)t.b * t.nh + t.kvh * g + r) * ATT_SPL + s) * (HD + 2) + w * 16 + r16] = acc[e];
    }
    if (tid < g) {
        float* dst = part + (((int64_t)t.b * t.nh + t.kvh * g + tid) * ATT_SPL + s) * (HD + 2);
        dst[HD] = l.mrow[tid], dst[HD + 1] = l.lrow[tid];
    }
}

__device__ __noinline__ void op_attn_wmma(const MegaOp& op, const Ctx& c) {
    using namespace aw;
    float* part = (float*)op.p[2];
    const int nh = op.i[0], nkv = op.i[1];
    const AwLds l = aw_lds(c.sm);
    const int lane = c.lane;
    const AwTask shared{(const float*)op.p[0], (const uint8_t*)op.p[1], nullptr, 0, 0, nh, nkv, 0, 0, op.i[4], op.f[0]};
    for (int it = blockIdx.x; it < c.nb * nkv * ATT_SPL; it += gridDim.x) {
        const int s = it % ATT_SPL, kvh = it / ATT_SPL % nkv, b = it / ATT_SPL / nkv;
        AwTask t = shared;
        t.b = b, t.kvh = kvh;
        int n_cells;
        attn_cells(op, c, b, t.cells, n_cells);
        const int ch = att_chunk_blk(n_cells, PNB, nkv, gridDim.x);
        t.cs = s * ch;
        if (t.cs >= n_cells) continue;
        t.ce = min(n_cells, t.cs + ch);
        half8 qa[2] = {};
        aw_load_q(t, qa, lane);
        lds_sync();  // LDS of the previous task is free
        if (threadIdx.x < 16) l.mrow[threadIdx.x] = -INFINITY, l.lrow[threadIdx.x] = 0;
        float8 acc = {};
        for (int c0 = t.cs; c0 < t.ce; c0 += CC) {
            const int cn = min(CC, t.ce - c0);
            lds_sync();
            aw_stage(t, l, c0, cn);
            lds_sync();
            aw_scores(qa, l, lane);
            lds_sync();
            aw_softmax(l, cn, lane);
            lds_sync();
            aw_accumulate(acc, l, lane);
        }
        aw_store(t, l, acc, s, part, lane);
    }
}

// The chunk of cells one wave works on in op_attn_split: cells [c0, c0 + n) of the list of its (token, head)
struct SplitChunk {
    const uint8_t* kv;
    const int32_t* cells;  // null: the positions themselves
    float* sw;             // scores of this wave
    int64_t row;
    int c0, n, kvh, soff, l;
};

// Scores of the chunk into sw, returns their maximum. Lane l holds 8 dimensions of q (qa, qb), 4 cells at a time: all
// loads first, then the sums (the compiler does not unroll loops with shuffles).
__device__ __forceinline__ float split_scores(const SplitChunk t, const float4 qa, const float4 qb, const MegaOp& op) {
    constexpr int hd = 256;
    const int koff = t.kvh * 2 * hd + 8 * t.l;
    float mx = -INFINITY;
    for (int c4 = 0; c4 < t.n; c4 += 4) {
        uint2 kk[4];
        float ks[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int ci = min(c4 + j, t.n - 1);
            const uint8_t* r = t.kv + (t.cells ? (int64_t)t.cells[t.c0 + ci] : (int64_t)(t.c0 + ci)) * t.row;
            kk[j] = *(const uint2*)(r + koff);
            ks[j] = ((const float*)(r + t.soff))[t.kvh * 16 + t.l / 4];
        }
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int8_t* k8 = (const int8_t*)&kk[j];
            const float d = qa.x * k8[0] + qa.y * k8[1] + qa.z * k8[2] + qa.w * k8[3] + qb.x * k8[4] + qb.y * k8[5] +
                            qb.z * k8[6] + qb.w * k8[7];
            const float sc = wave_sum(d * ks[j]) * op.f[0];
            if (c4 + j < t.n) {
                if (t.l == 0) t.sw[c4 + j] = sc;
                mx = fmaxf(mx, sc);
            }
        }
    }
    return mx;
}

// acc[8] += Σ p·V of the chunk (p = e^(score − mx)), sum += Σ p
__device__ __forceinline__ void split_values(const SplitChunk t, float mx, float (&acc)[8], float& sum) {
    constexpr int hd = 256;
    const int voff = t.kvh * 2 * hd + hd + 8 * t.l;
#pragma unroll 4
    for (int ci = 0; ci < t.n; ci++) {
        const int64_t cell = t.cells ? t.cells[t.c0 + ci] : t.c0 + ci;
        const uint8_t* r = t.kv + cell * t.row;
        const uint2 vv = *(const uint2*)(r + voff);
        const float p = __expf(t.sw[ci] - mx);
        const float w = p * ((const float*)(r + t.soff))[t.kvh * 16 + 8 + t.l / 4];
        const int8_t* v8 = (const int8_t*)&vv;
#pragma unroll
        for (int k = 0; k < 8; k++) acc[k] += w * v8[k];
        sum += p;
    }
}

// p[2]: partial results, hd + 2 floats per (token, head, chunk)
__device__ __noinline__ void op_attn_split(const MegaOp& op, const Ctx& c) {
    const float* q = (const float*)op.p[0];
    const uint8_t* kv = (const uint8_t*)op.p[1];
    float* part = (float*)op.p[2];
    const int nh = op.i[0], nkv = op.i[1];
    constexpr int hd = 256;
    const int64_t row = op.i[4];
    float* sw = c.sm + 64 + 256 + (threadIdx.x / WAVE) * ATT_CH;  // scores of this wave
    const int l = c.lane;
    for (int64_t u = c.gw; u < (int64_t)c.nb * nh * ATT_SPL; u += c.GW) {
        const int s = (int)(u % ATT_SPL), h = (int)(u / ATT_SPL % nh), b = (int)(u / ATT_SPL / nh);
        const int32_t* cells;
        int n_cells;
        attn_cells(op, c, b, cells, n_cells);
        const int ch = att_chunk(n_cells, PNB, nh, c.GW), c0 = s * ch;
        if (c0 >= n_cells) continue;
        const int n = min(ch, n_cells - c0), kvh = h / (nh / nkv);
        const float4* qp = (const float4*)(q + ((int64_t)b * nh + h) * hd + 8 * l);
        const float4 qa = qp[0], qb = qp[1];
        const SplitChunk t{kv, cells, sw, row, c0, n, kvh, nkv * 2 * hd, l};
        const float mx = split_scores(t, qa, qb, op);
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        float acc[8] = {}, sum = 0;
        split_values(t, mx, acc, sum);
        float* dst = part + (((int64_t)b * nh + h) * ATT_SPL + s) * (hd + 2);
        reinterpret_cast<float4*>(dst + 8 * l)[0] = make_float4(acc[0], acc[1], acc[2], acc[3]);
        reinterpret_cast<float4*>(dst + 8 * l)[1] = make_float4(acc[4], acc[5], acc[6], acc[7]);
        if (l == 0) dst[hd] = mx, dst[hd + 1] = sum;
        __builtin_amdgcn_wave_barrier();  // reuse sw only after all lanes have read it
    }
}

// One wave per (token, head): weight the chunks with e^(m_s − M), divide by the total sum, · σ(gate), int8
__device__ __noinline__ void op_attn_merge(const MegaOp& op, const Ctx& c) {
    const float* part = (const float*)op.p[2];
    const float* gate = (const float*)op.p[4];
    float* o = (float*)op.p[5];
    int8_t* q8 = (int8_t*)op.p[6];
    Q8Meta* meta = (Q8Meta*)op.p[7];
    const int nh = op.i[0];
    constexpr int hd = 256;
    const int64_t os = op.i[3];
    // Unit = (token, head, 32 values): with 8 x 32 values per wave only 6 of 32 blocks were busy
    for (int64_t g = c.gw; g < (int64_t)c.nb * nh * 8; g += c.GW) {
        const int k = (int)(g % 8);
        const int64_t u = g / 8;
        const int h = (int)(u % nh), b = (int)(u / nh);
        const int32_t* cells;
        int n_cells;
        attn_cells(op, c, b, cells, n_cells);
        const int ch = op.i[6] ? att_chunk_blk(n_cells, PNB, op.i[1], gridDim.x) : att_chunk(n_cells, PNB, nh, c.GW);
        const int ns = (n_cells + ch - 1) / ch;
        const float* src = part + ((int64_t)b * nh + h) * ATT_SPL * (hd + 2);
        float M = -INFINITY;
        for (int s = 0; s < ns; s++) M = fmaxf(M, src[s * (hd + 2) + hd]);
        float L = 0, w[ATT_SPL];
#pragma unroll
        for (int s = 0; s < ATT_SPL; s++) {
            w[s] = s < ns ? __expf(src[s * (hd + 2) + hd] - M) : 0.0f;
            L += s < ns ? w[s] * src[s * (hd + 2) + hd + 1] : 0.0f;
        }
        const int e = c.lane + 32 * k;
        float a = 0;
#pragma unroll
        for (int s = 0; s < ATT_SPL; s++)
            if (s < ns) a += w[s] * src[s * (hd + 2) + e];
        const float ov = a / L * sigm(gate[((int64_t)b * nh + h) * hd + e]);
        o[b * os + (int64_t)h * hd + e] = ov;
        wave_q8(ov, b * os + (int64_t)h * hd + e, q8, meta);
    }
}

// Runs the step if it is a QSA step; false for the other groups.
__device__ __forceinline__ bool dispatch_qsa(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_QSA_PREP:
            op_qsa_prep(op, c);
            return true;
        case OP_IDX_SCORE:
            op_idx_score(op, c);
            return true;
        case OP_TOPK:
            op_topk(op, c);
            return true;
        case OP_ATTN:
            op_attn(op, c);
            return true;
        case OP_ATTN_SPLIT:
            op_attn_split(op, c);
            return true;
        case OP_ATTN_MERGE:
            op_attn_merge(op, c);
            return true;
        case OP_ATTN_WMMA:
            op_attn_wmma(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
