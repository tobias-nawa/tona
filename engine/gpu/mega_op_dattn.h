// Drafter attention: windowed over ring and block, WMMA, split-K and merge.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_attn.h"
#include "gpu/mega_op_common.h"
#include "gpu/mega_op_draft.h"

namespace tona::gpu {
namespace {

// Cells a token of the drafter block attends to: the window of swa positions (0: the whole context) up to the end of
// the block. Returns their number, c0 receives the first position.
__device__ __forceinline__ int dattn_cells(const MegaOp& op, const Ctx& c, int b, int64_t& c0) {
    const int swa = op.i[3];
    const int64_t p = c.pos + b;
    c0 = swa > 0 ? std::max<int64_t>(0, p - swa + 1) : 0;
    return (int)(c.pos + c.nb - c0);
}

// Drafter attention with WMMA: a group of 4 heads x 4 tokens of the block = 16 rows, all see the same cells (context
// window and block); only the start of the window moves by one cell per token (masked). One block per (KV head, chunk
// of the union), K/V in f16 from the ring: per 16 cells K and V^T into LDS, S = Q·K^T (waves 0-3, 32 dimensions
// each), online softmax (wave per row), O += P·V (waves 0-7, 16 dimensions each). Partial results as
// op_dattn_split, merged by op_dattn_merge with i[5] = 1.
namespace dw {
constexpr int HD = 128, CC = 16, LQ = HD + 8;
constexpr int BYTES = CC * LQ * 2 + HD * CC * 2 + 16 * CC * 2 + 4 * 16 * CC * 4 + 48 * 4;
}  // namespace dw
static_assert(dw::BYTES <= SM_FLOATS * 4, "op_dattn_wmma must fit in sm");

// Views into the LDS (layout see dw::BYTES)
struct DwLds {
    _Float16 *Ks, *Vt, *Ps;
    float *pt, *mrow, *lrow, *arow;
};

__device__ __forceinline__ DwLds dw_lds(float* sm) {
    using namespace dw;
    DwLds l;
    l.Ks = reinterpret_cast<_Float16*>(sm);
    l.Vt = l.Ks + CC * LQ;
    l.Ps = l.Vt + HD * CC;
    l.pt = reinterpret_cast<float*>(l.Ps + 16 * CC);  // partial scores [4][16][CC]
    l.mrow = l.pt + 4 * 16 * CC;
    l.lrow = l.mrow + 16;
    l.arow = l.lrow + 16;
    return l;
}

// One task: KV head kvh, cells [cs, ce) of the union starting at position cu, K/V in a ring of `ring` positions
struct DwTask {
    const float* q;
    const _Float16 *K, *V;
    int nh, nkv, kvh, nb;
    int64_t cu, ring;
    float scale;
};

// A parts of Q (waves 0-3): row r16 = (token r16 / g, head r16 % g), dimensions w*32 + 16*k2 + kh ... + 7
__device__ __forceinline__ void dw_load_q(const DwTask t, aw::half8 (&qa)[2], int lane) {
    using namespace dw;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int g = t.nh / t.nkv;
    if (w < 4 && r16 / g < t.nb) {
        const float* qr = t.q + ((int64_t)(r16 / g) * t.nh + t.kvh * g + r16 % g) * HD + w * 32 + kh;
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++)
#pragma unroll
            for (int j = 0; j < 8; j++) qa[k2][j] = (_Float16)(qr[16 * k2 + j] * t.scale);
    }
}

// K and V^T of the cells [c0, c0 + cn) of the union into LDS, one thread per (cell, 8 dimensions)
__device__ __forceinline__ void dw_stage(const DwTask t, const DwLds l, int c0, int cn) {
    using namespace dw;
    using aw::half8;
    const int tid = threadIdx.x;
    if (tid < 16 * (HD / 8)) {
        const int cc = tid & 15, gq = tid >> 4;
        half8 kh8 = {}, vh8 = {};
        if (cc < cn) {
            const int64_t off = (((t.cu + c0 + cc) % t.ring) * t.nkv + t.kvh) * HD + gq * 8;
            kh8 = *reinterpret_cast<const half8*>(t.K + off);
            vh8 = *reinterpret_cast<const half8*>(t.V + off);
        }
        *reinterpret_cast<half8*>(&l.Ks[cc * LQ + gq * 8]) = kh8;
#pragma unroll
        for (int j = 0; j < 8; j++) l.Vt[(gq * 8 + j) * CC + cc] = vh8[j];
    }
}

// Partial scores S = Q·K^T per wave 0-3 (32 dimensions each) into pt
__device__ __forceinline__ void dw_scores(const aw::half8 (&qa)[2], const DwLds l, int lane) {
    using namespace dw;
    using aw::float8;
    using aw::half8;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    if (w < 4) {
        float8 sv = {};
#pragma unroll
        for (int k2 = 0; k2 < 2; k2++) {
            const half8 bb = *reinterpret_cast<const half8*>(&l.Ks[r16 * LQ + w * 32 + 16 * k2 + kh]);
            sv = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(qa[k2], bb, sv);
        }
#pragma unroll
        for (int e = 0; e < 8; e++) l.pt[(w * 16 + kh + e) * CC + r16] = sv[e];
    }
}

// Online softmax, wave w < 16: row w, lane < 16 = cell; cells before lo_w (start of the token's window) are masked
__device__ __forceinline__ void dw_softmax(const DwLds l, int cn, int c0, int lo_w, int lane) {
    using namespace dw;
    const int w = threadIdx.x / WAVE;
    if (w < 16) {
        float v = -INFINITY;
        if (lane < cn && c0 + lane >= lo_w)
            v = l.pt[(0 * 16 + w) * CC + lane] + l.pt[(1 * 16 + w) * CC + lane] + l.pt[(2 * 16 + w) * CC + lane] +
                l.pt[(3 * 16 + w) * CC + lane];
        const float mo = l.mrow[w], mx = fmaxf(mo, wave_max(v));
        const float e = v > -INFINITY ? __expf(v - mx) : 0.0f;
        if (lane < CC) l.Ps[w * CC + lane] = (_Float16)e;
        const float sum = wave_sum(e);
        if (lane == 0) {
            const float al = mo > -INFINITY ? __expf(mo - mx) : 0.0f;
            l.mrow[w] = mx, l.lrow[w] = l.lrow[w] * al + sum, l.arow[w] = al;
        }
    }
}

// O += P·V for the dimensions w*16 ... (waves 0-7), after rescaling by the softmax factors
__device__ __forceinline__ void dw_accumulate(aw::float8& acc, const DwLds l, int lane) {
    using namespace dw;
    using aw::half8;
    const int w = threadIdx.x / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    if (w < HD / 16) {
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] *= l.arow[kh + e];
        const half8 a = *reinterpret_cast<const half8*>(&l.Ps[r16 * CC + kh]);
        const half8 bb = *reinterpret_cast<const half8*>(&l.Vt[(w * 16 + r16) * CC + kh]);
        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, bb, acc);
    }
}

// acc[e]: row kh + e (token, head), dimension w*16 + r16; max and sum go behind the HD values
__device__ __forceinline__ void dw_store(const DwTask t, const DwLds l, const aw::float8 acc, int s, float* part,
                                         int lane) {
    using namespace dw;
    const int tid = threadIdx.x, w = tid / WAVE, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int g = t.nh / t.nkv;
    if (w < HD / 16) {
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int r = kh + e, b = r / g;
            if (b < t.nb)
                part[(((int64_t)b * t.nh + t.kvh * g + r % g) * ATT_SPL + s) * (HD + 2) + w * 16 + r16] = acc[e];
        }
    }
    if (tid < 16 && tid / g < t.nb) {
        float* dst = part + (((int64_t)(tid / g) * t.nh + t.kvh * g + tid % g) * ATT_SPL + s) * (HD + 2);
        dst[HD] = l.mrow[tid], dst[HD + 1] = l.lrow[tid];
    }
}

__device__ __noinline__ void op_dattn_wmma(const MegaOp& op, const Ctx& c) {
    using namespace dw;
    using aw::float8;
    using aw::half8;
    float* part = (float*)op.p[6];
    const int nh = op.i[0], nkv = op.i[1], g = nh / nkv;
    const DwLds l = dw_lds(c.sm);
    const int lane = c.lane;
    const int tid = threadIdx.x, w = tid / WAVE;
    DwTask t{(const float*)op.p[0],
             (const _Float16*)op.p[1],
             (const _Float16*)op.p[2],
             nh,
             nkv,
             0,
             c.nb,
             0,
             op.i[4],
             op.f[0]};
    const int n_u = dattn_cells(op, c, 0, t.cu);  // union: token 0 has the earliest start of the window
    const int ch = att_chunk_blk(n_u, 1, nkv, gridDim.x);
    // row w (softmax): token w / g, first valid cell of the union
    const int bw = w / g;
    int lo_w = INT32_MAX;
    if (w < 16 && bw < c.nb) {
        int64_t cb;
        dattn_cells(op, c, bw, cb);
        lo_w = (int)(cb - t.cu);
    }
    for (int it = blockIdx.x; it < nkv * ATT_SPL; it += gridDim.x) {
        const int s = it % ATT_SPL;
        t.kvh = it / ATT_SPL;
        const int cs = s * ch;
        if (cs >= n_u) continue;
        const int ce = min(n_u, cs + ch);
        half8 qa[2] = {};
        dw_load_q(t, qa, lane);
        lds_sync();  // LDS of the previous task is free
        if (tid < 16) l.mrow[tid] = -INFINITY, l.lrow[tid] = 0;
        float8 acc = {};
        for (int c0 = cs; c0 < ce; c0 += CC) {
            const int cn = min(CC, ce - c0);
            lds_sync();
            dw_stage(t, l, c0, cn);
            lds_sync();
            dw_scores(qa, l, lane);
            lds_sync();
            dw_softmax(l, cn, c0, lo_w, lane);
            lds_sync();
            dw_accumulate(acc, l, lane);
        }
        dw_store(t, l, acc, s, part, lane);
    }
}

// One block per (token, head): keys [max(0, p − window + 1), pos + nb), i.e. the context and the whole block.
// q, o: nh*128 per token
__device__ __noinline__ void op_dattn(const MegaOp& op, const Ctx& c) {
    const float* q = (const float*)op.p[0];
    const float* K = (const float*)op.p[1];
    const float* V = (const float*)op.p[2];
    float* o = (float*)op.p[3];
    int8_t* q8 = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int nh = op.i[0], nkv = op.i[1], hd = 128, swa = op.i[3];
    const int64_t ring = op.i[4];
    float* red = c.sm;
    float* qh = c.sm + 64;
    float* sc = qh + 256;
    float* part =
        sc + MAX_CELLS - 512;  // partial sums of the 4 groups; n_cells <= MAX_CELLS - 512 (checked at build time)
    const int wv = threadIdx.x / WAVE;
    for (int hb = blockIdx.x; hb < nh * c.nb; hb += gridDim.x) {
        const int b = hb / nh, h = hb % nh;
        const int64_t p = c.pos + b, c0 = swa > 0 ? std::max<int64_t>(0, p - swa + 1) : 0;
        const int n_cells = (int)(c.pos + c.nb - c0);
        const int kvh = h / (nh / nkv);
        lds_sync();
        for (int i = threadIdx.x; i < hd; i += THREADS) qh[i] = q[((int64_t)b * nh + h) * hd + i];
        lds_sync();
        for (int ci = wv; ci < n_cells; ci += MWAVES) {
            const float* kc = K + (((c0 + ci) % ring) * nkv + kvh) * hd;
            float dot = 0;
            for (int i = c.lane; i < hd; i += WAVE) dot += qh[i] * kc[i];
            dot = wave_sum(dot);
            if (c.lane == 0) sc[ci] = dot * op.f[0];
        }
        lds_sync();
        float mx = -INFINITY;
        for (int ci = threadIdx.x; ci < n_cells; ci += THREADS) mx = fmaxf(mx, sc[ci]);
        mx = block_max(mx, red);
        float sum = 0;
        for (int ci = threadIdx.x; ci < n_cells; ci += THREADS) {
            const float e = __expf(sc[ci] - mx);
            sc[ci] = e;
            sum += e;
        }
        sum = block_sum(sum, red);
        lds_sync();
        const int i = threadIdx.x % hd, grp = threadIdx.x / hd;  // 4 groups share the keys
        float acc = 0;
        for (int ci = grp; ci < n_cells; ci += THREADS / hd)
            acc += sc[ci] * V[(((c0 + ci) % ring) * nkv + kvh) * hd + i];
        part[grp * hd + i] = acc;
        lds_sync();
        if (threadIdx.x < hd) {
            float s = 0;
            for (int g = 0; g < THREADS / hd; g++) s += part[g * hd + i];
            const float ov = s / sum;
            o[((int64_t)b * nh + h) * hd + i] = ov;
            wave_q8(ov, ((int64_t)b * nh + h) * hd + i, q8, meta);
        }
    }
}

// Split-K of the drafter attention like op_attn_split: head dim 128 (lane = 4 dimensions), K/V as f16 in the ring,
// window i[3] (0: the whole context), all tokens of the block see the whole block. p[6]: partial results, hd + 2
__device__ __noinline__ void op_dattn_split(const MegaOp& op, const Ctx& c) {
    const float* q = (const float*)op.p[0];
    const _Float16* K = (const _Float16*)op.p[1];  // K/V as f16 (OP_DQK with i[4] = 1)
    const _Float16* V = (const _Float16*)op.p[2];
    float* part = (float*)op.p[6];
    const int nh = op.i[0], nkv = op.i[1];
    constexpr int hd = 128;
    const int64_t ring = op.i[4];
    float* sw = c.sm + 64 + 256 + (threadIdx.x / WAVE) * ATT_CH;
    const int l = c.lane;
    for (int64_t u = c.gw; u < (int64_t)c.nb * nh * ATT_SPL; u += c.GW) {
        const int s = (int)(u % ATT_SPL), h = (int)(u / ATT_SPL % nh), b = (int)(u / ATT_SPL / nh);
        int64_t cb;
        const int n_cells = dattn_cells(op, c, b, cb);
        const int ch = att_chunk(n_cells, PNB, nh, c.GW), c0 = s * ch;
        if (c0 >= n_cells) continue;
        const int n = min(ch, n_cells - c0), kvh = h / (nh / nkv);
        const float4 qv = reinterpret_cast<const float4*>(q + ((int64_t)b * nh + h) * hd)[l];
        float mx = -INFINITY;
        for (int c4 = 0; c4 < n; c4 += 4) {
            uint2 kk[4];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int ci = min(c4 + j, n - 1);
                kk[j] = reinterpret_cast<const uint2*>(K + (((cb + c0 + ci) % ring) * nkv + kvh) * hd)[l];
            }
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const _Float16* k4 = (const _Float16*)&kk[j];
                const float sc =
                    wave_sum(qv.x * (float)k4[0] + qv.y * (float)k4[1] + qv.z * (float)k4[2] + qv.w * (float)k4[3]) *
                    op.f[0];
                if (c4 + j < n) {
                    if (l == 0) sw[c4 + j] = sc;
                    mx = fmaxf(mx, sc);
                }
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
        __builtin_amdgcn_wave_barrier();
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
        float4 acc = make_float4(0, 0, 0, 0);
        float sum = 0;
#pragma unroll 4
        for (int ci = 0; ci < n; ci++) {
            const uint2 vv = reinterpret_cast<const uint2*>(V + (((cb + c0 + ci) % ring) * nkv + kvh) * hd)[l];
            const _Float16* v = (const _Float16*)&vv;
            const float p = __expf(sw[ci] - mx);
            acc.x += p * (float)v[0], acc.y += p * (float)v[1], acc.z += p * (float)v[2], acc.w += p * (float)v[3];
            sum += p;
        }
        float* dst = part + (((int64_t)b * nh + h) * ATT_SPL + s) * (hd + 2);
        reinterpret_cast<float4*>(dst)[l] = acc;
        if (l == 0) dst[hd] = mx, dst[hd + 1] = sum;
        __builtin_amdgcn_wave_barrier();
    }
}

__device__ __noinline__ void op_dattn_merge(const MegaOp& op, const Ctx& c) {
    const float* part = (const float*)op.p[6];
    float* o = (float*)op.p[3];
    int8_t* q8 = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int nh = op.i[0];
    constexpr int hd = 128;
    for (int64_t u = c.gw; u < (int64_t)c.nb * nh; u += c.GW) {
        const int h = (int)(u % nh), b = (int)(u / nh);
        int64_t cb;
        int n_cells = dattn_cells(op, c, b, cb);
        int ch;
        if (op.i[5]) {  // chunks of op_dattn_wmma: over the union of all tokens of the block
            n_cells = dattn_cells(op, c, 0, cb);
            ch = att_chunk_blk(n_cells, 1, op.i[1], gridDim.x);
        } else {
            ch = att_chunk(n_cells, PNB, nh, c.GW);
        }
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
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int e = c.lane + 32 * k;
            float a = 0;
#pragma unroll
            for (int s = 0; s < ATT_SPL; s++)
                if (s < ns) a += w[s] * src[s * (hd + 2) + e];
            const float ov = a / L;
            o[((int64_t)b * nh + h) * hd + e] = ov;
            wave_q8(ov, ((int64_t)b * nh + h) * hd + e, q8, meta);
        }
    }
}

// Runs the step if it is a drafter step; false for the other groups.
__device__ __forceinline__ bool dispatch_draft(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_DCONV:
            op_dconv(op, c);
            return true;
        case OP_DQK:
            op_dqk(op, c);
            return true;
        case OP_DATTN:
            op_dattn(op, c);
            return true;
        case OP_DATTN_SPLIT:
            op_dattn_split(op, c);
            return true;
        case OP_DATTN_MERGE:
            op_dattn_merge(op, c);
            return true;
        case OP_DATTN_WMMA:
            op_dattn_wmma(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
