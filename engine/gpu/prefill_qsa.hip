// Prefill kernels of the quantized self-attention (QSA) layers before the attention itself: preparation of q/K/V and
// the indexer, scoring of the pool blocks and the top-k selection.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

namespace {

// Which (token, unit) a wave works on; lane in the wave
struct QsaUnit {
    int64_t t, u, pos;
    int lane;
};

// K or V of one KV head to int8 in the KV cache (K gets norm + RoPE first)
__device__ __forceinline__ void qsa_kv_unit(const QsaPrepArgs a, const QsaUnit w) {
    const bool is_v = w.u >= a.nh + a.nkv;
    const int h = (int)(w.u - a.nh - (is_v ? a.nkv : 0));
    float* src = (is_v ? (float*)a.tv : a.tk) + (w.t * a.nkv + h) * a.hd;
    if (!is_v) wave_norm_rope<8>(src, src, a.k_norm, w.pos, a.base, a.eps);
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; k++) v[k] = src[w.lane + 32 * k];
    uint8_t* r = a.kv + w.pos * a.row;
    wave_q8_256(v, (int8_t*)r + h * 2 * a.hd + (is_v ? a.hd : 0),
                (float*)(r + (int64_t)a.nkv * 2 * a.hd) + h * 16 + (is_v ? 8 : 0), w.lane);
}

// Pool block of the indexer, when the token completes one: mean of the raw keys, norm, RoPE
__device__ __forceinline__ void qsa_pool_unit(const QsaPrepArgs a, const QsaUnit w) {
    if ((w.pos + 1) % a.blk != 0) return;
    const int64_t first = w.pos + 1 - a.blk;
    float* dst = a.pool + (w.pos / a.blk) * a.idim;
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) {
        float m = 0;
        for (int r = 0; r < a.blk; r++) {
            const int64_t p = first + r;
            m += (p >= a.pos0 ? a.rk + (p - a.pos0) * a.idim : a.ring + (p % RAW_RING) * a.idim)[w.lane + 32 * k];
        }
        v[k] = m * (1.0f / a.blk);
        ss += v[k] * v[k];
    }
    const float scale = rsqrtf(wave_sum(ss) / a.idim + a.eps);
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] *= scale * a.idx_k_norm[w.lane + 32 * k];
    rope_pair_yarn(v[0], v[1], w.lane, first, a.base);
#pragma unroll
    for (int k = 0; k < 4; k++) dst[w.lane + 32 * k] = v[k];
}

__global__ void k_qsa_prep(QsaPrepArgs a) {
    const auto& [qfull, qn, g, q_norm, tk, k_norm, tv, kv, row, rk, ring, pool, idx_k_norm, iq, idx_q_norm, n, pos0, nh,
                 nkv, hd, idim, ih, blk, eps, base] = a;
    const int64_t per = nh + 2 * nkv + 1 + ih;
    const int64_t u0 = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (u0 >= n * per) return;
    const int64_t t = u0 / per, u = u0 % per, pos = pos0 + t;
    const float* qf = qfull + t * 2 * nh * hd;
    if (u < nh) {
        wave_norm_rope<8>(qf + u * 2 * hd, qn + t * nh * hd + u * hd, q_norm, pos, base, eps);
        for (int i = lane; i < hd; i += WAVE) g[t * nh * hd + u * hd + i] = qf[u * 2 * hd + hd + i];
    } else if (u < nh + 2 * nkv) {
        qsa_kv_unit(a, {t, u, pos, lane});
    } else if (u == nh + 2 * nkv) {
        qsa_pool_unit(a, {t, u, pos, lane});
    } else {
        float* q = iq + t * ih * idim + (u - nh - 2 * nkv - 1) * idim;
        wave_norm_rope<4>(q, q, idx_q_norm, pos, base, eps);
    }
}

__global__ void k_idx_score(IdxScoreArgs a) {
    const auto& [iq, pool, score, t0, cnt, pos0, heads, dim, blk, topk, per, sstride] = a;
    const int64_t u = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (u >= cnt * per) return;
    const int64_t tt = u / per, pb = u % per, pos = pos0 + t0 + tt;
    if (!qsa_sel(pos, blk, topk) || pb >= (pos + 1) / blk) return;
    const float* qb = iq + (t0 + tt) * heads * dim;
    float total = 0;
    for (int h = 0; h < heads; h++) {
        float dot = 0;
        for (int i = lane; i < dim; i += WAVE) dot += qb[h * dim + i] * pool[pb * dim + i];
        total += fmaxf(wave_sum(dot), 0.0f);
    }
    if (lane == 0) score[tt * sstride + pb] = total * rsqrtf((float)dim);
}

// The same with WMMA for 4 heads of 128: the rows of the tile are (token, head), token-major with the heads inside,
// the columns are pool blocks. Block: 32 tokens (128 rows) x 128 pool blocks, K = 128 in two steps; waves 2 x 4 of
// 64 x 32. In the result one lane per WMMA tile holds 8 consecutive rows = 2 tokens x 4 heads: the ReLU sum needs no
// exchange. The scalar version ran at ~0.4 TFLOPS (1.5 s of 23 s at 20,000 tokens).
namespace ix {
constexpr int TT = 32, RB = 128, PB = 128, KS = 64, LD = KS + 8;
}  // namespace ix
// Position of a thread in the tile of a block: first token tt0 and first pool block pb0 of the tile, wave origin
// (wr, wc) and the lane's row r16 and K offset kh in the WMMA fragments
struct IxCtx {
    int tt0;
    int64_t pb0;
    int wr, wc, r16, kh;
};

// Per thread: row tid >> 1, half (tid & 1) * 32 of Q and of the pool, K slice from k0
__device__ __forceinline__ void ix_load(const IdxScoreWmmaArgs a, const IxCtx c, int k0, _Float16* Qs, _Float16* Ps) {
    using namespace ix;
    const int r = threadIdx.x >> 1, cc = (threadIdx.x & 1) * 32;
    const bool qv = c.tt0 + r / 4 < a.cnt;
    const float* qs = a.iq + ((int64_t)(a.t0 + c.tt0) * 4 + r) * 128 + k0 + cc;
    const bool pv = c.pb0 + r < a.per;
    const float* ps = a.pool + (c.pb0 + r) * 128 + k0 + cc;
#pragma unroll
    for (int q8 = 0; q8 < 4; q8++) {
        half8 va, vb;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            va[j] = qv ? (_Float16)qs[8 * q8 + j] : (_Float16)0.0f;
            vb[j] = pv ? (_Float16)ps[8 * q8 + j] : (_Float16)0.0f;
        }
        *reinterpret_cast<half8*>(&Qs[r * LD + cc + 8 * q8]) = va;
        *reinterpret_cast<half8*>(&Ps[r * LD + cc + 8 * q8]) = vb;
    }
}

// Score per (token, pool block): ReLU sum over the 4 heads, only where the token selects
__device__ __forceinline__ void ix_store(const IdxScoreWmmaArgs a, const IxCtx c, const float8 (&acc)[4][2]) {
    const float sc = rsqrtf(128.0f);
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int h2 = 0; h2 < 2; h2++) {
            const int tt = c.tt0 + (c.wr + i * 16 + c.kh) / 4 + h2;
            if (tt >= a.cnt) continue;
            const int64_t pos = a.pos0 + a.t0 + tt;
            if (!qsa_sel(pos, a.blk, a.topk)) continue;
            const int64_t nv = (pos + 1) / a.blk;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int64_t pb = c.pb0 + c.wc + j * 16 + c.r16;
                if (pb >= nv) continue;
                float tot = 0;
#pragma unroll
                for (int e = 0; e < 4; e++) tot += fmaxf(acc[i][j][4 * h2 + e], 0.0f);
                a.score[(int64_t)tt * a.sstride + pb] = tot * sc;
            }
        }
}

__global__ void __launch_bounds__(TPB) k_idx_score_wmma(IdxScoreWmmaArgs a) {
    using namespace ix;
    __shared__ __attribute__((aligned(16))) _Float16 Qs[RB * LD], Ps[PB * LD];
    const int tt0 = blockIdx.y * TT;
    const int64_t pb0 = (int64_t)blockIdx.x * PB;
    const int64_t last = a.pos0 + a.t0 + std::min(tt0 + TT, a.cnt) - 1;
    if (pb0 >= (last + 1) / a.blk) return;  // whole tile behind the last visible block
    const int tid = threadIdx.x, lane = tid & 31, w = tid >> 5;
    const int wr = (w >> 2) * 64, wc = (w & 3) * 32;
    const int r16 = lane & 15, kh = (lane >> 4) * 8;
    const IxCtx c{tt0, pb0, wr, wc, r16, kh};
    float8 acc[4][2] = {};
    for (int k0 = 0; k0 < 128; k0 += KS) {
        ix_load(a, c, k0, Qs, Ps);
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < KS; kk += 16) {
            half8 av[4], bv[2];
#pragma unroll
            for (int i = 0; i < 4; i++)
                av[i] = *reinterpret_cast<const half8*>(&Qs[(wr + i * 16 + r16) * LD + kk + kh]);
#pragma unroll
            for (int j = 0; j < 2; j++)
                bv[j] = *reinterpret_cast<const half8*>(&Ps[(wc + j * 16 + r16) * LD + kk + kh]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av[i], bv[j], acc[i][j]);
        }
        __syncthreads();
    }
    ix_store(a, c, acc);
}
}  // namespace

void set_prefill_yarn(const rope::YarnTable& t) { upload_yarn(t); }
void check_prefill_yarn(const rope::YarnTable& t) { check_yarn(t, "prefill_qsa.hip"); }
void launch_qsa_prep(const QsaPrepArgs& a, hipStream_t st) {
    const int64_t per = a.nh + 2 * a.nkv + 1 + a.ih;
    k_qsa_prep<<<blocks_for((int64_t)a.n * per * WAVE), TPB, 0, st>>>(a);
}
void launch_idx_score(const IdxScoreArgs& a, hipStream_t st) {
    k_idx_score<<<blocks_for((int64_t)a.cnt * a.per * WAVE), TPB, 0, st>>>(a);
}
void launch_idx_score_wmma(const IdxScoreWmmaArgs& a, hipStream_t st) {
    k_idx_score_wmma<<<dim3((unsigned)((a.per + ix::PB - 1) / ix::PB), (a.cnt + ix::TT - 1) / ix::TT), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
