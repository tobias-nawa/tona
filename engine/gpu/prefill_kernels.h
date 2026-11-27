// Arguments and launchers of the prefill kernels. A kernel takes one struct by value (the argument buffer is the same
// as with separate parameters: same order, same alignment); the launcher next to the kernel computes the grid. The
// host code in prefill_*.cpp only sees this header, because a kernel can only be launched from its own .hip file.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "backend/rope_yarn.h"

namespace tona::gpu {

// Residual streams and gated residual (prefill_hc.hip)

// R[t][g * d + i] = e[t][i] (e: 2 * d per token, embedding first)
struct BcastArgs {
    const float* e;
    float* R;
    int n, hc, d;
};
void launch_bcast(const BcastArgs& a, hipStream_t st);

// Block per token: scale per stream (1/RMS) to sc[t][4], xn = R * scale * w
struct Norm16Args {
    const float* R;
    const float* w;
    _Float16* xn;
    float* sc;
    int hc, d;
    float eps;
};
void launch_norm16(const Norm16Args& a, int n_tokens, hipStream_t st);

// MTP input, block per (token i, stream g) of a part from t0 on: x16 = [enorm(e_i) | hnorm(h_g)] with h = R at
// t0 + i - 1 (hc * d per token), for t0 + i = 0 hfirst (null: 0)
struct MtpIn16Args {
    const float* e;
    const float* R;
    const float* hfirst;
    const float* wn_e;
    const float* wn_h;
    _Float16* x16;
    int t0, hc, d;
    float eps;
};
void launch_mtp_in16(const MtpIn16Args& a, int blocks, hipStream_t st);

// k_combine and k_norm16 in one: R += y * 2 sigmoid(inject / hc) per stream, then the scale per stream and xn
struct CombineNorm16Args {
    float* R;
    const float* y;
    const float* inj;
    const float* w;
    _Float16* xn;
    float* sc;
    int hc, d;
    float eps;
};
void launch_combine_norm16(const CombineNorm16Args& a, int n_tokens, hipStream_t st);

// y16 = f16(silu(x * scale)), n values
struct Act16Args {
    const float* x;
    _Float16* y;
    int64_t n;
    float scale;
};
void launch_act16(const Act16Args& a, hipStream_t st);

// mixed = mean over the streams of norm(R) * sigmoid(gate), float and f16
struct MixArgs {
    const float* R;
    const float* sc;
    const float* w;
    const float* gate;
    float* mixed;
    _Float16* mixed16;
    int n, hc, d;
};
void launch_mix(const MixArgs& a, hipStream_t st);

// R += y * 2 sigmoid(inject / hc) per stream; inject: 4 values per token
struct CombineArgs {
    float* R;
    const float* y;
    const float* inj;
    int n, hc, d;
};
void launch_combine(const CombineArgs& a, hipStream_t st);

// f16 copy: rows rows of cols values, row stride ldx and ldy
struct To16Args {
    const float* x;
    int64_t ldx;
    _Float16* y;
    int64_t ldy;
    int rows, cols;
};
void launch_to16(const To16Args& a, hipStream_t st);

// Mean over the streams of R (features for the drafter): F[t][col0 + i]
struct HcMeanArgs {
    const float* R;
    float* F;
    int n, hc, d, ldf, col0;
};
void launch_hc_mean(const HcMeanArgs& a, hipStream_t st);

// Ring after a chunk: the last min(n, ring) rows of src at position mod ring
struct RingStoreArgs {
    const float* src;
    float* ring;
    int n;
    int64_t pos0;
    int cols, nring;
};
void launch_ring_store(const RingStoreArgs& a, hipStream_t st);

// DeltaNet (prefill_dn.hip)

// Causal conv + SiLU followed by the L2 norm of the q and k heads (the first `heads` heads of DS channels, wave per
// (token, head)); input at position p from the chunk (p >= pos0) or the ring (K - 1 + BMAX places, as the
// megakernel), null before position 0
struct DnConvArgs {
    const float* qkv;
    const float* ring;
    const float* w;
    float* out;
    int n;
    int64_t pos0;
    int ch, K, heads;
    float eps;
    int t0 = 0, m = -1;  // the launch covers the tokens t0 ... t0 + m - 1 of the n (m < 0: all of them)
};
void launch_dn_conv_l2(const DnConvArgs& a, hipStream_t st);

// Delta rule over the tokens of a chunk, one after the other (block-synchronized form). First commits the m still
// open tokens of the last decode pass (half `half` of the intermediate buffers).
// conv: ch per token; ba: per token [beta logits nv | alpha logits nv]; raw: vd per token
struct DnRecurArgs {
    const float* conv;
    const float* ba;
    const float* dt_bias;
    const float* A;
    float* state;
    const float* pk;
    const float* pd;
    const float* pa;
    int m, half;
    float* raw;
    int n, nk, nv, kd, ch;
};
void launch_dn_recur(const DnRecurArgs& a, hipStream_t st);

// Delta rule token by token without a block-wide sync: one wave per (V head, 16 v columns)
struct DnWaveArgs {
    const float* conv;
    const float* ba;
    const float* dt_bias;
    const float* A;
    float* state;
    float* raw;
    int n, nk, nv, kd, ch;
};
void launch_dn_wave(const DnWaveArgs& a, hipStream_t st);

// Norm per V head * w * sigmoid(z) to f16 (the Coder), wave per (token, head)
struct DnNormArgs {
    const float* raw;
    const float* wn;
    const float* z;
    _Float16* o;
    int n, nv;
    float eps;
};
void launch_dn_norm(const DnNormArgs& a, hipStream_t st);
// The same with silu(z) as the gate (Ornith)
void launch_dn_norm_silu(const DnNormArgs& a, hipStream_t st);

// Quantized self-attention: preparation, indexer, top-k (prefill_qsa.hip)

// Wave per (token, unit) like op_qsa_prep: q heads (norm + RoPE, split off the gate), K/V per KV head to int8 into the
// KV cache, indexer pool (when the token completes a block), indexer q (norm + RoPE).
// qfull: 2 * nh * hd per token; qn, g: nh * hd; tk, tv: nkv * hd; rk: raw indexer key (idim); iq: ih * idim;
// ring: raw keys before the chunk (position mod RAW_RING)
struct QsaPrepArgs {
    const float* qfull;
    float* qn;
    float* g;
    const float* q_norm;
    float* tk;
    const float* k_norm;
    const float* tv;
    uint8_t* kv;
    int64_t row;
    const float* rk;
    const float* ring;
    float* pool;
    const float* idx_k_norm;
    float* iq;
    const float* idx_q_norm;
    int n;
    int64_t pos0;
    int nh, nkv, hd, idim, ih, blk;
    float eps, base;
};
// YaRN table of the RoPE in launch_qsa_prep (constant memory of this translation unit); native until set
void set_prefill_yarn(const rope::YarnTable& t);
void check_prefill_yarn(const rope::YarnTable& t);  // read-back; throws if the upload did not happen
void launch_qsa_prep(const QsaPrepArgs& a, hipStream_t st);

// Score of (token t0 + tt, pool block pb): sum over heads of ReLU(q_h . k_b) / sqrt(dim), only for tokens with a
// selection
struct IdxScoreArgs {
    const float* iq;
    const float* pool;
    float* score;
    int t0, cnt;
    int64_t pos0;
    int heads, dim, blk, topk;
    int64_t per, sstride;
};
void launch_idx_score(const IdxScoreArgs& a, hipStream_t st);

// The same with WMMA for 4 heads of 128
struct IdxScoreWmmaArgs {
    const float* iq;
    const float* pool;
    float* score;
    int t0, cnt;
    int64_t pos0;
    int blk, topk;
    int64_t per, sstride;
};
void launch_idx_score_wmma(const IdxScoreWmmaArgs& a, hipStream_t st);

// Block (512 threads) per token with a selection: the n_top best pool blocks like op_topk. cells: MAX_CELLS per token
struct TopkArgs {
    const float* score_f;
    int32_t* cells_all;
    int32_t* ncells;
    int t0;
    int64_t pos0;
    int blk, topk;
    int64_t sstride;
};
void launch_topk(const TopkArgs& a, int blocks, hipStream_t st);

// Attention (prefill_attn.hip)

// Block per (token, head), 256 threads (= hd): softmax attention over the cells (selection or all up to pos),
// * sigmoid(gate) to f16. q, g, o: nh * hd per token
struct AttnArgs {
    const float* q;
    const uint8_t* kv;
    int64_t row;
    const int32_t* cells_all;
    const int32_t* ncells;
    const float* gate;
    _Float16* o;
    int64_t pos0;
    int nh, nkv, hd, blk, topk;
    float scale;
};
void launch_attn(const AttnArgs& a, int n_tokens, hipStream_t st);

// Flash attention with WMMA: block per (token, KV head); hd = 256, the group of q heads fills one 16-row tile
struct AttnWmmaArgs {
    const float* q;
    const uint8_t* kv;
    int64_t row;
    const int32_t* cells_all;
    const int32_t* ncells;
    const float* gate;
    _Float16* o;
    int64_t pos0;
    int nh, nkv, blk, topk;
    float scale;
    bool kv_host;  // the cache is in pinned RAM: the direct gather is faster there
};
void launch_attn_wmma(const AttnWmmaArgs& a, int n_tokens, hipStream_t st);
// The gather of the kernel by number of the second kernel (0 V after S, 1 V with K, 2 next rows ahead), 3 and 4 the
// first kernel (direct, pipelined): tests and benchmarks
void launch_attn_wmma_gather(const AttnWmmaArgs& a, int n_tokens, hipStream_t st, int mode);

// MoE (prefill_moe.hip)

// Wave per token: softmax over E (<= max_e) router logits, the ne largest (ties: the smaller index), weights p / sum p
// of the chosen; counts per expert. max_e: 256 or 512 (expert_cap), one instantiation of the kernel each
struct RouteArgs {
    const float* logits;
    int32_t* sel;
    float* sw;
    int32_t* cnt;
    int n, E, ne;
    int max_e = 256;
};
void launch_route(const RouteArgs& a, hipStream_t st);

// Wave per assignment (token, rank): take a row in the expert's batch, copy x (f16, d values) there
struct ScatterArgs {
    const int32_t* sel;
    const int32_t* off;
    int32_t* fill;
    int32_t* rowmap;
    const _Float16* x;
    _Float16* xg;
    int n, ne, d;
};
void launch_scatter(const ScatterArgs& a, hipStream_t st);

// act16 = f16(silu(g) * u), rows rows of ff from row0
struct Swiglu16Args {
    const float* g;
    const float* u;
    _Float16* act;
    int64_t row0, rows;
    int ff;
};
void launch_swiglu16(const Swiglu16Args& a, hipStream_t st);

// TONA_PF_CHECK: flag[0] counts the violations, flag[1..3] hold assignment, expert and row of the first one. stage 0
// (after the router): sel inside [0, E). stages 1 (after the scatter) and 2 (before k_moe_sum): rowmap inside the range
// of the expert.
struct MoeCheckArgs {
    const int32_t* sel;
    const int32_t* rowmap;
    const int32_t* cnt;
    const int32_t* off;
    int32_t* flag;
    int n, ne, E, stage;
};
void launch_moe_check(const MoeCheckArgs& a, hipStream_t st);

// y = sum_r w_r * out[rowmap] + sigmoid(g_shared) * out_shared
struct MoeSumArgs {
    const float* out;
    const int32_t* rowmap;
    const float* sw;
    const float* sho;
    const float* shi;
    float* y;
    int n, d, ne;
};
void launch_moe_sum(const MoeSumArgs& a, hipStream_t st);

// PLE (prefill_ple.hip)

// Block per (token, stream) like op_ple_gate. R, key, gated, normed: hc * d per token; val: d per token
struct PleGateArgs {
    const float* R;
    const float* key;
    const float* val;
    const float* w_key;
    const float* w_query;
    const float* w_conv;
    float* gated;
    float* normed;
    int hc, d;
    float eps;
};
void launch_ple_gate(const PleGateArgs& a, int n_tokens, hipStream_t st);

// Thread per (token, channel): dilated causal conv over normed (from the chunk or the ring with (K - 1) * dil + BMAX
// places), R += gated + silu(conv)
struct PleConvArgs {
    float* R;
    const float* gated;
    const float* normed;
    const float* w;
    const float* hist;
    int n;
    int64_t pos0;
    int cols, K, dil;
};
void launch_ple_conv(const PleConvArgs& a, hipStream_t st);

// LM head for the check (prefill_head.hip)

// Block per token: running maximum and sum of exp over a slice of the logits [r0, r0 + rs), logit of the target
struct LseArgs {
    const float* L;
    int rs, r0;
    const int32_t* tgt;
    float* mrun;
    float* srun;
    float* tlog;
};
void launch_lse(const LseArgs& a, int n_tokens, hipStream_t st);

// p[0..n) = v
struct FillArgs {
    float* p;
    int n;
    float v;
};
void launch_fill(const FillArgs& a, hipStream_t st);

}  // namespace tona::gpu
