// Small GPU operations of the forward pass (version 1: one kernel per step, activations in F32).
// Same computation as cpu/ref.cpp; the layouts of the states may differ (see each).
// Operations with many parameters take them as a struct; the stream is always the last argument.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "backend/rope_yarn.h"

namespace tona::gpu::ops {

// out[g*d + i] = x[g*d + i] / sqrt(mean_g(x^2) + eps) * w[g*d + i]   (w may be null, then 1)
struct RmsNormArgs {
    const float* x;
    const float* w;
    float* out;
    int groups, d;
    float eps;
};
void rms_norm_groups(const RmsNormArgs& a, hipStream_t s = nullptr);
// v <- silu(v * scale)
void silu_scale(float* v, int n, float scale, hipStream_t s = nullptr);
// mixed[i] = mean_c xn[c*d + i] * sigmoid(gate[c*d + i])
void hc_mix(const float* xn, const float* gate, float* mixed, int hc, int d, hipStream_t s = nullptr);
// R[c*d + i] += y[i] * 2 sigmoid(inject[c] / hc)
void hc_combine(float* R, const float* y, const float* inject, int hc, int d, hipStream_t s = nullptr);
// R[c*d + i] = e[i] for all streams c
void broadcast_rows(float* R, const float* e, int hc, int d, hipStream_t s = nullptr);

// DeltaNet: causal conv of width K over the ring buffer state [K-1][ch] (slot pos mod (K-1)), then SiLU
struct DnConvArgs {
    const float* qkv;
    float* state;
    const float* w;
    float* out;
    int ch, K;
    int64_t pos;
};
void dn_conv(const DnConvArgs& a, hipStream_t s = nullptr);
// L2 norm per head like build_gdn_l2_norm: rms_norm(x, eps/n) * 1/sqrt(n)
void l2norm_heads(float* x, int heads, int dim, float eps, hipStream_t s = nullptr);
// Delta rule per V head, then RMSNorm * w_norm * sigmoid(z). State in GPU layout [V head][k][v] (v contiguous).
struct DnRecurArgs {
    const float *q, *k, *v, *b, *a, *dt_bias, *A, *w_norm, *z;
    float *state, *o;
    int nk, nv, S;
    float eps;
};
void dn_recur(const DnRecurArgs& a, hipStream_t s = nullptr);

// QSA: qfull per head [q | gate] -> q, gate
void qsa_split(const float* qfull, float* q, float* gate, int heads, int hd, hipStream_t s = nullptr);
// YaRN table of the RoPE below (this translation unit has its own constant memory); native until set
void set_yarn(const rope::YarnTable& t);
void check_yarn(const rope::YarnTable& t);  // read-back; throws if the upload did not happen
// RMSNorm per head with weight, then NeoX RoPE on the first n_rot dimensions at position pos
struct HeadNormRopeArgs {
    float* x;
    const float* w;
    int heads, hd, n_rot;
    int64_t pos;
    float base, eps;
};
void head_norm_rope(const HeadNormRopeArgs& a, hipStream_t s = nullptr);
// Indexer: mean of blk raw keys from row first -> RMSNorm * w -> RoPE at first -> out
struct IdxPoolArgs {
    const float* raw;
    int64_t first;
    int blk, dim;
    const float* w;
    int n_rot;
    float base, eps;
    float* out;
};
void idx_pool(const IdxPoolArgs& a, hipStream_t s = nullptr);
// score[b] = sum_h ReLU(q_h . pool_b) / sqrt(dim)
struct IdxScoreArgs {
    const float* q;
    const float* pool;
    float* score;
    int heads, dim;
    int64_t n_pool;
};
void idx_score(const IdxScoreArgs& a, hipStream_t s = nullptr);
// Softmax attention over cells (or 0..n_cells-1 if cells is null); Q head h reads KV head h / (nh/nkv)
struct AttentionArgs {
    const float *q, *K, *V;
    const int32_t* cells;
    int n_cells;
    float* o;
    int nh, nkv, hd;
    float scale;
};
void attention(const AttentionArgs& a, hipStream_t s = nullptr);
// x[i] *= sigmoid(g[i])
void mul_sigmoid(float* x, const float* g, int n, hipStream_t s = nullptr);

// MoE: g[i] = silu(g[i]) * u[i]
void silu_mul(float* g, const float* u, int n, hipStream_t s = nullptr);
// y[i] += w * x[i]   resp. y[i] += sigmoid(*gate) * x[i]
void axpy(float* y, const float* x, float w, int n, hipStream_t s = nullptr);
void axpy_sigmoid(float* y, const float* x, const float* gate, int n, hipStream_t s = nullptr);

// PLE after reading key/val: gate_c, gated, normalization, dilated conv over the ring buffer hist
// [(K-1)*dil][hc*d], R += gated + silu(conv)
struct PleArgs {
    float* R;
    const float *key, *val, *w_key, *w_query, *w_conv_norm, *conv_w;
    float *hist, *scratch;
    int hc, d, K, dil;
    int64_t pos;
    float eps;
};
void ple_apply(const PleArgs& a, hipStream_t s = nullptr);

}  // namespace tona::gpu::ops
