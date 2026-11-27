// Building blocks of the Coder's program: the gated residual (hyper-connection) and the QSA attention steps.
#include <cmath>

#include "backend/env.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// Gated residual: xn is ready -> down (with silu(./hc)) and inject, up, mix + int8
void Qwen4ExpProgram::gr(const GatedResidual& S, float* inject_out, PfGR* rec) {
    const int64_t r_in = (int64_t)S.down->dims[0], r_out = (int64_t)S.down->elements() / r_in;
    const int64_t u_out = (int64_t)S.up->elements() / r_out;
    const void* wdown = hc8_ ? g_.upload_q8_0(*S.down) : g_.upload(*S.down);
    const void* winj = S.inject ? g_.upload(*S.inject) : nullptr;
    pb_.mv_begin();
    if (hc8_)
        pb_.add({8, r_in, r_out, wdown, b_.xn, &b_.qxn, b_.lo}, {-1, 0, 1.0f / dm_.hc});
    else
        pb_.addw(*S.down, wdown, b_.xn, nullptr, b_.lo, 1.0f / dm_.hc);
    if (S.inject) pb_.addw(*S.inject, winj, b_.xn, nullptr, inject_out);
    pb_.mv_end("gr.down");
    // quantization of lo while gr.up loads it (TONA_FUSE_QUANT=0: a step of its own)
    static const bool fuse_q = env::integer("TONA_FUSE_QUANT", 1) != 0;
    if (hc8_ && !fuse_q) pb_.op(OP_QUANT, {b_.lo, b_.qlo.q, b_.qlo.meta}, {(int)r_out});
    const void* wup = hc8_ ? g_.upload_q8_0(*S.up) : g_.upload(*S.up);
    pb_.mv_begin();
    if (hc8_)
        pb_.add({8, r_out, u_out, wup, b_.lo, &b_.qlo, b_.gate});
    else
        pb_.addw(*S.up, wup, b_.lo, nullptr, b_.gate);
    pb_.mv_end("gr.up");
    if (hc8_ && fuse_q) pb_.program().ops->back().i[6] = (int)r_out, pb_.program().ops->back().p[2] = b_.lo;
    pb_.op(OP_HC_MIX_QUANT, {b_.xn, b_.gate, b_.mixed, b_.qmix.q, b_.qmix.meta}, {dm_.hc, dm_.d});
    rec->down = {hc8_ ? 8u : S.down->type, r_in, r_out, wdown};
    rec->up = {hc8_ ? 8u : S.up->type, r_out, u_out, wup};
    if (S.inject) rec->inject = pb_.pw(*S.inject, winj);
    rec->norm = g_.upload_f32(*S.norm);
}

void Qwen4ExpProgram::combine(const float* yy, const float* inj, const GatedResidual* next) {
    pb_.op(OP_COMBINE_NORM,
           {b_.R[cur_], (void*)yy, (void*)inj, b_.R[cur_ ^ 1], next ? (void*)g_.upload_f32(*next->norm) : nullptr,
            b_.xn, b_.qxn.q, b_.qxn.meta},
           {dm_.hc, dm_.d}, {dm_.eps});
    cur_ ^= 1;
}

// The input of layer l is in R[cur] after the last combine before it (before PLE)
void Qwen4ExpProgram::feat(uint32_t l) {
    const int nfeat = (int)g_.feat_layers_.size() * dm_.d;
    for (size_t k = 0; k < g_.feat_layers_.size(); k++)
        if (g_.feat_layers_[k] == (int64_t)l)
            pb_.op(OP_HC_MEAN, {b_.R[cur_], g_.feat_}, {dm_.hc, dm_.d, nfeat, (int)k * dm_.d});
}

// Index scores, top-k and attention over the KV cache `kv` (target layer or MTP head) with the pool `pool`
void Qwen4ExpProgram::emit_attention(uint8_t* kv, float* pool) {
    const int topk = (int)h_.idx_top_k, kv_row = (int)g_.kv_row_, sstride = (int)g_.sstride_;
    const float scale = 1.0f / std::sqrt((float)dm_.hd);
    // A shared layer attends to b_.cells as the last full layer left it: the buffer is one per pass and no step
    // between the two layers writes it.
    if (!idx_shared_) {
        pb_.op(OP_IDX_SCORE, {b_.t_idxq, pool, b_.scores},
               {(int)h_.idx_heads, dm_.idim, dm_.blk, sstride, 0, dm_.blk, 0, topk});
        pb_.op(OP_TOPK, {b_.scores, b_.cells}, {dm_.blk, topk, 0, sstride, 0, dm_.blk, 0, topk});
    }
    if (attn_old_) {
        pb_.op(OP_ATTN, {b_.t_qn, kv, nullptr, b_.cells, b_.t_g, b_.t_o, b_.q_o.q, b_.q_o.meta},
               {dm_.nh, dm_.nkv, dm_.hd, dm_.ost, kv_row, dm_.blk, 0, topk}, {scale});
        return;
    }
    pb_.op(attn_wave_ ? OP_ATTN_SPLIT : OP_ATTN_WMMA,
           {b_.t_qn, kv, b_.attn_part, b_.cells, b_.t_g, b_.t_o, b_.q_o.q, b_.q_o.meta},
           {dm_.nh, dm_.nkv, dm_.hd, dm_.ost, kv_row, dm_.blk, 0, topk}, {scale});
    pb_.op(OP_ATTN_MERGE, {b_.t_qn, kv, b_.attn_part, b_.cells, b_.t_g, b_.t_o, b_.q_o.q, b_.q_o.meta},
           {dm_.nh, dm_.nkv, dm_.hd, dm_.ost, kv_row, dm_.blk, attn_wave_ ? 0 : 1, topk});
}

}  // namespace tona::gpu
