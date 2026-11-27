// The target program: embedding, the layers (QSA attention or DeltaNet, then router), PLE and the head.
#include "backend/env.h"
#include "backend/idx_share.h"
#include "gpu/arch/qwen4exp/prefill.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void Qwen4ExpProgram::build_target() {
    res_ = &Qwen4ExpPfRes::reset(g_.pf_arch_, h_.n_layer);
    int n_qsa = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) n_qsa += h_.is_qsa(l);
    idx_full_ = env::idx_full_layers(n_qsa);
    pb_.op(OP_BCAST_NORM,
           {g_.e_, b_.R[cur_], b_.xn, (void*)g_.upload_f32(*m_.layers()[0].gr_attn.norm), b_.qxn.q, b_.qxn.meta},
           {dm_.hc, dm_.d}, {dm_.eps});
    layer(0);
    for (uint32_t l = 1; l <= h_.n_layer; l++) {
        experts(l - 1, l == h_.ple_layer);
        between_layers(l);
    }
}

// What follows the experts of layer l - 1: the head, PLE, or the combine into layer l
void Qwen4ExpProgram::between_layers(uint32_t l) {
    if (l == h_.n_layer) {
        head_block();
    } else if (l == h_.ple_layer) {
        ple_block(l);
    } else {
        combine(b_.y, b_.inject_f, &m_.layers()[l].gr_attn);
        feat(l);
        layer(l);
    }
}

void Qwen4ExpProgram::head_block() {
    combine(b_.y, b_.inject_f, &m_.gr_head());
    gr(m_.gr_head(), nullptr, &res_->head);
    pb_.mv_begin();
    g_.head_w_ = g_.upload(*m_.output());
    g_.pf_out_ = pb_.pw(*m_.output(), g_.head_w_);
    pb_.addw(*m_.output(), g_.head_w_, b_.mixed, &b_.qmix, g_.logits_);
    pb_.mv_end("head.out");
}

void Qwen4ExpProgram::ple_block(uint32_t l) {
    const Layer& P = m_.layers()[h_.ple_layer];
    combine(b_.y, b_.inject_f, nullptr);
    feat(l);
    g_.pf_ple_nk_ = g_.upload_f32(*P.ple_norm_key), g_.pf_ple_nq_ = g_.upload_f32(*P.ple_norm_query);
    g_.pf_ple_nc_ = g_.upload_f32(*P.ple_norm_conv), g_.pf_ple_conv_ = g_.upload_f32(*P.ple_conv1d);
    g_.pf_ple_hist_ = b_.ple_hist;
    pb_.op(OP_PLE_GATE,
           {b_.R[cur_], b_.ple_k, b_.ple_v, (void*)g_.pf_ple_nk_, (void*)g_.pf_ple_nq_, (void*)g_.pf_ple_nc_, b_.ple_g,
            b_.ple_n},
           {dm_.hc, dm_.d}, {dm_.eps});
    pb_.op(OP_PLE_CONV, {b_.R[cur_], b_.ple_g, b_.ple_n, (void*)g_.pf_ple_conv_, b_.ple_hist, b_.R[cur_ ^ 1]},
           {(int)dm_.hd_all, (int)h_.ple_conv, (int)h_.ple_ngram});
    cur_ ^= 1;
    pb_.op(OP_COMBINE_NORM,
           {b_.R[cur_], nullptr, nullptr, nullptr, (void*)g_.upload_f32(*m_.layers()[l].gr_attn.norm), b_.xn, b_.qxn.q,
            b_.qxn.meta},
           {dm_.hc, dm_.d}, {dm_.eps});
    layer(l);
}

void Qwen4ExpProgram::layer(uint32_t l) {
    const Layer& S = m_.layers()[l];
    PfLayer& F = g_.pf_[l];
    gr(S.gr_attn, b_.inject_a, &res_->layers[l].attn);
    if (h_.is_qsa(l)) {
        idx_shared_ = !idx_full_[(l + 1) / h_.full_attn_interval - 1];
        layer_qsa(S, F);
        idx_shared_ = false;
    } else
        layer_deltanet(S, F);
    combine(b_.y, b_.inject_a, &S.gr_ffn);
    gr(S.gr_ffn, b_.inject_f, &res_->layers[l].ffn);
    layer_router(l, S, F);
}

void Qwen4ExpProgram::layer_qsa(const Layer& S, PfLayer& F) {
    const Dims& s = dm_;
    const size_t kv_bytes = (size_t)g_.max_ctx_ * g_.kv_row_, pool_floats = (size_t)(g_.max_ctx_ / s.blk + 1) * s.idim;
    uint8_t* kv = pb_.alloc_kv(kv_bytes);
    if (g_.kv_host_) g_.kv_hptr_.push_back(static_cast<uint8_t*>(g_.pin_allocs_.back()));
    g_.seq_buf(kv, kv_bytes, g_.kv_host_ ? g_.kv_hptr_.back() : nullptr);
    // The ring of raw indexer keys is part of the state: after a jump back to the restore point the unfinished block
    // of 4 needs the keys of its first tokens, which the generation would overwrite
    float* ring = g_.alloc_state((size_t)RAW_RING * s.idim);
    float* pool = g_.alloc(pool_floats);
    g_.seq_buf(pool, pool_floats * sizeof(float));
    F.attn.kv = kv, F.attn.idx.raw_ring = ring, F.attn.idx.pool = pool;
    F.attn.idx.idx_k = pb_.pw(*S.idx_k, g_.upload(*S.idx_k));
    F.attn.q = pb_.pw(*S.q, g_.upload(*S.q));
    F.attn.k = pb_.pw(*S.k, g_.upload(*S.k));
    F.attn.v = pb_.pw(*S.v, g_.upload(*S.v));
    F.attn.idx.idx_q = pb_.pw(*S.idx_q, g_.upload(*S.idx_q));
    pb_.mv_begin();
    pb_.addw(*S.idx_k, F.attn.idx.idx_k.w, b_.mixed, &b_.qmix, b_.t_rk);
    pb_.addw(*S.q, F.attn.q.w, b_.mixed, &b_.qmix, b_.t_qfull);
    pb_.addw(*S.k, F.attn.k.w, b_.mixed, &b_.qmix, b_.t_k);
    pb_.addw(*S.v, F.attn.v.w, b_.mixed, &b_.qmix, b_.t_v);
    pb_.addw(*S.idx_q, F.attn.idx.idx_q.w, b_.mixed, &b_.qmix, b_.t_idxq);
    pb_.mv_end("qsa.in");
    F.attn.q_norm = g_.upload_f32(*S.q_norm), F.attn.k_norm = g_.upload_f32(*S.k_norm);
    F.attn.idx.idx_k_norm = g_.upload_f32(*S.idx_k_norm), F.attn.idx.idx_q_norm = g_.upload_f32(*S.idx_q_norm);
    pb_.op(OP_QSA_PREP,
           {b_.t_qfull, b_.t_qn, b_.t_g, (void*)F.attn.q_norm, b_.t_k, (void*)F.attn.k_norm, b_.t_rk, pool,
            (void*)F.attn.idx.idx_k_norm, b_.t_idxq, (void*)F.attn.idx.idx_q_norm, kv, b_.t_v, ring},
           {s.nh, s.nkv, s.hd, s.idim, (int)h_.idx_heads, s.blk, (int)g_.kv_row_, (int)h_.idx_top_k}, {s.eps, s.base});
    emit_attention(kv, pool);
    F.attn.o = pb_.pw(*S.o, g_.upload(*S.o));
    pb_.mv_begin();
    pb_.addw(*S.o, F.attn.o.w, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("qsa.out");
}

void Qwen4ExpProgram::layer_deltanet(const Layer& S, PfLayer& F) {
    const Dims& s = dm_;
    float* state = g_.alloc_state((size_t)s.nv * DS * DS);
    float* ring = g_.alloc_state((size_t)(h_.dn_conv - 1 + MEGA_BMAX) * s.ch);
    float* pend_d = g_.alloc_state((size_t)2 * MEGA_BMAX * s.vd);
    float* pend_a = g_.alloc_state((size_t)2 * MEGA_BMAX * s.nv);
    float* pend_k = g_.alloc_state((size_t)2 * MEGA_BMAX * s.kd);
    F.dn.state = state, F.dn.ring = ring, F.dn.pend_d = pend_d, F.dn.pend_a = pend_a, F.dn.pend_k = pend_k;
    F.dn.qkv = pb_.pw(*S.qkv, g_.upload(*S.qkv));
    F.dn.gate = pb_.pw(*S.gate, g_.upload(*S.gate));
    F.dn.beta = pb_.pw(*S.beta, g_.upload(*S.beta));
    F.dn.alpha = pb_.pw(*S.alpha, g_.upload(*S.alpha));
    pb_.mv_begin();
    pb_.addw(*S.qkv, F.dn.qkv.w, b_.mixed, &b_.qmix, b_.t_qkv);
    pb_.addw(*S.gate, F.dn.gate.w, b_.mixed, &b_.qmix, b_.t_z);
    pb_.addw(*S.beta, F.dn.beta.w, b_.mixed, &b_.qmix, b_.t_b);
    pb_.addw(*S.alpha, F.dn.alpha.w, b_.mixed, &b_.qmix, b_.t_a);
    pb_.mv_end("dn.in");
    F.dn.conv_w = g_.upload_f32(*S.conv1d), F.dn.dt_bias = g_.upload_f32(*S.dt_bias), F.dn.A = g_.upload_f32(*S.a);
    F.dn.ssm_norm = g_.upload_f32(*S.ssm_norm);
    pb_.op(OP_DN_CONV, {b_.t_qkv, ring, (void*)F.dn.conv_w, b_.t_conv}, {s.kd, s.vd, (int)h_.dn_conv, DS, s.nk},
           {s.eps});
    pb_.op(OP_DN_RECUR,
           {b_.t_conv, b_.t_b, b_.t_a, (void*)F.dn.dt_bias, (void*)F.dn.A, pend_d, pend_a, state, b_.t_raw, pend_k},
           {s.nk, s.nv, DS, s.kd}, {s.eps});
    pb_.op(OP_DN_NORM, {b_.t_raw, (void*)F.dn.ssm_norm, b_.t_z, b_.t_o, b_.q_o.q, b_.q_o.meta}, {s.nv, s.ost}, {s.eps});
    F.dn.ssm_out = pb_.pw(*S.ssm_out, g_.upload(*S.ssm_out));
    pb_.mv_begin();
    pb_.addw(*S.ssm_out, F.dn.ssm_out.w, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("dn.out");
}

// Router weights once per layer in VRAM (layer l also needs that of l + 1 for prefetching)
const void* Qwen4ExpProgram::router_weights(uint32_t l) {
    if (!router_dev_[l]) router_dev_[l] = g_.upload(*m_.layers()[l].router);
    return router_dev_[l];
}

void Qwen4ExpProgram::layer_router(uint32_t l, const Layer& S, PfLayer& F) {
    F.moe.router = pb_.pw(*S.router, router_weights(l));
    const bool pre = g_.cpu_mode_ && g_.prefetch_n_ > 0 && l + 1 < h_.n_layer;
    pb_.mv_begin();
    pb_.addw(*S.router, F.moe.router.w, b_.mixed, &b_.qmix, b_.router_out);
    if (pre) pb_.addw(*m_.layers()[l + 1].router, router_weights(l + 1), b_.mixed, &b_.qmix, b_.pred_out);
    pb_.mv_end("router");
    const LayerCache& C = g_.cache_[l];
    if (g_.cpu_mode_)
        pb_.op(OP_HOST_WAIT,
               {g_.mbox_dev_, b_.router_out, b_.qmix.q, b_.qmix.meta, g_.cpu_io_dev_, g_.cpu_io_dev_ + g_.cpu_xm_off_,
                pre ? b_.pred_out : nullptr},
               {(int)l, dm_.d, (int)g_.experts_.experts(l)});
    else
        pb_.op(OP_HOST_WAIT, {g_.mbox_dev_, b_.router_out}, {(int)l, 0, (int)g_.experts_.experts(l)});
    pb_.program().ops->back().f[3] = HOST_WAIT_MS + 200;  // the other blocks wait with it while the CPU reads
    shared(l, l + 1 == h_.ple_layer);
    pb_.op(OP_EXPERT_COPY, {g_.cpu_mode_ ? g_.fill_dev_ + l * g_.fill_stride_ : g_.staging_dev_, C.slots},
           {(int)C.slot_bytes, g_.cpu_mode_ ? (int)(g_.fill_stride_ / 2 / C.slot_bytes) : 0});
}

}  // namespace tona::gpu
