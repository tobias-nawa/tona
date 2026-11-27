// Encoders of the layer parts for nb tokens: n-gram embedding, DeltaNet, QSA and MoE (engine/cpu/ref_qwen4exp.cpp, same
// names)
#include <cmath>

#include "backend/metal/host_math.h"
#include "backend/metal/kernels/params.h"
#include "backend/metal/pass.h"

namespace tona::metal {

namespace {

MTLSize n1(uint64_t n) { return MTLSizeMake(n, 1, 1); }
uint64_t ceil_div(uint64_t a, uint64_t b) { return (a + b - 1) / b; }

}  // namespace

void Pass::ple(Dispatch& d, const Layer& L, Recurrent& r, int nb) {
    const Hparams& h = h_;
    d.run("gather_" + ple_.kernel_type, {s_.ple_stage, s_.ple_rows, s_.e}, GatherParams{h.ple_dim, ple_.row_bytes, 1},
          MTLSizeMake(ceil_div(h.ple_dim / 8, 64), (NSUInteger)h.ple_heads * nb, 1), MTLSizeMake(64, 1, 1));
    mv(d, w_.mat(L.ple_key), s_.e, s_.key, nb);
    mv(d, w_.mat(L.ple_value), s_.e, s_.val, nb);
    const uint32_t hist = (h.ple_conv - 1) * h.ple_ngram;
    d.run("ple_mix",
          {s_.R, s_.key, s_.val, w_.vec(L.ple_norm_key), w_.vec(L.ple_norm_query), w_.vec(L.ple_norm_conv),
           w_.vec(L.ple_conv1d), r.ple_ring},
          PleParams{h.d, h.n_hc, h.ple_conv, h.ple_ngram, hist, r.start, (uint32_t)nb, h.rms_eps}, n1(h.n_hc), n1(256));
}

// A decode pass keeps the recurrence's inputs per layer (st.keep), so commit(m < nb) can replay it (replay_dn)
void Pass::deltanet(Dispatch& d, const Layer& L, int i, SeqState& st, int nb) {
    const Hparams& h = h_;
    const uint32_t nv = h.dn_v_heads;
    const bool keep = !gemm_ && nb <= BMAX;
    const DnIn in{keep ? st.keep.qkv[i] : s_.qkv, keep ? st.keep.alpha[i] : s_.alpha, keep ? st.keep.beta[i] : s_.beta,
                  nb};
    const DnParams p{2 * h.dn_k_dim() + h.dn_v_dim(), h.dn_conv, h.dn_k_heads, nv, (uint32_t)nb, h.rms_eps};
    part(d, "dn_proj");
    mv(d, w_.mat(L.qkv), s_.mixed, in.qkv, nb);
    mv(d, w_.mat(L.gate), s_.mixed, s_.z, nb);
    mv(d, w_.mat(L.beta), s_.mixed, in.beta, nb);
    mv(d, w_.mat(L.alpha), s_.mixed, in.alpha, nb);
    part(d, "dn_scan");
    dn_recur(d, L, i, st.live, in);
    d.run("dn_out", {s_.o, s_.z, w_.vec(L.ssm_norm)}, p, n1((uint64_t)nv * nb), n1(256));
    part(d, "dn_proj");
    mv(d, w_.mat(L.ssm_out), s_.o, s_.y, nb);
}

// Conv window, L2 norm and state step of in.nb tokens; the outputs into s_.conv and s_.o
void Pass::dn_recur(Dispatch& d, const Layer& L, int i, Recurrent& r, const DnIn& in) {
    const Hparams& h = h_;
    const uint32_t nv = h.dn_v_heads;
    const DnParams p{2 * h.dn_k_dim() + h.dn_v_dim(), h.dn_conv, h.dn_k_heads, nv, (uint32_t)in.nb, h.rms_eps};
    d.run("dn_conv", {in.qkv, r.dn_win[i], w_.vec(L.conv1d), s_.conv}, p, n1(ceil_div(p.ch, 256)), n1(256));
    d.run("dn_l2", {s_.conv}, p, n1(2ull * h.dn_k_heads * in.nb), n1(256));
    // 4 simdgroups of 8 state rows each per threadgroup: 128 rows in 4 threadgroups per V head
    d.run("dn_step", {r.dn_state[i], s_.conv, in.alpha, in.beta, w_.vec(L.dt_bias), w_.vec(L.a), s_.o}, p,
          MTLSizeMake(4, nv, 1), n1(128));
}

void Pass::replay_dn(id<MTLComputeCommandEncoder> enc, SeqState& st, int m) {
    Dispatch d{enc, dev_};
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        if (h_.is_qsa(l)) continue;
        const int i = slot_of_layer_[l];
        dn_recur(d, m_.layers()[l], i, st.live, DnIn{st.keep.qkv[i], st.keep.alpha[i], st.keep.beta[i], m});
    }
}

// What later tokens read of these nb positions: raw indexer keys, the pool rows of the blocks they complete, K and V
void Pass::qsa_kv(Dispatch& d, const Layer& L, int i, SeqState& st, int64_t pos, int nb) {
    const Hparams& h = h_;
    const uint32_t idim = h.idx_dim, blk = h.idx_block, hd = h.head_dim, nkv = h.n_head_kv;
    part(d, "indexer");
    mv(d, w_.mat(L.idx_k), s_.mixed, st.idx_raw[i].at((uint64_t)pos * idim * 4), nb);
    for (size_t j = 0; j < blocks_.size(); j++)
        d.run("idx_pool", {st.idx_raw[i], st.idx_pool[i], w_.vec(L.idx_k_norm), s_.cs_blk.at(j * h.rope_dims * 4)},
              PoolParams{idim, blk, (uint32_t)blocks_[j], h.rope_dims, h.rms_eps}, n1(1), n1(256));
    part(d, "attn_proj");
    const uint64_t at = (uint64_t)pos * nkv * hd * 4;
    mv(d, w_.mat(L.k), s_.mixed, st.kv_k[i].at(at), nb);
    d.run("norm_rope", {st.kv_k[i].at(at), w_.vec(L.k_norm), s_.cs_pos},
          NormRopeParams{hd, hd, h.rope_dims, nkv, nkv * hd, h.rms_eps}, n1((uint64_t)nkv * nb), n1(256));
    mv(d, w_.mat(L.v), s_.mixed, st.kv_v[i].at(at), nb);
}

// Indexer: keys and values first (qsa_kv), then per token score the blocks and pick its cells. A token reads only pool
// rows of blocks complete at its position.
void Pass::qsa(Dispatch& d, const Layer& L, int i, SeqState& st, const PassIn& in) {
    const Hparams& h = h_;
    const uint32_t idim = h.idx_dim, blk = h.idx_block, nb = in.nb;
    qsa_kv(d, L, i, st, in.pos, in.nb);
    part(d, "indexer");
    mv(d, w_.mat(L.idx_q), s_.mixed, s_.qi, nb);
    const uint32_t q_row = h.idx_heads * idim;
    d.run("norm_rope", {s_.qi, w_.vec(L.idx_q_norm), s_.cs_pos},
          NormRopeParams{idim, idim, h.rope_dims, h.idx_heads, q_row, h.rms_eps}, n1((uint64_t)h.idx_heads * nb),
          n1(256));
    if (gemm_) {
        qsa_select(d, i, st, in);
        qsa_attend(d, L, i, st, in);
        return;
    }
    for (uint32_t t = 0; t < nb; t++) {
        const QsaCells c = qsa_cells(in.pos + t, blk, h.idx_top_k);
        if (c.top_blocks == 0) continue;
        const uint32_t n_pool = (uint32_t)c.n_pool;
        d.run("idx_score", {s_.qi.at((uint64_t)t * q_row * 4), st.idx_pool[i], s_.score},
              ScoreParams{idim, h.idx_heads, n_pool}, n1(ceil_div(n_pool, 8)), n1(256));
        d.run("idx_top_r", {s_.score, s_.cells.at((uint64_t)t * cell_cap_ * 4)},
              TopParams{n_pool, (uint32_t)c.top_blocks, blk}, n1(1), n1(1024));
    }
    qsa_attend(d, L, i, st, in);
}

// Q with its gate, attention per token over its cells, output projection
void Pass::qsa_attend(Dispatch& d, const Layer& L, int i, SeqState& st, const PassIn& in) {
    const Hparams& h = h_;
    const uint32_t hd = h.head_dim, nkv = h.n_head_kv, nq = h.n_head * hd, nb = in.nb;
    part(d, "attn_proj");
    mv(d, w_.mat(L.q), s_.mixed, s_.qfull, nb);
    d.run("norm_rope", {s_.qfull, w_.vec(L.q_norm), s_.cs_pos},
          NormRopeParams{hd, 2 * hd, h.rope_dims, h.n_head, 2 * nq, h.rms_eps}, n1((uint64_t)h.n_head * nb), n1(256));
    part(d, "attend");
    if (gemm_) attend_tiles(d, i, st, in);
    for (uint32_t t = 0; t < nb && !gemm_; t++) {
        const QsaCells c = qsa_cells(in.pos + t, h.idx_block, h.idx_top_k);
        // Split over the cells: 24 threadgroups serial over 2051 cells took 26 ms of a 66 ms pass at position 2048
        // (research/measurements/2026-11-11-mac-decode-position.md)
        const AttnParams p{h.n_head, nkv, hd, (uint32_t)c.count(), 1.0f / std::sqrt((float)hd)};
        const Buf qt = s_.qfull.at((uint64_t)t * 2 * nq * 4);
        d.run("attend_part", {qt, st.kv_k[i], st.kv_v[i], s_.cells.at((uint64_t)t * cell_cap_ * 4), attn_part_}, p,
              MTLSizeMake(ceil_div(p.ncells, ATTN_PART), h.n_head, 1), n1(256));
        d.run("attend_join", {qt, attn_part_, s_.o.at((uint64_t)t * nq * 4)}, p, n1(h.n_head), n1(256));
    }
    part(d, "attn_proj");
    mv(d, w_.mat(L.o), s_.o, s_.y, nb);
}

// Prefill: the scores of all tokens as matrix products, then per token its top blocks as a bitmap (qsa_prefill.metal)
void Pass::qsa_select(Dispatch& d, int i, SeqState& st, const PassIn& in) {
    const Hparams& h = h_;
    const uint32_t nb = in.nb, blk = h.idx_block, n_pool = (uint32_t)((in.pos + nb) / blk);
    if (n_pool == 0) return;  // every cell of the chunk is in a block in progress
    const ScoreTParams sp{h.idx_dim, h.idx_heads, h.idx_heads * h.idx_dim, pool_cap_, (uint32_t)in.pos, nb, blk};
    d.run("idx_score_t", {s_.qi, st.idx_pool[i], s_.tscore}, sp, MTLSizeMake(ceil_div(n_pool, 32), ceil_div(nb, 8), 1),
          n1(128));
    const SelectParams cp{(uint32_t)in.pos, blk, h.idx_top_k / blk, pool_cap_, pool_cap_ / 32};
    d.run("idx_select", {s_.tscore, s_.bits}, cp, n1(nb), n1(1024));
}

// Prefill: attention in tiles of 8 tokens × 3 query heads of one KV head
void Pass::attend_tiles(Dispatch& d, int i, SeqState& st, const PassIn& in) {
    const Hparams& h = h_;
    const uint32_t hd = h.head_dim, nq = h.n_head * hd, group = h.n_head / h.n_head_kv;
    if (hd != 256 || 8 % h.idx_block)
        throw std::runtime_error("Metal: attend_t needs head_dim 256 and blocks dividing 8");
    AttnTParams p{h.n_head,
                  h.n_head_kv,
                  hd,
                  2 * nq,
                  nq,
                  h.idx_block,
                  (uint32_t)in.pos,
                  (uint32_t)in.nb,
                  pool_cap_ / 32,
                  0,
                  1.0f / std::sqrt((float)hd)};
    // Slices of ATTN_SLICE tokens, each a part of its own: in a prefill each gets its own command buffer
    for (uint32_t t = 0; t < (uint32_t)in.nb; t += ATTN_SLICE) {
        if (t > 0) part(d, "attend");
        p.t_base = t;
        // One query head per threadgroup (qsa_prefill.metal)
        d.run("attend_t", {s_.qfull, st.kv_k[i], st.kv_v[i], s_.bits, s_.o}, p,
              MTLSizeMake(ceil_div(std::min<uint32_t>(ATTN_SLICE, in.nb - t), 8), h.n_head_kv, group), n1(128));
    }
}

// Routed experts: one slot per (token, chosen expert); slot s of the up mat-vecs reads token s / used. The expert count
// is the router's (164 in the cut Coder, 512 in the MTP head). With the arena (arena.h) the pass stops after the router
// ("cold": the runner reads the cold experts) and computes the shared expert and the grouping meanwhile; it waits
// only before the routed experts ("cold_wait"). The same kernels on the same inputs: the same bits.
void Pass::moe(Dispatch& d, const Layer& L, int nb) {
    const Hparams& h = h_;
    const Mat router = w_.mat(L.router);
    const uint32_t used = h.n_expert_used, slots = used * nb, n_expert = router.rows;
    part(d, "router");
    mv(d, router, s_.mixed, s_.rlogits, nb);
    d.run("route", {s_.rlogits, s_.ids, s_.ew, s_.nan}, RouteParams{n_expert, used}, n1(nb), n1(1024));
    Arena* ar = w_.arena();
    const bool arena = ar && ar->holds(L.gate_exps);
    // A single token has one slot per expert: no grouping (the dispatch would cost ~20 µs per layer)
    const bool grouped = gemm_ || nb > 1;
    if (arena) {
        if (!split) throw std::logic_error("Metal: an arena layer in a pass without the cold-read split");
        cold_layer_ = ar->layer_of(L.gate_exps);
        part(d, "cold");
        const uint32_t next = cold_layer_ + 1;
        ahead_now_ = ahead_logits_.b && !gemm_ && next < h.n_layer && ar->holds(m_.layers()[next].gate_exps);
        if (ahead_now_) predict(d, next, nb), part(d, "ahead");
        shared_expert(d, L, nb);
        if (grouped) group(d, n_expert, slots);
        part(d, "cold_wait");
    } else {
        part(d, "experts");
        if (grouped) group(d, n_expert, slots);
    }
    const auto mat = [&](T t) {
        if (!arena) return w_.mat(t);
        return Weights::shape(*t);  // the table holds the addresses
    };
    const auto tab = [&](T t) { return arena ? ar->table(t) : Buf{}; };
    mv_experts(d, mat(L.gate_exps), s_.mixed, s_.g, {slots, s_.ids, h.d, used, grouped, tab(L.gate_exps)});
    mv_experts(d, mat(L.up_exps), s_.mixed, s_.u, {slots, s_.ids, h.d, used, grouped, tab(L.up_exps)});
    const uint32_t n = slots * h.expert_ff;
    d.run("swiglu", {s_.g, s_.u}, n, n1(ceil_div(n, 256)), n1(256));
    part(d, "experts_down");
    mv_experts(d, mat(L.down_exps), s_.g, s_.eout, {slots, s_.ids, h.expert_ff, 1, grouped, tab(L.down_exps)});
    if (!arena) shared_expert(d, L, nb);
    d.run("moe_sum", {s_.eout, s_.ew, s_.so, s_.gd, s_.y}, SumParams{h.d, used, (uint32_t)nb},
          n1(ceil_div((uint64_t)h.d * nb, 256)), n1(256));
}

// The next layer's router on this layer's input: which experts it will likely route to, so that the runner can read
// the cold ones while the GPU computes this layer (as Pre-gated MoE, Hwang et al. 2024)
void Pass::predict(Dispatch& d, uint32_t l, int nb) {
    const Layer& L = m_.layers()[l];
    const Mat router = w_.mat(L.router);
    // =2: the next layer's own gated read of the residual as it is now (before this layer's experts), =1: this
    // layer's read
    if (ahead_hc_.mixed.b) hc_mix_to(d, L.gr_ffn, nb, ahead_hc_);
    mv(d, router, ahead_hc_.mixed.b ? ahead_hc_.mixed : s_.mixed, ahead_logits_, nb);
    d.run("route", {ahead_logits_, ahead_ids_, ahead_ew_, ahead_nan_}, RouteParams{router.rows, ahead_k_}, n1(nb),
          n1(1024));
}

void Pass::group(Dispatch& d, uint32_t n_expert, uint32_t slots) {
    d.run("moe_group", {s_.ids, s_.goff, s_.perm, s_.glist}, GroupParams{n_expert, slots, gemm_ ? 0u : mvg_split_},
          n1(1), n1(1024));
}

void Pass::shared_expert(Dispatch& d, const Layer& L, int nb) {
    part(d, "shared_exp");
    mv(d, w_.mat(L.shexp_gate), s_.mixed, s_.sg, nb);
    mv(d, w_.mat(L.shexp_up), s_.mixed, s_.su, nb);
    const uint32_t ns = h_.shexp_ff * nb;
    d.run("swiglu", {s_.sg, s_.su}, ns, n1(ceil_div(ns, 256)), n1(256));
    mv(d, w_.mat(L.shexp_down), s_.sg, s_.so, nb);
    mv(d, w_.mat(L.shexp_gate_inp), s_.mixed, s_.gd, nb);
}

}  // namespace tona::metal
