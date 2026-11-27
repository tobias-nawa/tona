// The Coder's prefill sequence for one layer of one chunk: embedding, features for the drafter, PLE, and the two kinds
// of attention layers (quantized self-attention and DeltaNet). The MoE part is shared (prefill_moe_run.cpp).
#include <algorithm>
#include <cmath>

#include "gpu/arch/qwen4exp/prefill_step.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_extra.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

// The state of the running chunk is copied from the driver; calls into the driver go through r_
PrefillRun::Step<Qwen4ExpPrefillTag>::Step(PrefillRun& r)
    : r_(r),
      pend_inj_(run_of(r).pend_inj),
      X(run_of(r).fx),
      hc_((int)model_as<Qwen4ExpModel>(r.M.m_).hp().n_hc),
      rank_((int)model_as<Qwen4ExpModel>(r.M.m_).hp().hc_rank),
      M(r.M),
      D(r.D),
      B(r.B),
      st(r.st),
      C(r.C),
      n(r.n),
      pos0(r.pos0),
      flags_(r.flags_),
      arena_end(r.arena_end),
      kv_now_(r.kv_now_) {}

// First layer: embedding and PLE vector of the chunk to the GPU, residual streams = embedding
void PrefillRun::Step<Qwen4ExpPrefillTag>::embed_chunk(int j) {
    const float* e = (*r_.job_.emb)(j * C, n);
    HIPCHECK(hipMemcpyAsync(B.e, e, (size_t)n * 2 * D.d * sizeof(float), hipMemcpyHostToDevice, st));
    launch_bcast({B.e, B.R, n, hc_, D.d}, st);
}

// Mean over the streams at the layers whose output the drafter reads. k_hc_mean and the PLE read R: the open combine
// goes first.
void PrefillRun::Step<Qwen4ExpPrefillTag>::feature_taps(uint32_t l) {
    const auto& fl = M.feat_layers_;
    if (l == model_as<Qwen4ExpModel>(M.m_).hp().ple_layer || std::find(fl.begin(), fl.end(), (int64_t)l) != fl.end())
        flush();
    for (size_t k = 0; l > 0 && B.feat && k < fl.size(); k++)
        if (fl[k] == (int64_t)l) launch_hc_mean({B.R, B.feat, n, hc_, D.d, D.nfeat, (int)k * D.d}, st);
}

// Per-layer embedding: key and value from the PLE vector, gating of the streams, dilated causal conv
void PrefillRun::Step<Qwen4ExpPrefillTag>::ple_step() {
    Carve p{B.phase, 0, arena_end};
    const PleBufs b = carve_ple(p, D, C);
    launch_to16({B.e + D.d, 2 * D.d, b.e16, D.d, n, D.d}, st);
    r_.gemmq_f16(M.pf_ple_key_, b.e16, D.d, b.pk, (int)D.hcd, n);
    r_.gemmq_f16(M.pf_ple_val_, b.e16, D.d, b.pv, D.d, n);
    launch_ple_gate({B.R, b.pk, b.pv, M.pf_ple_nk_, M.pf_ple_nq_, M.pf_ple_nc_, b.pg, b.pn, hc_, D.d, D.eps}, n, st);
    const int K = (int)model_as<Qwen4ExpModel>(M.m_).hp().ple_conv, dil = (int)M.h_.ple_ngram,
              RG = (K - 1) * dil + MEGA_BMAX;
    launch_ple_conv({B.R, b.pg, b.pn, M.pf_ple_conv_, M.pf_ple_hist_, n, pos0, (int)D.hcd, K, dil}, st);
    launch_ring_store({b.pn, M.pf_ple_hist_, n, pos0, (int)D.hcd, RG}, st);
}

// Selection of the pool blocks, only for tokens that see more blocks than top_k holds
void PrefillRun::Step<Qwen4ExpPrefillTag>::select_blocks(const PfLayer& F, const QsaBufs& b) {
    const int64_t first_sel = (int64_t)D.topk + D.blk - 1;  // first position with a selection: (pos+1)/blk > topk/blk
    const bool scalar = flags_.idx_scalar || D.ih != 4 || D.idim != 128;
    for (int t0 = (int)std::clamp<int64_t>(first_sel - pos0, 0, n); t0 < n; t0 += SCORE_TOK) {
        const int cnt = std::min(SCORE_TOK, n - t0);
        const int64_t per_pool = (pos0 + t0 + cnt) / D.blk;
        if (scalar)
            launch_idx_score(
                {b.iq, F.attn.idx.pool, b.score, t0, cnt, pos0, D.ih, D.idim, D.blk, D.topk, per_pool, M.sstride_}, st);
        else
            launch_idx_score_wmma({b.iq, F.attn.idx.pool, b.score, t0, cnt, pos0, D.blk, D.topk, per_pool, M.sstride_},
                                  st);
        launch_topk({b.score, b.cells, b.ncells, t0, pos0, D.blk, D.topk, M.sstride_}, cnt, st);
    }
}

// Quantized self-attention layer: projections, KV cache and indexer, block selection, attention, o projection.
// Returns false for the last layer without check: only KV, pool and indexer ring are written there.
bool PrefillRun::Step<Qwen4ExpPrefillTag>::qsa_layer(const PfLayer& F, bool tail) {
    Carve p{B.phase, 0, arena_end};
    uint8_t* kv = kv_now_ ? kv_now_ : F.attn.kv;
    const QsaBufs b = carve_qsa(p, D, C, M.sstride_);
    const int d = D.d, nh = D.nh, nkv = D.nkv, hd = D.hd;
    r_.share_x8(true);
    r_.gemmq(F.attn.idx.idx_k, B.mixed16, d, b.rk, D.idim, n);
    if (!tail) r_.gemmq(F.attn.q, B.mixed16, d, b.qfull, 2 * nh * hd, n);
    r_.gemmq(F.attn.k, B.mixed16, d, b.tk, nkv * hd, n);
    r_.gemmq(F.attn.v, B.mixed16, d, b.tv, nkv * hd, n);
    r_.share_x8(false);
    if (!tail) r_.gemmq_f16(F.attn.idx.idx_q, B.mixed16, d, b.iq, D.ih * D.idim, n);
    launch_qsa_prep({b.qfull,
                     b.qn,
                     b.g,
                     F.attn.q_norm,
                     b.tk,
                     F.attn.k_norm,
                     b.tv,
                     kv,
                     M.kv_row_,
                     b.rk,
                     F.attn.idx.raw_ring,
                     F.attn.idx.pool,
                     F.attn.idx.idx_k_norm,
                     b.iq,
                     F.attn.idx.idx_q_norm,
                     n,
                     pos0,
                     nh,
                     nkv,
                     hd,
                     D.idim,
                     D.ih,
                     D.blk,
                     D.eps,
                     D.base},
                    st);
    launch_ring_store({b.rk, F.attn.idx.raw_ring, n, pos0, D.idim, MEGA_RAW_RING}, st);
    if (tail) return false;  // KV, pool and indexer ring stand, the rest is never read
    select_blocks(F, b);
    const float scale = 1.0f / std::sqrt((float)hd);
    if (hd == 256 && nh / nkv <= 16 && !flags_.attn_scalar)
        launch_attn_wmma({b.qn, kv, M.kv_row_, b.cells, b.ncells, b.g, b.o16, pos0, nh, nkv, D.blk, D.topk, scale,
                          M.kv_host() && !kv_now_},
                         n, st);
    else
        launch_attn({b.qn, kv, M.kv_row_, b.cells, b.ncells, b.g, b.o16, pos0, nh, nkv, hd, D.blk, D.topk, scale}, n,
                    st);
    r_.gemmq_out(F.attn.o, b.o16, nh * hd, B.y, d, n);
    return true;
}

// DeltaNet layer: projections, causal conv, delta rule, output norm. pending: decode tokens still open from the last
// pass, which the first chunk commits.
void PrefillRun::Step<Qwen4ExpPrefillTag>::dn_layer(const PfLayer& F, int pending) {
    Carve p{B.phase, 0, arena_end};
    const DnBufs b = carve_dn(p, D, C);
    const int d = D.d, nv = D.nv, ch = D.ch;
    r_.share_x8(true);
    r_.gemmq(F.dn.qkv, B.mixed16, d, b.qkv, ch, n);
    r_.gemmq(F.dn.gate, B.mixed16, d, b.z, D.vd, n);
    r_.gemmq(F.dn.beta, B.mixed16, d, b.ba, 2 * nv, n);
    r_.gemmq(F.dn.alpha, B.mixed16, d, b.ba + nv, 2 * nv, n);
    r_.share_x8(false);
    launch_dn_conv_l2({b.qkv, F.dn.ring, F.dn.conv_w, b.conv, n, pos0, ch,
                       (int)model_as<Qwen4ExpModel>(M.m_).hp().dn_conv, 2 * D.nk, D.eps},
                      st);
    launch_ring_store({b.qkv, F.dn.ring, n, pos0, ch, (int)model_as<Qwen4ExpModel>(M.m_).hp().dn_conv - 1 + MEGA_BMAX},
                      st);
    const int parity = M.parity_ ^ 1;
    const DnRecurArgs rec{b.conv,  b.ba,   F.dn.dt_bias, F.dn.A, F.dn.state, F.dn.pend_k, F.dn.pend_d, F.dn.pend_a,
                          pending, parity, b.raw,        n,      D.nk,       nv,          D.kd,        ch};
    if (flags_.dn_old) {
        launch_dn_recur(rec, st);
    } else {
        if (pending > 0) {
            DnRecurArgs commit = rec;
            commit.n = 0;
            launch_dn_recur(commit, st);
        }
        launch_dn_wave({b.conv, b.ba, F.dn.dt_bias, F.dn.A, F.dn.state, b.raw, n, D.nk, nv, D.kd, ch}, st);
    }
    launch_dn_norm({b.raw, F.dn.ssm_norm, b.z, b.o16, n, nv, D.eps}, st);
    r_.gemmq_out(F.dn.ssm_out, b.o16, D.vd, B.y, d, n);
}

// All layers: gated residual around attention (or DeltaNet) and MoE
void PrefillRun::Step<Qwen4ExpPrefillTag>::body(uint32_t l, int j) {
    const PfLayer& F = M.pf_[l];
    const Qwen4ExpPfRes& res = Qwen4ExpPfRes::of(M.pf_arch_);
    const bool tail = r_.is_tail(l);
    const int pending = j == 0 ? M.pending_commit_ : 0;  // the first chunk commits the open decode tokens
    M.beat("prefill", (int)l);
    if (l == 0) embed_chunk(j);
    feature_taps(l);
    if (l == model_as<Qwen4ExpModel>(M.m_).hp().ple_layer) ple_step();
    gated_residual(res.layers[l].attn, X.inj_a);
    if (M.h_.is_qsa(l)) {
        if (!qsa_layer(F, tail)) return;
    } else {
        dn_layer(F, pending);
    }
    combine(X.inj_a);
    gated_residual(res.layers[l].ffn, X.inj_f);
    r_.moe(l, j);
    combine(X.inj_f);
}

QsaBufs carve_qsa(Carve& p, const PfDims& D, int C, int64_t sstride) {
    QsaBufs b;
    b.rk = p.take<float>((size_t)C * D.idim);
    b.qfull = p.take<float>((size_t)C * 2 * D.nh * D.hd);
    b.tk = p.take<float>((size_t)C * D.nkv * D.hd);
    b.tv = p.take<float>((size_t)C * D.nkv * D.hd);
    b.iq = p.take<float>((size_t)C * D.ih * D.idim);
    b.qn = p.take<float>((size_t)C * D.nh * D.hd);
    b.g = p.take<float>((size_t)C * D.nh * D.hd);
    b.o16 = p.take<_Float16>((size_t)C * D.nh * D.hd);
    b.score = p.take<float>((size_t)SCORE_TOK * sstride);
    b.cells = p.take<int32_t>((size_t)C * MEGA_MAX_CELLS);
    b.ncells = p.take<int32_t>(C);
    return b;
}

PleBufs carve_ple(Carve& p, const PfDims& D, int C) {
    PleBufs b;
    b.e16 = p.take<_Float16>((size_t)C * D.d);
    b.pk = p.take<float>((size_t)C * D.hcd);
    b.pv = p.take<float>((size_t)C * D.d);
    b.pg = p.take<float>((size_t)C * D.hcd);
    b.pn = p.take<float>((size_t)C * D.hcd);
    return b;
}

namespace {

class Qwen4ExpPrefill final : public PrefillSequence {
public:
    PfDims dims(const ModelBase& m, size_t n_feat_layers) const override;
    bool fp8_default() const override {
        return true;
    }  // maintainer decision 2026-11-03, 2026-11-03-prefill-fp8-quality.md
    size_t phase_bytes(const ModelBase&, const PfDims& D, int C, int64_t sstride) const override;
    void carve_fixed(Carve& c, const ModelBase& m, const PfDims& D, int C, std::any* run) const override;
    void retarget_state(std::any& arch, const std::function<void(float*&)>& tr) const override {
        tr(std::any_cast<Qwen4ExpPfRes&>(arch).mtp.ring);
    }
    void dump_mtp(const std::any& arch, const PfDumpSink& out) const override;
    size_t wcache_bytes(const std::vector<PfLayer>& layers, const std::any& arch, const MoeShape& h,
                        const WcSizer& w) const override;
    void each_arch_weight(const std::any& arch, const std::function<void(const PfW&)>& f) const override;
    void dump_layer(const std::any& arch, size_t l, const PfDumpSink& out) const override;
    void dump_head(const std::any& arch, const PfDumpSink& out) const override;
    void layer(PrefillRun& run, uint32_t l, int j) const override {
        PrefillRun::Step<Qwen4ExpPrefillTag>(run).body(l, j);
    }
    void end_chunk(PrefillRun& run) const override { PrefillRun::Step<Qwen4ExpPrefillTag>(run).end_chunk(); }
    void head_input(PrefillRun& run, int j) const override { PrefillRun::Step<Qwen4ExpPrefillTag>(run).head_input(j); }
    bool mtp_entries(PrefillRun& run) const override { return PrefillRun::Step<Qwen4ExpPrefillTag>(run).mtp_entries(); }
};

PfDims Qwen4ExpPrefill::dims(const ModelBase& m, size_t n_feat_layers) const {
    const Hparams& h = model_as<Qwen4ExpModel>(m).hp();
    PfDims D{};
    D.d = (int)h.d, D.ff = (int)h.expert_ff, D.sff = (int)h.shexp_ff;
    D.ne = (int)h.n_expert_used;
    D.max_e = expert_cap(h.n_expert);
    D.hcd = (int64_t)h.n_hc * D.d;
    D.kd = (int)h.dn_k_dim(), D.vd = (int)h.dn_v_dim(), D.ch = 2 * D.kd + D.vd;
    D.nk = (int)h.dn_k_heads, D.nv = (int)h.dn_v_heads;
    D.nh = (int)h.n_head, D.nkv = (int)h.n_head_kv, D.hd = (int)h.head_dim, D.idim = (int)h.idx_dim;
    D.ih = (int)h.idx_heads, D.blk = (int)h.idx_block, D.topk = (int)h.idx_top_k;
    D.eps = h.rms_eps, D.base = h.rope_base;
    D.nfeat = (int)n_feat_layers * D.d;
    D.dn_conv = (int)h.dn_conv;
    D.in_row = 2 * D.d;
    return D;
}

// DeltaNet, quantized self-attention and PLE: the largest of their phases
size_t Qwen4ExpPrefill::phase_bytes(const ModelBase&, const PfDims& D, int C, int64_t sstride) const {
    size_t ph = phase_area([&](Carve& p) { carve_dn(p, D, C); });
    ph = std::max(ph, phase_area([&](Carve& p) { carve_qsa(p, D, C, sstride); }));
    return std::max(ph, phase_area([&](Carve& p) { carve_ple(p, D, C); }));
}

// The weights of a gated residual with the route they take: down and inject as a pair (one unpacked matrix) when it
// fits, up with the mix in the epilogue
size_t gated_res_bytes(const WcSizer& w, const PfGR& g, bool mix_fused) {
    const bool pair = !Qwen4ExpPfFlags::get().pair_off && g.down.in == g.inject.in &&
                      (g.down.rows + g.inject.rows) * g.down.in <= (int64_t)w.w16;
    const auto up = [](size_t b) { return (b + 255) / 256 * 256; };
    const size_t b = pair ? up((size_t)((g.down.rows + g.inject.rows) * g.down.in) * sizeof(_Float16))
                          : w.dense(g.down) + w.dense(g.inject);
    return b + (mix_fused ? up((size_t)(g.up.rows * g.up.in) * sizeof(_Float16)) : w.dense(g.up));
}

// The weights of the sequence (layer) and of gated_residual, each with the route it takes
size_t Qwen4ExpPrefill::wcache_bytes(const std::vector<PfLayer>& layers, const std::any& arch, const MoeShape& h,
                                     const WcSizer& w) const {
    const Qwen4ExpPfRes& res = Qwen4ExpPfRes::of(arch);
    const bool mix = qwen4exp_mix_fused(h.n_hc, h.d);
    size_t most = 0;
    for (uint32_t l = 2; l < h.n_layer; l++) {
        const PfLayer& F = layers[l];
        size_t b =
            gated_res_bytes(w, res.layers[l].attn, mix) + gated_res_bytes(w, res.layers[l].ffn, mix) + w.moe(F.moe);
        if (h.is_qsa(l))
            b += w.dense(F.attn.idx.idx_k) + w.dense(F.attn.q) + w.dense(F.attn.k) + w.dense(F.attn.v) +
                 w.sliced(F.attn.idx.idx_q, false) + w.out(F.attn.o);
        else
            b +=
                w.dense(F.dn.qkv) + w.dense(F.dn.gate) + w.dense(F.dn.beta) + w.dense(F.dn.alpha) + w.out(F.dn.ssm_out);
        most = std::max(most, b);
    }
    return most;
}

void Qwen4ExpPrefill::each_arch_weight(const std::any& arch, const std::function<void(const PfW&)>& f) const {
    const Qwen4ExpPfRes& res = Qwen4ExpPfRes::of(arch);
    for (const Qwen4ExpPfRes::Layer& L : res.layers)
        for (const PfGR* g : {&L.attn, &L.ffn})
            for (const PfW* w : {&g->down, &g->up, &g->inject}) f(*w);
    for (const PfW* w : {&res.head.down, &res.head.up, &res.head.inject}) f(*w);
}

// The program dump keeps the order of the records it had when they were part of PfLayer
void Qwen4ExpPrefill::carve_fixed(Carve& c, const ModelBase& m, const PfDims&, int C, std::any* run) const {
    qwen4exp_carve_fixed(c, model_as<Qwen4ExpModel>(m).hp(), C, run);
}

void Qwen4ExpPrefill::dump_mtp(const std::any& arch, const PfDumpSink& out) const {
    qwen4exp_dump_mtp(Qwen4ExpPfRes::of(arch), out);
}

void Qwen4ExpPrefill::dump_layer(const std::any& arch, size_t l, const PfDumpSink& out) const {
    const Qwen4ExpPfRes::Layer& L = Qwen4ExpPfRes::of(arch).layers[l];
    for (const PfGR* gr : {&L.attn, &L.ffn}) {
        out.weight("gr.down", gr->down), out.weight("gr.up", gr->up), out.weight("gr.inject", gr->inject);
        out.pointer("gr.norm", gr->norm);
    }
}

void Qwen4ExpPrefill::dump_head(const std::any& arch, const PfDumpSink& out) const {
    const PfGR& gr = Qwen4ExpPfRes::of(arch).head;
    out.weight("head.down", gr.down), out.weight("head.up", gr.up), out.weight("head.inject", gr.inject);
}

}  // namespace

const PrefillSequence& qwen4exp_prefill_sequence() {
    static const Qwen4ExpPrefill seq;
    return seq;
}

}  // namespace tona::gpu
