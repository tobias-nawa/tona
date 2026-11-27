// Ornith's prefill sequence for one layer of one chunk: one plain residual stream (R += y, then the RMSNorm of the next
// consumer in one kernel), gated DeltaNet with a silu readout gate, full-context gated attention (preparation as the
// quantized attention's without the indexer, then the attention over the whole prefix on WMMA). The MoE is shared
// (prefill_moe_run.cpp). The chunked gated delta rule is the Coder's launch_dn_wave.
#include <algorithm>
#include <climits>
#include <cmath>
#include <stdexcept>

#include "gpu/arch/qwen35moe/prefill_step.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_extra.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_fattn.h"
#include "gpu/prefill_kv_split.h"
#include "gpu/prefill_res.h"
#include "model/qwen35moe.h"

namespace tona::gpu {

FattnBufs carve_fattn(Carve& p, const PfDims& D, int C) {
    FattnBufs b;
    b.qfull = p.take<float>((size_t)C * 2 * D.nh * D.hd);
    b.qn = p.take<float>((size_t)C * D.nh * D.hd);
    b.g = p.take<float>((size_t)C * D.nh * D.hd);
    b.tk = p.take<float>((size_t)C * D.nkv * D.hd);
    b.tv = p.take<float>((size_t)C * D.nkv * D.hd);
    b.o16 = p.take<_Float16>((size_t)C * D.nh * D.hd);
    return b;
}

PrefillRun::Step<Qwen35MoePrefillTag>::Step(PrefillRun& r)
    : r_(r),
      flags_(r.flags_),
      M(r.M),
      D(r.D),
      B(r.B),
      st(r.st),
      C(r.C),
      n(r.n),
      pos0(r.pos0),
      arena_end(r.arena_end),
      kv_now_(r.kv_now_) {}

// First layer: the embedding of the chunk is the residual stream; its norm feeds the first mixer
void PrefillRun::Step<Qwen35MoePrefillTag>::embed_chunk(int j) {
    const float* e = (*r_.job_.emb)(j * C, n);
    HIPCHECK(hipMemcpyAsync(B.R, e, (size_t)n * D.d * sizeof(float), hipMemcpyHostToDevice, st));
    launch_res_norm16({B.R, nullptr, Qwen35PfRes::of(M.pf_arch_).norm_attn[0], B.mixed16, D.d, D.eps}, n, st);
}

// Gated DeltaNet layer: projections, causal conv, delta rule, output norm with silu(z). pending: decode tokens still
// open from the last pass, which the first chunk commits.
void PrefillRun::Step<Qwen35MoePrefillTag>::dn_layer(const PfLayer& F, int pending) {
    Carve p{B.phase, 0, arena_end};
    const DnBufs b = carve_dn(p, D, C);
    const PfDeltaNet& N = F.dn;
    const int d = D.d, nv = D.nv, ch = D.ch;
    r_.share_x8(true);
    r_.gemmq(N.qkv, B.mixed16, d, b.qkv, ch, n);
    r_.gemmq(N.gate, B.mixed16, d, b.z, D.vd, n);
    r_.gemmq(N.beta, B.mixed16, d, b.ba, 2 * nv, n);
    r_.gemmq(N.alpha, B.mixed16, d, b.ba + nv, 2 * nv, n);
    r_.share_x8(false);
    launch_dn_conv_l2({b.qkv, N.ring, N.conv_w, b.conv, n, pos0, ch, D.dn_conv, 2 * D.nk, D.eps}, st);
    launch_ring_store({b.qkv, N.ring, n, pos0, ch, D.dn_conv - 1 + MEGA_BMAX}, st);
    if (pending > 0) {
        const DnRecurArgs commit{b.conv,  b.ba,          N.dt_bias, N.A, N.state, N.pend_k, N.pend_d, N.pend_a,
                                 pending, M.parity_ ^ 1, b.raw,     0,   D.nk,    nv,       D.kd,     ch};
        launch_dn_recur(commit, st);
    }
    launch_dn_wave({b.conv, b.ba, N.dt_bias, N.A, N.state, b.raw, n, D.nk, nv, D.kd, ch}, st);
    launch_dn_norm_silu({b.raw, N.ssm_norm, b.z, b.o16, n, nv, D.eps}, st);
    r_.gemmq_out(N.ssm_out, b.o16, D.vd, B.y, d, n);
}

// Full-context gated attention layer: projections, K and V into the int8 cache, attention over every cached position,
// o projection. Returns false for the last layer without check: only K and V are written there.
bool PrefillRun::Step<Qwen35MoePrefillTag>::attn_layer(const PfLayer& F, bool tail) {
    Carve p{B.phase, 0, arena_end};
    const FattnBufs b = carve_fattn(p, D, C);
    const PfGatedAttn& A = F.attn;
    uint8_t* kv = kv_now_ ? kv_now_ : A.kv;
    const KvCut cut = kv_now_ || !A.kv_cold ? KvCut{} : KvCut{A.kv_cold, M.kv_split_.hot};
    const int d = D.d, nh = D.nh, nkv = D.nkv, hd = D.hd;
    r_.share_x8(true);
    if (!tail) r_.gemmq(A.q, B.mixed16, d, b.qfull, 2 * nh * hd, n);
    r_.gemmq(A.k, B.mixed16, d, b.tk, nkv * hd, n);
    r_.gemmq(A.v, B.mixed16, d, b.tv, nkv * hd, n);
    r_.share_x8(false);
    // The quantized attention's preparation without its indexer: no pool unit (block size beyond any position), no
    // indexer heads
    launch_qsa_prep_cut(
        {b.qfull, b.qn,    b.g, A.q_norm, b.tk, A.k_norm, b.tv, kv, M.kv_row_, nullptr, nullptr, nullptr, nullptr,
         nullptr, nullptr, n,   pos0,     nh,   nkv,      hd,   0,  0,         INT_MAX, D.eps,   D.base},
        cut, st);
    if (tail) return false;
    const FattnPrefillArgs a{b.qn,     kv,     M.kv_row_, b.g, b.o16, pos0, n, nh, nkv, hd, 1.0f / std::sqrt((float)hd),
                             cut.cold, cut.hot};
    if (!launch_fattn_prefill(a, st)) throw std::runtime_error("gpu::Mega: no prefill attention kernel for this shape");
    r_.gemmq_out(A.o, b.o16, nh * hd, B.y, d, n);
    return true;
}

// Mixer, residual and the norm of the MoE, MoE, residual and the norm of the next layer's mixer
void PrefillRun::Step<Qwen35MoePrefillTag>::body(uint32_t l, int j) {
    const PfLayer& F = M.pf_[l];
    const Qwen35PfRes& res = Qwen35PfRes::of(M.pf_arch_);
    const bool tail = r_.is_tail(l);
    M.beat("prefill", (int)l);
    if (l == 0) embed_chunk(j);
    if (M.h_.is_qsa(l)) {
        if (!attn_layer(F, tail)) return;
    } else {
        dn_layer(F, j == 0 ? M.pending_commit_ : 0);  // the first chunk commits the open decode tokens
    }
    launch_res_norm16({B.R, B.y, res.norm_ffn[l], B.mixed16, D.d, D.eps}, n, st);
    r_.moe(l, j);
    const bool last = l + 1 == M.h_.n_layer;
    // The last layer only adds: the output norm is the head's (head_input)
    launch_res_norm16({B.R, B.y, last ? nullptr : res.norm_attn[l + 1], last ? nullptr : B.mixed16, D.d, D.eps}, n, st);
}

void PrefillRun::Step<Qwen35MoePrefillTag>::resume(uint32_t l) {
    launch_res_norm16({B.R, nullptr, Qwen35PfRes::of(M.pf_arch_).norm_attn[l], B.mixed16, D.d, D.eps}, n, st);
}

void PrefillRun::Step<Qwen35MoePrefillTag>::head_input() {
    launch_res_norm16({B.R, nullptr, Qwen35PfRes::of(M.pf_arch_).head_norm, B.mixed16, D.d, D.eps}, n, st);
}

namespace {

class Qwen35MoePrefill final : public PrefillSequence {
public:
    PfDims dims(const ModelBase& m, size_t n_feat_layers) const override;
    // Off: FP8 on top of 4-bit weights doubled the distance to the oracle (mean |d nll| 0.216 against 0.123 with f16 on
    // 117 tokens, research/measurements/2026-11-19-ornith-prefill.md)
    bool fp8_default() const override { return false; }
    // From 640 tokens layer by layer: the next layer's experts are copied ahead, where chunk by chunk waits for each
    // batch on PCIe. Prefill against chunk by chunk: 300 tokens -12.8 %, 617 -0.2 %, 959 +12.9 %, 1933 +15.3 %
    // (research/measurements/2026-11-20-short-layered.md)
    int short_layered_min() const override { return 640; }
    size_t phase_bytes(const ModelBase&, const PfDims& D, int C, int64_t) const override;
    void carve_fixed(Carve&, const ModelBase&, const PfDims&, int, std::any*) const override {}
    void retarget_state(std::any&, const std::function<void(float*&)>&) const override {}
    void dump_mtp(const std::any&, const PfDumpSink&) const override {}
    size_t wcache_bytes(const std::vector<PfLayer>& layers, const std::any&, const MoeShape& h,
                        const WcSizer& w) const override;
    void each_arch_weight(const std::any&, const std::function<void(const PfW&)>&) const override {}
    void dump_layer(const std::any&, size_t, const PfDumpSink&) const override {}
    void dump_head(const std::any&, const PfDumpSink&) const override {}
    void layer(PrefillRun& run, uint32_t l, int j) const override {
        PrefillRun::Step<Qwen35MoePrefillTag>(run).body(l, j);
    }
    void end_chunk(PrefillRun&) const override {}  // every residual add is made in its layer
    void resume(PrefillRun& run, uint32_t l) const override { PrefillRun::Step<Qwen35MoePrefillTag>(run).resume(l); }
    void head_input(PrefillRun& run, int) const override { PrefillRun::Step<Qwen35MoePrefillTag>(run).head_input(); }
    bool mtp_entries(PrefillRun& run) const override {
        return PrefillRun::Step<Qwen35MoePrefillTag>(run).mtp_entries();
    }
};

PfDims Qwen35MoePrefill::dims(const ModelBase& m, size_t n_feat_layers) const {
    const Qwen35Hparams& h = model_as<Qwen35MoeModel>(m).hp();
    PfDims D{};
    D.d = (int)h.d, D.ff = (int)h.expert_ff, D.sff = (int)h.shexp_ff;
    D.ne = (int)h.n_expert_used;
    D.max_e = expert_cap(h.n_expert);
    D.hcd = D.d;
    D.kd = (int)(h.dn_k_heads * h.dn_head_dim), D.vd = (int)(h.dn_v_heads * h.dn_head_dim), D.ch = 2 * D.kd + D.vd;
    D.nk = (int)h.dn_k_heads, D.nv = (int)h.dn_v_heads;
    D.nh = (int)h.n_head, D.nkv = (int)h.n_head_kv, D.hd = (int)h.head_dim;
    D.idim = 0, D.ih = 0, D.blk = 0, D.topk = 0;
    D.eps = h.rms_eps, D.base = h.rope_base;
    D.nfeat = (int)n_feat_layers * D.d;
    D.dn_conv = (int)h.dn_conv;
    D.in_row = D.d;
    return D;
}

size_t Qwen35MoePrefill::phase_bytes(const ModelBase&, const PfDims& D, int C, int64_t) const {
    return std::max(phase_area([&](Carve& p) { carve_dn(p, D, C); }),
                    phase_area([&](Carve& p) { carve_fattn(p, D, C); }));
}

size_t Qwen35MoePrefill::wcache_bytes(const std::vector<PfLayer>& layers, const std::any&, const MoeShape& h,
                                      const WcSizer& w) const {
    size_t most = 0;
    for (uint32_t l = 2; l < h.n_layer; l++) {
        const PfLayer& F = layers[l];
        size_t b = w.moe(F.moe);
        if (h.is_qsa(l))
            b += w.dense(F.attn.q) + w.dense(F.attn.k) + w.dense(F.attn.v) + w.out(F.attn.o);
        else
            b +=
                w.dense(F.dn.qkv) + w.dense(F.dn.gate) + w.dense(F.dn.beta) + w.dense(F.dn.alpha) + w.out(F.dn.ssm_out);
        most = std::max(most, b);
    }
    return most;
}

}  // namespace

Qwen35PfRes& Qwen35PfRes::reset(std::any& store, size_t n_layer) {
    Qwen35PfRes& r = store.emplace<Qwen35PfRes>();
    r.norm_attn.assign(n_layer, nullptr);
    r.norm_ffn.assign(n_layer, nullptr);
    return r;
}

const Qwen35PfRes& Qwen35PfRes::of(const std::any& store) { return std::any_cast<const Qwen35PfRes&>(store); }

const PrefillSequence& qwen35moe_prefill_sequence() {
    static const Qwen35MoePrefill seq;
    return seq;
}

}  // namespace tona::gpu
