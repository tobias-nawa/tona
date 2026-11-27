// Bolt's prefill sequence for one block of one chunk: one plain residual stream (R += y, then the RMSNorm of the next
// consumer in one kernel), Mamba-2 (the scan runs token by token, so the state is the decode path's whatever the
// chunking), attention over the whole prefix without gate, QK norm and RoPE, and the latent MoE (shared with the
// driver's expert streaming, arch/nemotron_h/moe_prefill.cpp). Blocks are indexed 0 ... 87; the MoE blocks reach the
// expert caches under their ordinal.
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/arch/nemotron_h/prefill_step.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_fattn.h"
#include "gpu/prefill_kv_split.h"
#include "gpu/prefill_kvstore.h"
#include "gpu/prefill_mamba.h"
#include "gpu/prefill_res.h"
#include "model/nemotron_h_moe.h"

namespace tona::gpu {

MambaBufs carve_mamba(Carve& p, int C, const MambaShape& s) {
    MambaBufs b;
    b.zx = p.take<float>((size_t)C * s.in_dim);
    b.conv = p.take<float>((size_t)C * s.conv_dim);
    b.y = p.take<float>((size_t)C * s.inner);
    b.decay = p.take<float>((size_t)C * s.heads);
    b.dt = p.take<float>((size_t)C * s.heads);
    b.da = p.take<float>((size_t)C * s.heads);
    // the scores of every (chunk, group) of the chunked scan: a chunk of 64 tokens, at most one more than C / 64
    b.cb = p.take<float>((size_t)(C / 64 + 2) * s.groups * 64 * 64 * 3);
    b.o16 = p.take<_Float16>((size_t)C * s.inner);
    return b;
}

BoltAttnBufs carve_bolt_attn(Carve& p, const PfDims& D, int C) {
    BoltAttnBufs b;
    b.q = p.take<float>((size_t)C * D.nh * D.hd);
    b.tk = p.take<float>((size_t)C * D.nkv * D.hd);
    b.tv = p.take<float>((size_t)C * D.nkv * D.hd);
    b.o16 = p.take<_Float16>((size_t)C * D.nh * D.hd);
    return b;
}

namespace {

MambaShape mamba_shape(const ModelBase& m) {
    const Mamba2Dims D = model_as<NemotronHMoeModel>(m).hp().mamba2();
    return {(int)D.in_proj_dim(), (int)D.conv_dim(), (int)D.inner, (int)D.heads, (int)D.groups};
}

}  // namespace

PrefillRun::Step<NemotronHPrefillTag>::Step(PrefillRun& r)
    : r_(r),
      M(r.M),
      D(r.D),
      B(r.B),
      st(r.st),
      C(r.C),
      n(r.n),
      pos0(r.pos0),
      arena_end(r.arena_end),
      kv_now_(r.kv_now_) {}

// First block: the embedding of the chunk is the residual stream; its norm feeds the first block
void PrefillRun::Step<NemotronHPrefillTag>::embed_chunk(int j) {
    const float* e = (*r_.job_.emb)(j * C, n);
    HIPCHECK(hipMemcpyAsync(B.R, e, (size_t)n * D.d * sizeof(float), hipMemcpyHostToDevice, st));
    launch_res_norm16({B.R, nullptr, NemotronPfRes::of(M.pf_arch_).norm[0], B.mixed16, D.d, D.eps}, n, st);
}

// Mamba-2: projection, causal conv, scan, gate and group norm, out-projection. pending: decode tokens still open from
// the last pass, which the first chunk commits at the start of the scan. The NVFP4 tensor scales are applied by the
// conv, scan and norm kernels (as in the decode steps), never by the GEMMs.
void PrefillRun::Step<NemotronHPrefillTag>::mamba_block(const PfMamba2& P, int pending) {
    const MambaShape s = mamba_shape(M.m_);
    const Mamba2Dims md = model_as<NemotronHMoeModel>(M.m_).hp().mamba2();
    Carve p{B.phase, 0, arena_end};
    const MambaBufs b = carve_mamba(p, C, s);
    const int K = (int)md.conv, P_rows = s.inner / s.heads, groups = (int)md.groups;
    r_.gemmq(P.in_proj, B.mixed16, D.d, b.zx, s.in_dim, n);
    launch_mamba_dt({b.zx, P.dt_bias, P.A, b.decay, b.dt, b.da, n, s.heads, s.in_dim, s.in_dim - s.heads, P.in_scale},
                    st);
    launch_mamba_conv({b.zx, P.ring, P.conv_w, P.conv_b, b.conv, n, pos0, s.conv_dim, K, s.in_dim, s.inner, P.in_scale},
                      st);
    launch_mamba_ring({b.zx, P.ring, n, pos0, s.conv_dim, K, s.in_dim, s.inner, P.in_scale}, st);
    const bool ssd = env::integer("TONA_MAMBA_SSD", 1) != 0 && ssd_supported(s.heads, P_rows, MB_STATE, groups);
    // The sequential scan commits the open decode tokens (and is the whole scan when the chunked form is off)
    if (!ssd || pending > 0)
        launch_mamba_scan({b.conv, b.decay, b.dt, P.D, P.ssm, b.y, P.pdec, P.pxdt, P.pb, ssd ? 0 : n, pending,
                           M.parity_ ^ 1, s.heads, P_rows, groups, s.conv_dim, s.inner},
                          st);
    if (ssd)
        launch_mamba_ssd(
            {b.conv, b.da, b.dt, P.D, P.ssm, b.y, b.cb, pos0, n, 0, s.heads, P_rows, groups, s.conv_dim, s.inner}, st);
    launch_mamba_norm({b.y, b.zx, P.norm, b.o16, n, s.inner, groups, s.in_dim, D.eps, P.in_scale, P.out_scale}, st);
    r_.gemmq_out(P.out_proj, b.o16, s.inner, B.y, D.d, n);
}

// Attention: projections, K and V into the int8 cache, attention over every cached position, o projection
void PrefillRun::Step<NemotronHPrefillTag>::attention_block(const PfGatedAttn& A) {
    Carve p{B.phase, 0, arena_end};
    const BoltAttnBufs b = carve_bolt_attn(p, D, C);
    uint8_t* kv = kv_now_ ? kv_now_ : A.kv;
    const KvCut cut = kv_now_ || !A.kv_cold ? KvCut{} : KvCut{A.kv_cold, M.kv_split_.hot};
    const int d = D.d, nh = D.nh, nkv = D.nkv, hd = D.hd;
    r_.share_x8(true);
    r_.gemmq(A.q, B.mixed16, d, b.q, nh * hd, n);
    r_.gemmq(A.k, B.mixed16, d, b.tk, nkv * hd, n);
    r_.gemmq(A.v, B.mixed16, d, b.tv, nkv * hd, n);
    r_.share_x8(false);
    launch_kv_store_cut({b.tk, b.tv, kv, M.kv_row_, pos0, n, nkv}, cut, st);
    const FattnPrefillArgs a{
        b.q, kv, M.kv_row_, nullptr, b.o16, pos0, n, nh, nkv, hd, 1.0f / std::sqrt((float)hd), cut.cold, cut.hot};
    if (!launch_fattn_prefill(a, st)) throw std::runtime_error("gpu::Mega: no prefill attention kernel for this shape");
    r_.gemmq_out(A.o, b.o16, nh * hd, B.y, d, n);
}

// The block, then the residual add and the norm of the next block's input (the last block only adds: the output norm
// is the head's, head_input)
void PrefillRun::Step<NemotronHPrefillTag>::body(uint32_t l, int j) {
    const PfLayer& F = M.pf_[l];
    const NemotronPfRes& res = NemotronPfRes::of(M.pf_arch_);
    M.beat("prefill", (int)l);
    if (l == 0) embed_chunk(j);
    if (res.ordinal[l] >= 0)
        r_.moe((uint32_t)res.ordinal[l], j);
    else if (F.attn.kv)
        attention_block(F.attn);
    else
        mamba_block(F.mamba, j == 0 ? M.pending_commit_ : 0);  // the first chunk commits the open decode tokens
    const bool last = l + 1 == res.norm.size();
    launch_res_norm16({B.R, B.y, last ? nullptr : res.norm[l + 1], last ? nullptr : B.mixed16, D.d, D.eps}, n, st);
}

void PrefillRun::Step<NemotronHPrefillTag>::resume(uint32_t l) {
    launch_res_norm16({B.R, nullptr, NemotronPfRes::of(M.pf_arch_).norm[l], B.mixed16, D.d, D.eps}, n, st);
}

void PrefillRun::Step<NemotronHPrefillTag>::head_input() {
    launch_res_norm16({B.R, nullptr, NemotronPfRes::of(M.pf_arch_).head_norm, B.mixed16, D.d, D.eps}, n, st);
}

namespace {

class NemotronHPrefill final : public PrefillSequence {
public:
    PfDims dims(const ModelBase& m, size_t n_feat_layers) const override;
    // Off: FP8 on 2- and 4-bit weights is the Ornith finding (research/measurements/2026-11-19-ornith-prefill.md); not
    // measured for Bolt, so the exact f16 route
    bool fp8_default() const override { return false; }
    size_t phase_bytes(const ModelBase& m, const PfDims& D, int C, int64_t) const override;
    void carve_fixed(Carve&, const ModelBase&, const PfDims&, int, std::any*) const override {}
    void retarget_state(std::any&, const std::function<void(float*&)>&) const override {}
    void dump_mtp(const std::any&, const PfDumpSink&) const override {}
    size_t wcache_bytes(const std::vector<PfLayer>& layers, const std::any&, const MoeShape& h,
                        const WcSizer& w) const override;
    void each_arch_weight(const std::any&, const std::function<void(const PfW&)>&) const override {}
    void dump_layer(const std::any&, size_t, const PfDumpSink&) const override {}
    void dump_head(const std::any&, const PfDumpSink&) const override {}
    void layer(PrefillRun& run, uint32_t l, int j) const override {
        PrefillRun::Step<NemotronHPrefillTag>(run).body(l, j);
    }
    void end_chunk(PrefillRun&) const override {}  // every residual add is made in its block
    void resume(PrefillRun& run, uint32_t l) const override { PrefillRun::Step<NemotronHPrefillTag>(run).resume(l); }
    void head_input(PrefillRun& run, int) const override { PrefillRun::Step<NemotronHPrefillTag>(run).head_input(); }
    bool mtp_entries(PrefillRun&) const override { return false; }  // no MTP head
};

PfDims NemotronHPrefill::dims(const ModelBase& m, size_t n_feat_layers) const {
    const NemotronHMoeModel& bolt = model_as<NemotronHMoeModel>(m);
    const NemotronHparams& h = bolt.hp();
    const MoeShape sh = bolt.moe_shape();
    PfDims D{};
    D.d = (int)h.d, D.ff = (int)sh.expert_ff, D.sff = (int)h.shexp_ff, D.ne = (int)sh.n_expert_used;
    D.max_e = expert_cap(h.n_expert_max);
    D.hcd = D.d;
    D.nh = (int)h.n_head, D.hd = (int)h.head_dim;
    for (const NemotronLayer& L : bolt.layers())
        if (L.kind == BoltBlockKind::Attention) D.nkv = (int)L.n_kv;
    D.eps = h.rms_eps;
    D.nfeat = (int)n_feat_layers * D.d;
    D.in_row = D.d;
    return D;
}

size_t NemotronHPrefill::phase_bytes(const ModelBase& m, const PfDims& D, int C, int64_t) const {
    const MambaShape s = mamba_shape(m);
    return std::max(phase_area([&](Carve& p) { carve_mamba(p, C, s); }),
                    phase_area([&](Carve& p) { carve_bolt_attn(p, D, C); }));
}

size_t NemotronHPrefill::wcache_bytes(const std::vector<PfLayer>& layers, const std::any&, const MoeShape&,
                                      const WcSizer& w) const {
    size_t most = 0;
    for (uint32_t l = 2; l < layers.size(); l++) {
        const PfLayer& F = layers[l];
        const size_t b = w.moe(F.moe) + w.dense(F.moe.lat_down) + w.out(F.moe.lat_up) + w.dense(F.mamba.in_proj) +
                         w.out(F.mamba.out_proj) + w.dense(F.attn.q) + w.dense(F.attn.k) + w.dense(F.attn.v) +
                         w.out(F.attn.o);
        most = std::max(most, b);
    }
    return most;
}

}  // namespace

NemotronPfRes& NemotronPfRes::reset(std::any& store, size_t n_block) {
    NemotronPfRes& r = store.emplace<NemotronPfRes>();
    r.norm.assign(n_block, nullptr);
    r.ordinal.assign(n_block, -1);
    return r;
}

const NemotronPfRes& NemotronPfRes::of(const std::any& store) { return std::any_cast<const NemotronPfRes&>(store); }

const PrefillSequence& nemotron_h_prefill_sequence() {
    static const NemotronHPrefill seq;
    return seq;
}

}  // namespace tona::gpu
