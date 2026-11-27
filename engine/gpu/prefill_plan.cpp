// Prefill sizing: widest dense weights, the workspace layout shared by the measurement and the run, and the slots the
// layer-by-layer prefill borrows from the expert cache.
#include <algorithm>
#include <cstdint>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/alloc_trace.h"
#include "gpu/mega_extra.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

bool pf_fp8_on(const ModelBase& m) {
    const int env = pf_flags().fp8_env;
    return env >= 0 ? env != 0 : prefill_sequence(m).fp8_default();
}

const PfFlags& pf_flags() {
    // FP8 on by default (maintainer decision, 2026-11-03): prefill +2 to +4 %, no loss in the tokens after the prompt
    // or on the task benchmark (research/measurements/2026-11-03-prefill-fp8-quality.md); TONA_PREFILL_FP8=0 turns it
    // off
    static const PfFlags f{(int)tona::env::integer("TONA_PREFILL_FP8", -1),
                           tona::env::flag("TONA_PF_FULL"),
                           tona::env::text("TONA_MTP_TAIL") != nullptr,
                           tona::env::text("TONA_DN_KERNEL") && std::string(tona::env::text("TONA_DN_KERNEL")) == "old",
                           tona::env::text("TONA_MTP_PF_MEGA") != nullptr,
                           tona::env::flag("TONA_PREFILL_IDX_SCALAR"),
                           tona::env::flag("TONA_PREFILL_ATTN_SCALAR"),
                           tona::env::flag("TONA_PF_CHECK"),
                           tona::env::flag("TONA_PF_FP8_OUT"),
                           tona::env::integer("TONA_PF_WCACHE", 1) != 0};
    return f;
}

// Workspace layout: the same functions measure it and carve it

DnBufs carve_dn(Carve& p, const PfDims& D, int C) {
    DnBufs b;
    b.qkv = p.take<float>((size_t)C * D.ch);
    b.z = p.take<float>((size_t)C * D.vd);
    b.ba = p.take<float>((size_t)C * 2 * D.nv);
    b.conv = p.take<float>((size_t)C * D.ch);
    b.raw = p.take<float>((size_t)C * D.vd);
    b.o16 = p.take<_Float16>((size_t)C * D.vd);
    return b;
}

MoeBufs carve_moe(Carve& p, const PfDims& D, int C) {
    MoeBufs b;
    b.rl = p.take<float>((size_t)C * D.max_e);
    b.sel = p.take<int32_t>((size_t)C * D.ne);
    b.sw = p.take<float>((size_t)C * D.ne);
    b.rowmap = p.take<int32_t>((size_t)C * D.ne);
    b.cnt_d = p.take<int32_t>(route_count_words(D.max_e));  // counts, offsets, fill level: see prefill_check.h
    b.xg = p.take<_Float16>((size_t)C * D.ne * D.d);
    b.G = p.take<float>((size_t)C * D.ne * D.ff);
    b.U = p.take<float>((size_t)C * D.ne * D.ff);
    b.A16 = p.take<_Float16>((size_t)C * D.ne * D.ff);
    b.O = p.take<float>((size_t)C * D.ne * D.d);
    b.sg = p.take<float>((size_t)C * D.sff);
    b.su = p.take<float>((size_t)C * D.sff);
    b.sa16 = p.take<_Float16>((size_t)C * D.sff);
    b.sho = p.take<float>((size_t)C * D.d);
    b.shi = p.take<float>(C);
    return b;
}

namespace {

// Calls f(weight) for every dense weight of the prefill that exists (a descriptor an architecture does not use is
// empty); the architecture adds its own (residual) weights
template <class F>
void each_weight(const std::vector<PfLayer>& layers, const PrefillSequence& seq, const std::any& arch,
                 std::initializer_list<const PfW*> extra, F f) {
    for (const PfLayer& L : layers)
        for (const PfW* w :
             {&L.dn.qkv, &L.dn.gate, &L.dn.beta, &L.dn.alpha, &L.dn.ssm_out, &L.mamba.in_proj, &L.mamba.out_proj,
              &L.attn.q, &L.attn.k, &L.attn.v, &L.attn.o, &L.attn.idx.idx_q, &L.attn.idx.idx_k, &L.moe.router,
              &L.moe.sh_gate, &L.moe.sh_up, &L.moe.sh_down, &L.moe.sh_inp})
            if (w->w) f(*w);
    seq.each_arch_weight(arch, [&](const PfW& w) {
        if (w.w) f(w);
    });
    for (const PfW* w : extra)
        if (w->w) f(*w);
}

}  // namespace

// Widest input and most rows of the dense weights in the prefill (for the FP8 path)
int64_t Mega::pf_max_in() const {
    int64_t m = 0;
    each_weight(pf_, prefill_sequence(m_), pf_arch_, {&pf_out_, &pf_ple_key_, &pf_ple_val_},
                [&](const PfW& w) { m = std::max(m, w.in); });
    return (m + 63) / 64 * 64;
}
int64_t Mega::pf_max_rows() const {
    int64_t m = 0;
    each_weight(pf_, prefill_sequence(m_), pf_arch_, {&pf_out_, &pf_ple_key_, &pf_ple_val_},
                [&](const PfW& w) { m = std::max(m, w.rows); });
    return m;
}

// Workspace for chunks of C tokens: fixed buffers, unpacked weights, the largest phase (MoE, QSA, DeltaNet, PLE, LM
// head). The phases use the same carve functions as the run, so the layouts cannot drift apart.
size_t Mega::prefill_arena_bytes(int C) const {
    const PrefillSequence& seq = prefill_sequence(m_);
    const PfDims D = seq.dims(m_, feat_layers_.size());
    const int64_t d = D.d, ff = D.ff;
    Carve c{nullptr};
    c.take<float>((size_t)C * D.in_row);                       // e
    c.take<float>((size_t)C * D.hcd);                          // R
    c.take<float>((size_t)C * 4);                              // sc
    seq.carve_fixed(c, m_, D, C, nullptr);                     // the architecture's own buffers
    c.take<float>((size_t)C * d);                              // mixed
    c.take<float>((size_t)C * d);                              // y
    c.take<float>((size_t)C * std::max<int64_t>(D.nfeat, 1));  // feat
    c.take<_Float16>((size_t)C * d);                           // mixed16
    c.take<_Float16>((size_t)3 * EB * ff * d);                 // w16: batch of experts or slice of a dense matrix
    c.take<uint8_t>((size_t)C * pf_max_in());                  // x8
    c.take<float>(C);                                          // sx
    c.take<float>(pf_max_rows());                              // sw
    size_t ph = seq.phase_bytes(m_, D, C, sstride_);
    ph = std::max(ph, phase_area([&](Carve& p) { carve_moe(p, D, C); }));
    // LM head: logits of a slice, running sums, targets
    ph = std::max(ph, phase_area([&](Carve& p) {
                      p.take<float>((size_t)C * (3 * EB * ff * d / d)), p.take<float>(C), p.take<float>(C),
                          p.take<float>(C);
                      p.take<int32_t>(C);
                  }));
    return c.off + 256 + ph;
}

// Borrow at most TONA_PF_LOAN_PCT % of the layers (default 50), from the back
uint32_t Mega::pf_loan_layers() const {
    static const int pct = (int)std::clamp<int64_t>(tona::env::integer("TONA_PF_LOAN_PCT", 50), 1, 90);
    return (uint32_t)pct * h_.n_layer / 100;
}
size_t Mega::pf_loan_cap() const {
    const uint32_t k = pf_loan_layers();
    return k == 0 || !slots_base_ ? 0 : slots_total_ - slot_off_[h_.n_layer - k];
}

// Room for m tokens from pos in chunks of C: workspace, residual streams per chunk, features per chunk inside the
// drafter's window (from win0), two layer buffers of E experts each
size_t Mega::pf_layered_need(int m, int C, int64_t pos, int64_t win0, bool stage, bool wc) const {
    const int64_t hcd = (int64_t)h_.n_hc * h_.d, nfeat = (int64_t)feat_layers_.size() * h_.d;
    const size_t nsub = (m + C - 1) / C;
    size_t max_sb = 0;
    for (const LayerCache& LC : cache_) max_sb = std::max(max_sb, LC.slot_bytes);
    const size_t r_bytes = ((size_t)C * hcd * sizeof(float) + 255) / 256 * 256;
    const size_t f_bytes = ((size_t)C * std::max<int64_t>(nfeat, 1) * sizeof(float) + 255) / 256 * 256;
    size_t nf = 0;
    for (size_t j = 0; j < nsub; j++) nf += drafter_ && nfeat > 0 && pos + (int64_t)(j + 1) * C > win0;
    const size_t kv_stage = stage ? ((size_t)(pos + m) * kv_row_ + 255) / 256 * 256 : 0;
    return (prefill_arena_bytes(C) + 511) / 256 * 256 + nsub * r_bytes + nf * f_bytes + kv_stage +
           2 * (size_t)experts_.max_experts() * max_sb + (wc ? (pf_wcache_bytes() + 255) / 256 * 256 : 0);
}

// Unpacked dense weights of one layer for all its chunks (PrefillRun::wc_get): the weights of the architecture's layers
// and of its residual with the route each one takes, FP8 (rows x in bytes and a factor per row) or f16, every copy in
// slices like gemmq_fp8 and gemmq_f16, each slice starting at a multiple of 256 bytes. The largest layer decides;
// wc_get throws if a layer needs more, so a route changed in one place cannot go unnoticed.
size_t Mega::pf_wcache_bytes() const {
    const PfFlags& f = pf_flags();
    if (!f.wcache || pf_.empty()) return 0;
    const WcSizer w{f, pf_max_in(), pf_max_rows(), (size_t)3 * EB * h_.expert_ff * h_.d, pf_fp8_on(m_)};
    return prefill_sequence(m_).wcache_bytes(pf_, pf_arch_, h_, w);
}

size_t WcSizer::sliced(const PfW& w, bool as_fp8) const {
    if (w.rows <= 0) return 0;
    const size_t in = (size_t)w.in, rows = (size_t)w.rows;
    const size_t per = std::max<size_t>(1, (as_fp8 ? 2 * w16 : w16) / in) / 128 * 128;
    size_t sum = 0;
    for (size_t r0 = 0; r0 < rows; r0 += per) {
        const size_t rs = std::min(per, rows - r0);
        sum += ((as_fp8 ? rs * in + rs * 4 : rs * in * sizeof(_Float16)) + 255) / 256 * 256;
    }
    return sum;
}

size_t WcSizer::dense(const PfW& w) const { return sliced(w, pf_fp8_route(w, fp8, max_in, max_rows)); }

size_t WcSizer::out(const PfW& w) const { return flags.fp8_out ? dense(w) : sliced(w, false); }

size_t WcSizer::moe(const PfMoe& m) const {
    return sliced(m.router, false) + dense(m.sh_gate) + dense(m.sh_up) + dense(m.sh_down) + dense(m.sh_inp);
}

// Staging of the KV cache of the running layer in the borrowed slots (see PrefillRun::stage_wanted)
// With a cache split by position only a prompt that reaches past the split (end: its last position + 1) has rows in RAM
bool Mega::pf_stage_on(int64_t end) const {
    static const bool off = tona::env::integer("TONA_KV_STAGE", 1) == 0;
    return !off && kv_host_ && kv_row_ > 0 && (!kv_split_.planned() || end > std::min(kv_split_.hot, max_ctx_));
}

// Layered prefill possible: all experts pinned (otherwise the prefill would have to read from the NVMe)
bool Mega::pf_layered_ok() const {
    if (pin_.size() != h_.n_layer) return false;
    for (const PinLayer& P : pin_)
        for (int32_t i : P.idx)
            if (i < 0) return false;
    return true;
}

// Give the borrowed slots back: the same experts as before from the pinned RAM into their slots, restore the cache.
// Exception safe: first all caches back without throwing, then copy. If a copy fails, the layer loses its cache (all
// slots free) instead of claiming experts whose slots still hold prefill data.
void Mega::pf_return_loans() {
    if (pf_loans_.empty()) return;
    std::vector<PfLoan> loans = std::move(pf_loans_);
    pf_loans_.clear();
    if (trace::on()) trace::record("loan_return", slots_base_ + slot_off_[loans.front().layer], 0, "pf_return_loans");
    for (PfLoan& L : loans) cache_[L.layer] = std::move(L.saved);
    auto drop = [&](uint32_t l) { reset_slots(cache_[l], slots_); };
    bool ok = hipDeviceSynchronize() == hipSuccess;
    for (const PfLoan& L : loans) {
        LayerCache& C = cache_[L.layer];
        const PinLayer& P = pin_[L.layer];
        // Extra slots lie in the workspace: the loan did not touch them
        const bool lok = ok && std::all_of(C.where.begin(), C.where.end(), [&](const auto& kv) {
                             return kv.second.first >= slots_ ||
                                    hipMemcpyAsync(C.slots + (size_t)kv.second.first * C.slot_bytes,
                                                   P.host + (size_t)P.idx[kv.first] * C.slot_bytes, C.slot_bytes,
                                                   hipMemcpyHostToDevice, pf_cs_) == hipSuccess;
                         });
        if (!lok) drop(L.layer), ok = false;
    }
    if (hipStreamSynchronize(pf_cs_) != hipSuccess) {
        for (const PfLoan& L : loans) drop(L.layer);
        ok = false;
    }
    if (!ok) throw std::runtime_error("gpu::Mega: borrowed slots not copied back, cache of the layers dropped");
}

}  // namespace tona::gpu
