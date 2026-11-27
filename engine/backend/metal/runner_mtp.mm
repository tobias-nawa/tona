// The MTP head of a slot (engine/gpu/mega_mtp.cpp, same semantics): entry p takes the embedding of token p and the
// target's h at p - 1 and proposes token p + 1. fill catches the entries up to pos - 1 (KV only) and, with next,
// computes the entry at pos; step chains one entry behind the last with the head's own h. The target's h is known only
// for the rows of its last pass and the saved row at hlast_pos; other entries are computed with h = 0 and counted as
// stale.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "backend/metal/runner.h"
#include "model/quant.h"

namespace tona::metal {

const float* Runner::trunk_h(const Slot& s, int64_t q, bool& stale) const {
    const uint64_t hcd = (uint64_t)m_.hp().n_hc * m_.hp().d;
    const backend::HLoc at = s.mtp.h_of(q, {fw_.s == &s, fw_.pos, fw_.n});
    stale = at.src == backend::HSource::Unknown;
    if (at.src == backend::HSource::Pass) return pass_.residual().f32() + (uint64_t)at.row * hcd;
    return at.src == backend::HSource::Saved ? s.hlast.data() : nullptr;
}

// One MTP pass of in.nb entries from in.pos; hs: the h of each (nullptr: zeros). Without kv_only the truncated head of
// the last entry.
void Runner::mtp_pass(Slot& s, const int32_t* toks, const float* const* hs, const MtpPassIn& in) {
    const double prepared = GpuGaps::enabled() ? GpuGaps::now() : 0;
    const Hparams& h = m_.hp();
    const int nb = in.nb;
    const int64_t pos = in.pos;
    const uint64_t d = h.d, hcd = (uint64_t)h.n_hc * d;
    MtpBufs& mb = pass_.mtp_bufs();
    const TensorInfo& et = *m_.tok_embd();
    const uint64_t erb = Weights::shape(et).row_bytes;
    for (int b = 0; b < nb; b++) {
        if (toks[b] < 0 || (uint32_t)toks[b] >= h.vocab) throw std::runtime_error("Metal: MTP token outside the vocab");
        float* e = mb.e.f32() + b * hcd;
        dequantize_row(et.type, et.data + (uint64_t)toks[b] * erb, e, (int64_t)d);
        for (uint32_t g = 1; g < h.n_hc; g++) std::memcpy(e + g * d, e, d * sizeof(float));
        float* hin = mb.hin.f32() + b * hcd;
        if (hs[b])
            std::memcpy(hin, hs[b], hcd * sizeof(float));
        else
            std::memset(hin, 0, hcd * sizeof(float));
    }
    if (GpuGaps::enabled()) parts_.mtp = GpuGaps::now() - prepared;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [dev_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        pass_.mtp(enc, s.st, in);
        [enc endEncoding];
        finish_timed(cb, prepared, "MTP pass");
    }
    note_nans("MTP router logits", pos);
    mcount_.passes++;
    if (in.kv_only) return;
    if (route_log_) log_routes(nb);
    s.mtp.pos = pos, s.mtp.nb = nb;
    const float* last = mb.R.f32() + (uint64_t)(nb - 1) * hcd;
    s.hout.assign(last, last + hcd);
}

// TONA_MTP_ROUTE_STATS: one line per entry of the pass, its experts and their weights ("id:weight ..."), for the cut
// of the head to the experts the coding load uses (tools/coder_requant slice-mtp, tools/mtp_route_stats.py)
void Runner::log_routes(int nb) {
    const uint32_t used = m_.hp().n_expert_used;
    for (int b = 0; b < nb; b++) {
        for (uint32_t k = 0; k < used; k++)
            std::fprintf(route_log_, "%s%d:%.6g", k ? " " : "", pass_.route_ids()[b * used + k],
                         pass_.route_weights()[b * used + k]);
        std::fputc('\n', route_log_);
    }
    std::fflush(route_log_);
}

// The head's pick: from the GPU's reduction or the host's scan, both with the semantics of host_pick
backend::MtpPick Runner::mtp_pick() const {
    if (!gpu_pick_) return host_pick();
    const float* o = pass_.mtp_pick_out();
    uint32_t row = 0;
    std::memcpy(&row, o, sizeof row);
    const backend::MtpPick g =
        row < mvocab_->rows() ? backend::MtpPick{mvocab_->token(row), o[2] > 0 ? 1.0 / o[2] : 1.0} : backend::MtpPick{};
    if (!pick_check_) return g;
    const backend::MtpPick h = host_pick();
    checked_.picks++, checked_.mismatches += g.token != h.token;
    uint64_t bits[2] = {(uint64_t)(uint32_t)g.token, 0};
    std::memcpy(&bits[1], &g.q, sizeof g.q);
    for (const uint64_t b : bits) checked_.hash = (checked_.hash ^ b) * 1099511628211ULL;
    checked_.rel.push_back(std::fabs(g.q - h.q) / std::max(h.q, 1e-300));
    if (checked_.picks % 200 == 0) {
        std::vector<double>& r = checked_.rel;
        std::sort(r.begin(), r.end());
        std::fprintf(stderr,
                     "MTP pick check: %lld picks, %lld token mismatches, q relative deviation median %.2e max %.2e, "
                     "hash %016llx\n",
                     (long long)checked_.picks, (long long)checked_.mismatches, r[r.size() / 2], r.back(),
                     (unsigned long long)checked_.hash);
        r.clear();
    }
    return g;
}

// Argmax of the head's rows (NaN as -FLT_MAX, ties to the smaller row, free rows skipped) and its probability
backend::MtpPick Runner::host_pick() const {
    const float* lg = pass_.mtp_logits();
    const MtpVocab& v = *mvocab_;
    size_t best = v.rows();
    float mx = -FLT_MAX;
    for (size_t r = 0; r < v.rows(); r++) {
        const float l = std::isnan(lg[r]) ? -FLT_MAX : std::clamp(lg[r], -FLT_MAX, FLT_MAX);
        if (v.token(r) >= 0 && (best == v.rows() || l > mx)) best = r, mx = l;
    }
    if (best == v.rows()) return {};
    double sum = 0;
    for (size_t r = 0; r < v.rows(); r++)
        if (v.token(r) >= 0 && !std::isnan(lg[r])) sum += std::exp((double)lg[r] - mx);
    return {v.token(best), sum > 0 ? 1.0 / sum : 1.0};
}

// Decode passes over the entries from min(valid, end_pos) to end_pos (+ 1 with next >= 0); the one at s.pos takes
// next, the last one runs the head if next >= 0
void Runner::mtp_catch_up(Slot& s, int64_t end_pos, int32_t next) {
    int32_t toks[BMAX];
    const float* hs[BMAX];
    s.mtp.catch_up(
        {s.tokens, end_pos, next}, {BMAX, toks, hs},
        [&](int64_t q) {
            bool stale = false;
            const float* h = trunk_h(s, q, stale);
            mcount_.stale += stale;
            return h;
        },
        [&](int64_t p, int nb, bool last) { mtp_pass(s, toks, hs, MtpPassIn{nb, p, !last, false}); });
}

// After a prefill chunk of nb tokens: their entries per GEMM, each with the target's h of the row before (the first
// with h0, the target's h before the chunk); entries left out before the chunk by decode passes first
void Runner::mtp_prefill(Slot& s, const int32_t* tok, int nb, const float* h0) {
    const uint64_t hcd = (uint64_t)m_.hp().n_hc * m_.hp().d;
    const int64_t pos0 = s.pos - nb;
    mtp_catch_up(s, pos0, -1);
    const float* R = pass_.residual().f32();
    std::vector<const float*> hs(pass_.mtp_part());
    for (int t0 = 0; t0 < nb; t0 += pass_.mtp_part()) {
        const int n = std::min(pass_.mtp_part(), nb - t0);
        for (int i = 0; i < n; i++) hs[i] = t0 + i == 0 ? h0 : R + (uint64_t)(t0 + i - 1) * hcd;
        mtp_pass(s, tok + t0, hs.data(), MtpPassIn{n, pos0 + t0, true, true});
    }
    const float* last = R + (uint64_t)(nb - 1) * hcd;
    s.hlast.assign(last, last + hcd);
    s.mtp.after_prefill(s.pos);
    s.mtp.nb = 0;
}

backend::MtpPick Runner::mtp_fill(int32_t next) {
    const GpuGaps::Scope scope(gaps_, GpuGaps::kFill);
    if (!has_mtp()) throw std::runtime_error("Metal backend: no MTP head");
    check_idle("mtp_fill");
    Slot& s = cur();
    mvocab_->sync(s.tokens, next, s.mtp.scan);
    // the next target pass starts with next, then the drafts: their rows come in while the head drafts
    s.ahead.clear();
    if (next >= 0) s.ahead.push_back(next), prefetch_after(s, s.ahead.data(), 0, 1);
    const int64_t end = s.pos + (next >= 0);
    if (end > max_ctx_) throw std::runtime_error("Metal: context full");
    mtp_catch_up(s, s.pos, next);
    // The target's h at pos - 1 for the next fill: after a rollback of the sequence the start would be missing
    bool stale = false;
    const float* hl = trunk_h(s, s.pos - 1, stale);
    if (hl && hl != s.hlast.data()) s.hlast.assign(hl, hl + (size_t)m_.hp().n_hc * m_.hp().d);
    s.mtp.after_fill(s.pos, -1, hl != nullptr);
    return next >= 0 ? mtp_pick() : backend::MtpPick{};
}

backend::MtpPick Runner::mtp_step(int32_t tok) {
    const GpuGaps::Scope scope(gaps_, GpuGaps::kStep);
    if (!has_mtp()) throw std::runtime_error("Metal backend: no MTP head");
    check_idle("mtp_step");
    Slot& s = cur();
    if (s.mtp.nb == 0) throw std::logic_error("Metal backend: mtp_step without a fill");
    const int64_t pos = s.mtp.next_pos();
    if (pos >= max_ctx_) throw std::runtime_error("Metal: context full");
    if (!s.ahead.empty()) {
        s.ahead.push_back(tok);
        prefetch_after(s, s.ahead.data(), (int)s.ahead.size() - 1, (int)s.ahead.size());
    }
    const float* h = s.hout.data();  // the reference loads no rows for a step's token either
    mtp_pass(s, &tok, &h, MtpPassIn{1, pos, false, false});
    return mtp_pick();
}

}  // namespace tona::metal
