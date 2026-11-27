// The MTP head at run time: entries for the tokens of the context, drafting steps for one sequence or several, and
// the truncated head with its rows loaded on demand.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <string>
#include <string_view>

#include "backend/env.h"
#include "backend/host_trace.h"
#include "backend/nan_guard.h"
#include "gpu/evict_counter.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega.h"
#include "gpu/mega_kernels.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "model/progress.h"
#include "model/quant_iq4.h"

namespace tona::gpu {

namespace {

// NaN logits of the MTP head are clamped on the GPU (op_route, op_mtp_pick); the counts tell the CPU that something
// before the head is wrong. flags: the NaN count of each route wave; pick_nan < 0: no pick ran in this pass.
void report_mtp_nan(const float* flags, int pick_nan, int64_t pos) {
    int route = 0;
    for (int w = 0; w < MEGA_BMAX; w++) route += ((const int*)flags)[w];
    const std::string where = std::format("position {}", pos);
    if (route > 0) tona::nan::report("MTP router logits", route, where);
    if (pick_nan > 0) tona::nan::report("MTP head logits", pick_nan, where);
}

// NaN blocks of the pick in a pass that ran it (-1: the pass without the pick step, the field is old)
int pick_nan_of(const float* pick, bool ran) { return ran ? ((const int*)pick)[3] : -1; }

// A row the pick returned that is no row of the head (every logit NaN and nothing else valid): row 0
int32_t valid_row(int32_t r, size_t rows) {
    if (r >= 0 && (size_t)r < rows) return r;
    tona::nan::report("MTP pick outside the head", 1, std::format("row {} of {}", r, rows));
    return 0;
}

}  // namespace

// Row t of the LM head in the format of the truncated head
void Mega::head_row(int32_t t, uint8_t* dst) const {
    const TensorInfo& ot = *m_.lm_head();
    const int64_t d = h_.d, srb = row_bytes_of(ot.type, d);
    // pread, not the mapping: the mapping of the output tensor is removed after the upload
    std::vector<uint8_t> src(srb);
    m_.gguf().read(ot, (uint64_t)t * srb, src.data(), srb);
    if (!mtp_head_q4_) {
        std::memcpy(dst, src.data(), srb);
        return;
    }
    std::vector<float> f(d);
    dequantize_row(ot.type, src.data(), f.data(), d);
    iq4::quantize_iq4_nl(f.data(), dst, d);
}

// Tokens of the context (and next) without a row in the truncated head get one of the dynamic rows, round robin:
// rows r0 ... of the truncated head from the full LM head in VRAM
void Mega::mtp_sync_vocab(int32_t next) {
    if (!mtp_head_) return;
    const size_t nstat = mtp_vocab_.n_static(), dyn = mtp_vocab_.n_ctx();
    const auto [first, n] = mtp_vocab_.sync(tokens_, next, mtrk_.scan, [&](size_t ring, int32_t old, int32_t t) {
        if (old >= 0) mtp_dyn_.note(old, 0);
        mtp_dyn_.note(t, 1);
        if (mtp_head_q4_) head_row(t, mtp_rows_host_ + ring * mtp_rb_);
    });
    mtp_dyn_.flush();
    // at most two contiguous pieces (wrap-around of the ring)
    for (size_t done = 0; done < n;) {
        const size_t s = (first + done) % dyn, m = std::min(n - done, dyn - s);
        HIPCHECK(hipMemcpyAsync(mtp_map_dev_ + nstat + s, mtp_vocab_.data() + nstat + s, m * sizeof(int32_t),
                                hipMemcpyHostToDevice, nullptr));
        // Rows in the type of the target from its LM head in VRAM: from the mapping of the GGUF they sometimes came
        // from disk after a long prompt (prefill of 13K tokens: 1594 instead of ~1750 tok/s)
        if (mtp_head_q4_)
            HIPCHECK(hipMemcpyAsync(mtp_head_ + (nstat + s) * mtp_rb_, mtp_rows_host_ + s * mtp_rb_, m * mtp_rb_,
                                    hipMemcpyHostToDevice, nullptr));
        else
            launch_mtp_rows(static_cast<const uint8_t*>(head_w_), mtp_head_, mtp_map_dev_, (int64_t)(nstat + s),
                            (int)mtp_rb_, (unsigned)m);
        done += m;
    }
    stats_.mtp_dyn += (int64_t)n;
}

// h of the target at position q for the MTP entry q + 1 (device pointer; nullptr: 0 before position 0; mtp_bad_:
// unknown)
const float* Mega::mtp_h_of(int64_t q) const {
    const int64_t hcd = (int64_t)h_.n_hc * h_.d;
    const backend::HLoc at = mtrk_.h_of(q, {trunk_ok_, fw_pos_, fw_m_});
    switch (at.src) {
        case backend::HSource::None:
            return nullptr;
        case backend::HSource::Pass:
            return h_trunk_ + (fw_row0_ + at.row) * hcd;
        case backend::HSource::Saved:
            return mtp_hlast_;
        case backend::HSource::Unknown:
            break;
    }
    return mtp_bad_;
}

// Embeddings of the tokens to the device and h per token: in one piece if the rows follow each other, otherwise row
// by row into mtp_hin_. Returns the device pointer of h.
const float* Mega::mtp_stage_inputs(const int32_t* toks, int nb, const float* const* hs) {
    const int64_t d = h_.d, hcd = (int64_t)h_.n_hc * d;
    const TensorInfo& et = *m_.embedding_table();
    const int64_t erb = row_bytes_of(et.type, d);
    for (int b = 0; b < nb; b++) {
        if (toks[b] < 0 || toks[b] >= (int32_t)h_.vocab) throw std::runtime_error("gpu::Mega: mtp token outside");
        dequantize_row(et.type, et.data + (int64_t)toks[b] * erb, mtp_in_host_ + b * d, d);
    }
    HIPCHECK(hipMemcpyAsync(me_, mtp_in_host_, (size_t)nb * d * sizeof(float), hipMemcpyHostToDevice, nullptr));
    bool run = hs[0] != nullptr;
    for (int b = 1; b < nb && run; b++) run = hs[b] == hs[0] + b * hcd;
    if (run) return hs[0];
    for (int b = 0; b < nb; b++) {
        float* dst = mtp_hin_ + b * hcd;
        if (hs[b])
            HIPCHECK(hipMemcpyAsync(dst, hs[b], hcd * sizeof(float), hipMemcpyDeviceToDevice, nullptr));
        else
            HIPCHECK(hipMemsetAsync(dst, 0, hcd * sizeof(float), nullptr));
    }
    return mtp_hin_;
}

// Several slots: save the h of the other slots in mtp_hout_ before this pass overwrites it (not for "KV only", which
// does not write mtp_hout_)
void Mega::mtp_save_other_h() {
    const int64_t hcd = (int64_t)h_.n_hc * h_.d, hrows = (int64_t)MEGA_BMAX * hcd;
    for (int s = 0; s < n_seqs_; s++) {
        if (s == cur_slot_ || mtp_hsrc_[s] < mtp_hout_ || mtp_hsrc_[s] >= mtp_hout_ + hrows) continue;
        float* dst = mtp_hsave_ + s * hcd;
        HIPCHECK(hipMemcpyAsync(dst, mtp_hsrc_[s], hcd * sizeof(float), hipMemcpyDeviceToDevice, nullptr));
        mtp_hsrc_[s] = dst;
    }
}

// TONA_MTP_PROF=1: time per step of the program over the passes with one token, every 1000 to stderr
void Mega::mtp_profile_step() {
    std::vector<unsigned long long> t(2 * mops_.size());
    HIPCHECK(hipMemcpy(t.data(), d_mprof_, t.size() * sizeof(unsigned long long), hipMemcpyDeviceToHost));
    const double us = 1e3 / wall_khz_;
    unsigned long long prev = t[0];
    for (size_t o = 0; o < mops_.size(); o++) mprof_us_[o] += (double)(t[2 * o + 1] - prev) * us, prev = t[2 * o + 1];
    if (++mprof_n_ % 1000 != 0) return;
    double tot = 0;
    for (double v : mprof_us_) tot += v;
    std::fprintf(stderr, "MTP profile over %lld steps: %.1f us per step\n", (long long)mprof_n_, tot / mprof_n_);
    for (size_t o = 0; o < mops_.size(); o++)
        std::fprintf(stderr, "  %2zu %-14s %7.1f us\n", o, mlabels_[o].c_str(), mprof_us_[o] / mprof_n_);
}

void Mega::ensure_profile_buffer() {
    if (d_mprof_) return;
    HIPCHECK(trace::dev_malloc(&d_mprof_, 2 * mops_.size() * sizeof(unsigned long long)));
    mprof_us_.assign(mops_.size(), 0.0);
}

// Pick (and with mtp_keep_logits the logits) of the last row of a pass of nb tokens to the host
void Mega::fetch_pick(int nb) {
    if (keep_mlog_)
        HIPCHECK(hipMemcpyAsync(mlog_host_ + (size_t)cur_slot_ * h_.vocab, mlog_ + (int64_t)(nb - 1) * h_.vocab,
                                mtp_rows() * sizeof(float), hipMemcpyDeviceToHost, nullptr));
}

// One pass of the MTP head: token b at pos0 + b, h per token from hs[b] (nullptr: 0); want: logits of the last row
void Mega::mtp_pass(const int32_t* toks, int nb, int64_t pos0, const float* const* hs, bool want, bool kv_only) {
    if (nb < 1 || nb > MEGA_BMAX || pos0 + nb > max_ctx_) throw std::runtime_error("gpu::Mega: mtp pass outside");
    backend::HostTrace::mark("mtp_begin", nb);
    const float* h = mtp_stage_inputs(toks, nb, hs);
    mparams_host_->pos = pos0, mparams_host_->nb = nb, mparams_host_->mtp_h = h, mparams_host_->nseq = 0;
    if (n_seqs_ > 1 && !kv_only) mtp_save_other_h();
    static const bool prof = env::text("TONA_MTP_PROF") != nullptr;
    if (prof) ensure_profile_buffer();
    const bool prof_step = prof && nb == 1;
    if (kv_only)
        launch_program({nb, (int)mkops_.size(), d_mkops_, d_mktasks_, d_mparams_, nullptr}, 0, mparams_host_);
    else
        launch_program({nb, (int)mops_.size(), d_mops_, d_mtasks_, d_mparams_,
                        prof_step ? static_cast<unsigned long long*>(d_mprof_) : nullptr},
                       0, mparams_host_);
    backend::HostTrace::mark("mtp_launched");
    // The pick (3 floats), its NaN count and the NaN counts of the route: every pass copies them, the route runs in all
    HIPCHECK(hipMemcpyAsync(mpick_host_, mpick_, (4 + MEGA_BMAX) * sizeof(float), hipMemcpyDeviceToHost, nullptr));
    if (want) fetch_pick(nb);
    const bool sampling = want && keep_mlog_;
    mtp_dyn_.queue_tokens(mtp_map_dev_, sampling);
    beat("mtp-end", -1);
    evict_sync(mbox_host_);
    mtp_dyn_.adopt_tokens(mtp_vocab_.tokens(), sampling);
    backend::HostTrace::mark("mtp_synced");
    report_mtp_nan(mpick_host_ + 4, pick_nan_of(mpick_host_, want), pos0);
    HIPCHECK(hipGetLastError());
    check_aborts();
    const int64_t hcd = (int64_t)h_.n_hc * h_.d;
    if (!kv_only) mtrk_.pos = pos0, mtrk_.nb = nb, mtp_hsrc_[cur_slot_] = mtp_hout_ + (int64_t)(nb - 1) * hcd;
    stats_.mtp_passes++;
    if (prof_step && !kv_only) mtp_profile_step();
}

// Catch the MTP entries up to pos - 1 (h from the last forward or the saved h at pos - 1) and, with next >= 0,
// compute the entry at pos: logits [vocab] for pos + 1 (first draft), otherwise nullptr
Mega::MtpPick Mega::mtp_fill(int32_t next) {
    if (!mtp_on_) throw std::runtime_error("gpu::Mega: no MTP head");
    if (awaiting_commit_) throw std::runtime_error("gpu::Mega: mtp_fill before commit");
    mtp_sync_vocab(next);
    const float* hs[MEGA_BMAX];
    int32_t toks[MEGA_BMAX];
    mtrk_.catch_up(
        {tokens_, pos_, next}, {MEGA_BMAX, toks, hs},
        [&](int64_t q) -> const float* {
            const float* h = mtp_h_of(q);
            if (h != mtp_bad_) return h;
            stats_.mtp_stale++;
            return nullptr;
        },
        [&](int64_t p, int nb, bool want) { mtp_pass(toks, nb, p, hs, want, !want); });
    // Save h at pos - 1 (state): after a change of the sequence the start of the next entry would be missing. When h
    // is unknown there (another slot ran a pass or a prefill since) the saved row stays as it is and keeps its own
    // position: claiming pos - 1 for it would make the next fill read a stale row instead of counting it as stale.
    const float* hl = mtp_h_of(pos_ - 1);
    if (hl && hl != mtp_bad_ && hl != mtp_hlast_)
        HIPCHECK(
            hipMemcpyAsync(mtp_hlast_, hl, (size_t)h_.n_hc * h_.d * sizeof(float), hipMemcpyDeviceToDevice, nullptr));
    mtrk_.after_fill(pos_, next, hl != mtp_bad_);
    return next >= 0 ? pick() : MtpPick{};
}

void Mega::mtp_keep_logits(bool on) {
    if (on && !mlog_host_ && mtp_on_) {
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&mlog_host_), (size_t)MEGA_SMAX * h_.vocab * sizeof(float),
                                    hipHostMallocDefault));
        pin_allocs_.push_back(mlog_host_);
    }
    if (on && mtp_dyn_.on && !mtp_dyn_.tok_pin) {
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&mtp_dyn_.tok_pin), mtp_dyn_.rows * sizeof(int32_t),
                                    hipHostMallocDefault));
        pin_allocs_.push_back(mtp_dyn_.tok_pin);
    }
    keep_mlog_ = on && mlog_host_;
}

Mega::MtpPick Mega::pick() const {
    int32_t r;
    std::memcpy(&r, &mpick_host_[0], 4);
    // Dynamic rows: the step has translated the row into the token already
    r = valid_row(r, mtp_dyn_.on ? h_.vocab : mtp_rows());
    return {mtp_vocab_.empty() || mtp_dyn_.on ? r : mtp_vocab_[r], mpick_host_[2] > 0 ? 1.0 / mpick_host_[2] : 1.0};
}

Mega::MtpPick Mega::mtp_step(int32_t tok) {
    if (!mtp_on_ || awaiting_commit_) throw std::runtime_error("gpu::Mega: mtp_step without MTP or before commit");
    const float* h = n_seqs_ > 1 ? mtp_hsrc_[cur_slot_] : mtp_hout_ + (int64_t)(mtrk_.nb - 1) * h_.n_hc * h_.d;
    mtp_pass(&tok, 1, mtrk_.pos + mtrk_.nb, &h, true);
    return pick();
}

// Per slot: checks, the embedding of its token and its h into the shared buffers, rows and positions into the
// parameters
void Mega::mtp_stage_multi(const int* slots, const int32_t* toks, int k, int64_t* pos) {
    const int64_t d = h_.d, hcd = (int64_t)h_.n_hc * d;
    const TensorInfo& et = *m_.embedding_table();
    const int64_t erb = row_bytes_of(et.type, d);
    MegaParams& P = *mparams_host_;
    std::memset(P.snb, 0, sizeof P.snb);
    for (int i = 0; i < k; i++) {
        for (int j = 0; j < i; j++)
            if (slots[j] == slots[i]) throw std::runtime_error("gpu::Mega: slot twice in the MTP step");
        use_slot(slots[i]);
        if (awaiting_commit_) throw std::runtime_error("gpu::Mega: mtp_step_multi before commit");
        if (toks[i] < 0 || toks[i] >= (int32_t)h_.vocab) throw std::runtime_error("gpu::Mega: mtp token outside");
        pos[i] = mtrk_.pos + mtrk_.nb;
        if (pos[i] + 1 > max_ctx_) throw std::runtime_error("gpu::Mega: mtp pass outside");
        dequantize_row(et.type, et.data + (int64_t)toks[i] * erb, mtp_in_host_ + i * d, d);
        HIPCHECK(hipMemcpyAsync(mtp_hin_ + i * hcd, mtp_hsrc_[slots[i]], hcd * sizeof(float), hipMemcpyDeviceToDevice,
                                nullptr));
        const int s = slots[i];
        P.sb0[s] = i, P.snb[s] = 1, P.spos[s] = pos[i], P.scommit[s] = 0, P.sparity[s] = 0;
    }
}

// TONA_MTP_MULTI_CHECK=1 for one slot: compute it once more alone (writes the same KV entries) and compare
void Mega::mtp_check_slot(int slot, int32_t tok, int64_t pos, const MtpPick& got, const float* hin, const float* hout) {
    const int64_t hcd = (int64_t)h_.n_hc * h_.d;
    use_slot(slot);
    float* hs = mtp_hsave_ + slot * hcd;
    HIPCHECK(hipMemcpy(hs, hin, hcd * sizeof(float), hipMemcpyHostToDevice));
    const float* h = hs;
    mtp_pass(&tok, 1, pos, &h, true);
    const MtpPick a = pick();
    std::vector<float> h1(hcd);
    HIPCHECK(hipMemcpy(h1.data(), mtp_hout_, hcd * sizeof(float), hipMemcpyDeviceToHost));
    // The GPU sums the experts in the order of the union of the pass: rounding differs, token and q are almost always
    // the same (the target checks drafts, the output does not depend on it)
    float dh = 0;
    for (int64_t j = 0; j < hcd; j++) dh = std::max(dh, std::fabs(h1[j] - hout[j]));
    static double dq_max = 0;
    static int64_t n_chk = 0;
    dq_max = std::max(dq_max, std::fabs(a.q - got.q)), n_chk++;
    stats_.mtp_multi_diff += a.token != got.token;
    if (a.token != got.token || n_chk % 200 == 0)
        std::fprintf(stderr,
                     "MTP together/alone: %lld steps, %lld different tokens (slot %d: %d/%d, q %.4f/%.4f, |dh| %.2e), "
                     "largest |dq| so far %.2e\n",
                     (long long)n_chk, (long long)stats_.mtp_multi_diff, slot, got.token, a.token, got.q, a.q, dh,
                     dq_max);
}

void Mega::mtp_multi_check(const int* slots, const int32_t* toks, int k, const int64_t* pos, const MtpPick* out) {
    const int64_t hcd = (int64_t)h_.n_hc * h_.d;
    std::vector<float> hout((size_t)k * hcd), hin((size_t)k * hcd);
    HIPCHECK(hipMemcpy(hout.data(), mtp_hout_, hout.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIPCHECK(hipMemcpy(hin.data(), mtp_hin_, hin.size() * sizeof(float), hipMemcpyDeviceToHost));
    for (int i = 0; i < k; i++)
        mtp_check_slot(slots[i], toks[i], pos[i], out[i], hin.data() + i * hcd, hout.data() + i * hcd);
    HIPCHECK(hipMemcpy(const_cast<float*>(mtp_hout_), hout.data(), hout.size() * sizeof(float), hipMemcpyHostToDevice));
}

// MTP step of several sequences in one pass: slot slots[i] gets toks[i] after its last MTP entry, like mtp_step per
// slot (bit-identical); out[i] as its return value
void Mega::mtp_step_multi(const int* slots, const int32_t* toks, int k, MtpPick* out) {
    if (!mtp_on_ || !d_xmops_) throw std::runtime_error("gpu::Mega: mtp_step_multi without MTP or with one slot");
    if (k < 1 || k > n_seqs_) throw std::runtime_error(std::format("gpu::Mega: {} sequences in the MTP step", k));
    if (k == 1) {
        use_slot(slots[0]);
        out[0] = mtp_step(toks[0]);
        return;
    }
    const int64_t hcd = (int64_t)h_.n_hc * h_.d;
    int64_t pos[MEGA_SMAX];
    mtp_stage_multi(slots, toks, k, pos);
    mtp_run_multi(slots, k, pos);
    mparams_host_->nseq = 0;
    stats_.mtp_passes++, stats_.mtp_multi++;
    for (int i = 0; i < k; i++) out[i] = multi_pick(slots[i]);
    static const bool check = env::flag("TONA_MTP_MULTI_CHECK");
    if (check) mtp_multi_check(slots, toks, k, pos, out);
    for (int i = 0; i < k; i++) {
        use_slot(slots[i]);
        mtrk_.pos = pos[i], mtrk_.nb = 1;
        mtp_hsrc_[slots[i]] = mtp_hout_ + i * hcd;
    }
}

// Embeddings and patched programs to the device, launch, picks (and logits) back
void Mega::mtp_run_multi(const int* slots, int k, const int64_t* pos) {
    const int64_t d = h_.d;
    MegaParams& P = *mparams_host_;
    HIPCHECK(hipMemcpyAsync(me_, mtp_in_host_, (size_t)k * d * sizeof(float), hipMemcpyHostToDevice, nullptr));
    for (const XPatch& x : xmpatch_) xmops_host_[x.op].p[x.j] = x.base + (int64_t)P.sb0[x.seq] * x.row;
    HIPCHECK(hipMemcpyAsync(d_xmops_, xmops_host_, xmops_.size() * sizeof(MegaOp), hipMemcpyHostToDevice, nullptr));
    P.pos = pos[0], P.nb = k, P.nseq = k, P.mtp_h = mtp_hin_, P.commit = 0, P.parity = 0;
    launch_program({k, (int)xmops_.size(), d_xmops_, d_mtasks_, d_mparams_, nullptr}, 0, mparams_host_);
    HIPCHECK(hipMemcpyAsync(mpickx_host_, mpick_x_, (MEGA_SMAX * 4 + MEGA_BMAX) * sizeof(float), hipMemcpyDeviceToHost,
                            nullptr));
    if (keep_mlog_)
        for (int i = 0; i < k; i++)
            HIPCHECK(hipMemcpyAsync(mlog_host_ + (size_t)slots[i] * h_.vocab, mlog_ + (int64_t)i * h_.vocab,
                                    mtp_rows() * sizeof(float), hipMemcpyDeviceToHost, nullptr));
    beat("mtp-end", -1);
    evict_sync(mbox_host_);
    int pick_nan = 0;
    for (int i = 0; i < k; i++) pick_nan += ((const int*)mpickx_host_)[slots[i] * 4 + 3];
    report_mtp_nan(mpickx_host_ + MEGA_SMAX * 4, pick_nan, pos[0]);
    HIPCHECK(hipGetLastError());
    check_aborts();
}

Mega::MtpPick Mega::multi_pick(int slot) const {
    const float* r = mpickx_host_ + slot * 4;
    int32_t t;
    std::memcpy(&t, &r[0], 4);
    t = valid_row(t, mtp_rows());
    return {mtp_vocab_.empty() ? t : mtp_vocab_[t], r[2] > 0 ? 1.0 / r[2] : 1.0};
}

// Layout of the MTP experts: all in one buffer, per expert [gate | up | down] like a slot
void Mega::mtp_experts_alloc(bool q8) {
    const MtpHead& M = *mtp_;
    const int64_t d = h_.d, ff = h_.expert_ff, E = M.n_expert;
    mtp_ex_type_ = q8 ? 8u : 42u;
    auto up256 = [](int64_t v) { return (v + 255) / 256 * 256; };
    const int64_t g_b = ff * row_bytes_of(mtp_ex_type_, d), d_b = d * row_bytes_of(mtp_ex_type_, ff);
    mtp_ex_up_ = up256(g_b), mtp_ex_down_ = mtp_ex_up_ + up256(g_b), mtp_ex_bytes_ = mtp_ex_down_ + up256(d_b);
    const size_t total = (size_t)(E * mtp_ex_bytes_);
    void* p;
    HIPCHECK(trace::dev_malloc(&p, total + kMatvecReadPad));
    HIPCHECK(hipMemset(p, 0, total + kMatvecReadPad));
    allocs_.push_back(p);
    vram_ += total;
    mtp_ex_ = static_cast<uint8_t*>(p);
}

// Matrix k (0 gate, 1 up, 2 down) of all MTP experts, 32 experts at a time through a staging buffer in VRAM (56 MB)
void Mega::mtp_experts_matrix(int k, bool q8, uint8_t*& stage) {
    const MtpHead& M = *mtp_;
    const int64_t d = h_.d, ff = h_.expert_ff, E = M.n_expert;
    const TensorInfo* src[3] = {M.L.gate_exps, M.L.up_exps, M.L.down_exps};
    const int64_t off[3] = {0, mtp_ex_up_, mtp_ex_down_};
    const int64_t rows[3] = {ff, ff, d}, ins[3] = {d, d, ff};
    constexpr int CH = 32;
    const TensorInfo& t = *src[k];
    if (t.type != 8 || ins[k] % 64) throw std::runtime_error("gpu::Mega: MTP experts are not Q8_0");
    const int64_t srb = row_bytes_of(t.type, ins[k]), ebytes_src = rows[k] * srb;
    if (!stage)
        HIPCHECK(trace::dev_malloc(&stage, (size_t)CH * std::max(ff * row_bytes_of(8, d), d * row_bytes_of(8, ff))));
    for (int64_t e0 = 0; e0 < E; e0 += CH) {
        const int64_t ne = std::min<int64_t>(CH, E - e0);
        const uint8_t* s = t.data + e0 * ebytes_src;
        if (q8) {
            for (int64_t e = 0; e < ne; e++) {  // the rows of an expert are contiguous
                HIPCHECK(hipMemcpy(mtp_ex_ + (e0 + e) * mtp_ex_bytes_ + off[k], s + e * ebytes_src, ebytes_src,
                                   hipMemcpyHostToDevice));
                progress::add((double)ebytes_src);
            }
            continue;
        }
        HIPCHECK(hipMemcpy(stage, s, (size_t)(ne * ebytes_src), hipMemcpyHostToDevice));
        progress::add((double)(ne * ebytes_src));
        launch_q8_to_q2({stage, ne * rows[k] * (ins[k] / 64), (int)ins[k], (int)rows[k], (int)e0, mtp_ex_ + off[k],
                         mtp_ex_bytes_, row_bytes_of(mtp_ex_type_, ins[k])});
    }
}

// MTP experts (all of them) into one buffer. TONA_MTP_EXP=q8: as in the GGUF (Q8_0, 2.7 GB), otherwise Q2_0 (0.7 GB,
// research/06 section 8), converted on the GPU
void Mega::upload_mtp_experts() {
    const char* mode = env::text("TONA_MTP_EXP");
    const bool q8 = mode && std::string_view(mode) == "q8";
    const auto t0 = std::chrono::steady_clock::now();
    mtp_experts_alloc(q8);
    uint8_t* stage = nullptr;
    for (int k = 0; k < 3; k++) mtp_experts_matrix(k, q8, stage);
    HIPCHECK(hipDeviceSynchronize());
    if (stage) HIPCHECK(trace::dev_free(stage));
    std::fprintf(stderr, "Initialization: %lld MTP experts as %s (%.0f MB) in %.1f s\n", (long long)mtp_->n_expert,
                 q8 ? "Q8_0" : "Q2_0, converted on the GPU", (double)(mtp_->n_expert * mtp_ex_bytes_) / 1e6,
                 seconds_since<std::chrono::steady_clock>(t0));
}

}  // namespace tona::gpu
