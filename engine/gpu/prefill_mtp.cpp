// PrefillRun: the LM head check (negative log-likelihood), the MTP entries of the prompt, and what the prefill leaves
// behind for the decode path (tokens, MTP state, drafter features, returned slots).
#include <algorithm>
#include <chrono>
#include <cmath>

#include "backend/env.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_run.h"
#include "model/quant.h"

namespace tona::gpu {

// Targets of the chunk: the next token, for the last one of the chunk the first of the following
void PrefillRun::head_targets(int j, int32_t* tgt) {
    int32_t* th = M.pf_hostc_ + route_count_words(D.max_e);
    const int32_t* tk = job_.tokens + (size_t)j * C;
    const int32_t nx = j + 1 < nsub ? job_.tokens[(size_t)(j + 1) * C] : job_.next;
    for (int t = 0; t < n; t++) th[t] = t + 1 < n ? tk[t + 1] : nx;
    HIPCHECK(hipMemcpyAsync(tgt, th, n * sizeof(int32_t), hipMemcpyHostToDevice, st));
}

// Check: LM head for all tokens of chunk j, -log p of the following token
void PrefillRun::head(int j) {
    prefill_sequence(M.m_).head_input(*this, j);
    const int d = D.d;
    Carve p{B.phase, 0, arena_end};
    const int64_t rs_max = (int64_t)(B.w16_elems / d) / 128 * 128;
    float* L = p.take<float>((size_t)C * rs_max);
    float* mrun = p.take<float>(C);
    float* srun = p.take<float>(C);
    float* tlog = p.take<float>(C);
    int32_t* tgt = p.take<int32_t>(C);
    head_targets(j, tgt);
    launch_fill({mrun, n, -INFINITY}, st);
    launch_fill({srun, n, 0.0f}, st);
    launch_fill({tlog, n, 0.0f}, st);
    const PfW& out = M.pf_out_;
    const int64_t rb = row_bytes_of(out.type, out.in);
    for (int64_t r0 = 0; r0 < out.rows; r0 += rs_max) {
        const int64_t rs = std::min(rs_max, out.rows - r0);
        dequant_f16({out.type, rs, d, d}, static_cast<const uint8_t*>(out.w) + r0 * rb, B.w16, st);
        const GemmSeg s{B.mixed16, B.w16, L, n, (int32_t)rs, d, d, d, (int32_t)rs};
        gemm_f16(&s, 1, st);
        launch_lse({L, (int)rs, (int)r0, tgt, mrun, srun, tlog}, n, st);
    }
    std::vector<float> hm(n), hs(n), ht(n);
    HIPCHECK(hipMemcpyAsync(hm.data(), mrun, n * sizeof(float), hipMemcpyDeviceToHost, st));
    HIPCHECK(hipMemcpyAsync(hs.data(), srun, n * sizeof(float), hipMemcpyDeviceToHost, st));
    HIPCHECK(hipMemcpyAsync(ht.data(), tlog, n * sizeof(float), hipMemcpyDeviceToHost, st));
    HIPCHECK(hipStreamSynchronize(st));
    for (int t = 0; t < n; t++) (*job_.nll)[(size_t)j * C + t] = std::log(hs[t]) + hm[t] - ht[t];
}

// h of MTP entry q is R at q - 1 (the first one from the state)
const float* PrefillRun::mtp_h_at(int64_t q) {
    if (q < pos_all) {
        const float* h = M.mtp_h_of(q);
        if (h == M.mtp_bad_) M.stats_.mtp_stale++, h = nullptr;
        return h;
    }
    const int64_t j = (q - pos_all) / C;
    return Rp[j] + (q - pos_all - j * C) * D.hcd;
}

// MTP entries pos_all ... pos_ - 1, MEGA_BMAX at a time
void PrefillRun::mtp_publish() {
    const float* hs[MEGA_BMAX];
    // the entries of the prompt by passes (an architecture without its own GEMMs, or TONA_MTP_PF_MEGA) or a gap before
    // the chunk
    const int64_t mend = mtp_by_pass_ ? M.pos_ : pos_all;
    for (int64_t p = std::min(M.mtrk_.valid, pos_all); p < mend;) {
        const int nb = (int)std::min<int64_t>(MEGA_BMAX, mend - p);
        for (int b = 0; b < nb; b++) hs[b] = mtp_h_at(p + b - 1);
        M.mtp_pass(M.tokens_.data() + p, nb, p, hs, false, true);
        p += nb;
    }
    HIPCHECK(
        hipMemcpyAsync(M.mtp_hlast_, mtp_h_at(M.pos_ - 1), D.hcd * sizeof(float), hipMemcpyDeviceToDevice, nullptr));
    HIPCHECK(hipStreamSynchronize(nullptr));
    M.mtrk_.after_prefill(M.pos_);
    M.stats_.mtp_pf_s += std::chrono::duration<double>(std::chrono::steady_clock::now() - t_mtp).count();
}

// Drafter: feed the features of the positions in its window, MEGA_BMAX at a time
void PrefillRun::inject_features() {
    if (!M.drafter_ || D.nfeat <= 0) return;
    const int64_t a = std::max(job_.win0, pos_all);
    for (int64_t p = a; p < M.pos_; p += MEGA_BMAX) {
        const int64_t j = (p - pos_all) / C, q = p - pos_all - j * C;
        const int m = (int)std::min<int64_t>({(int64_t)MEGA_BMAX, M.pos_ - p, C - q});  // not across a chunk
        M.inject_draft(Fp[j] + q * D.nfeat, p, m);
        p += m - MEGA_BMAX;
    }
}

// The state after the prefill is what forward + commit would leave
void PrefillRun::finish() {
    M.tokens_.insert(M.tokens_.end(), job_.tokens, job_.tokens + job_.n);
    M.pos_ += job_.n;
    M.pending_commit_ = 0;  // k_dn_recur committed the open tokens of the last decode pass
    M.batch_.clear();
    if (M.mtp_on_) mtp_publish();
    M.trunk_ok_ = false;
    inject_features();
    // The slots stay borrowed between the groups of one prompt: the copy back took 129 to 348 ms of idle GPU at each
    // boundary (research/measurements/2026-11-16-prefill-breakdown.md); the next group reuses them
    static const bool keep = tona::env::integer("TONA_PF_LOAN_KEEP", 1) != 0;
    if (!keep || job_.next < 0) M.pf_return_loans();
}

}  // namespace tona::gpu
