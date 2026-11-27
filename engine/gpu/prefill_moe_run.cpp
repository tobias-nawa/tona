// PrefillRun: the MoE part of a layer. Router and grouping of the tokens by expert, then the experts in batches: their
// weights come from the cache, a layer buffer, the pinned RAM (copied while the previous batch computes) or the file.
#include <algorithm>
#include <cstdio>
#include <stdexcept>

#include "gpu/expert_parts.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

// The regions of the count area must not run into the NaN word and the flags
static_assert(route_nan_word(MEGA_MAX_EXPERTS) == 768 && route_flag_word(MEGA_MAX_EXPERTS) == 1000 &&
                  route_count_words(MEGA_MAX_EXPERTS) == 1024,
              "the Coder's count area changed");

// Without a check nobody reads the output of the last layer (the last prompt token is computed by forward): there only
// KV, indexer pool and ring are written, and attention, o, MoE and write-back are skipped. TONA_PF_FULL=1: everything.
// The MTP head needs h after the last layer for every token.
bool PrefillRun::is_tail(uint32_t l) const {
    return !job_.nll && !flags_.full && (!M.mtp_on_ || flags_.mtp_tail) && l + 1 == M.h_.n_layer && M.h_.is_qsa(l);
}

// Fill layer buffer l & 1 with the experts of layer l that are not in the cache (all, in order, without waiting for
// the routing); consecutive pinned places go in one copy
void PrefillRun::issue_layer(uint32_t l) {
    if (!layered || l >= M.h_.n_layer) return;
    if (is_tail(l)) return;  // no MoE there: do not fetch the experts over PCIe (~0.5 GB per group)
    const int b = (int)(l & 1);
    const LayerCache& LC = M.cache_[l];
    const size_t sb = LC.slot_bytes;
    const int E = (int)M.experts_.experts(l);
    HIPCHECK(hipStreamWaitEvent(M.pf_cs_, M.pf_lev_[b], 0));
    for (int e = 0; e < E;) {
        if (auto it = LC.where.find(e); it != LC.where.end()) {
            res[b][e] = LC.slots + (size_t)it->second.first * sb;
            e++;
            continue;
        }
        // Run: experts e ... e + r - 1 not in the cache, pinned one behind the other, targets one behind the other
        // (distance max_sb only if sb = max_sb; otherwise one by one)
        int r = 1;
        while (sb == max_sb && e + r < E && !LC.where.contains(e + r) && M.pin_[l].idx[e + r] == M.pin_[l].idx[e] + r &&
               lbuf[b][e + r] == lbuf[b][e] + (size_t)r * sb)
            r++;
        HIPCHECK(hipMemcpyAsync(lbuf[b][e], M.pin_[l].host + (size_t)M.pin_[l].idx[e] * sb, (size_t)r * sb,
                                hipMemcpyHostToDevice, M.pf_cs_));
        for (int q = 0; q < r; q++) res[b][e + q] = lbuf[b][e + q];
        e += r;
    }
    HIPCHECK(hipEventRecord(M.pf_lev_[2 + b], M.pf_cs_));
}

MoeBufs PrefillRun::moe_buffers() {
    Carve p{B.phase, 0, arena_end};
    return carve_moe(p, D, C);
}

// Router, counts to the host, and the shared expert while the CPU reads the counts
void PrefillRun::moe_route(uint32_t l, const MoeBufs& b) {
    const PfLayer& F = M.pf_[M.m_.moe_block(l)];
    const int d = D.d, sff = D.sff, E = (int)M.experts_.experts(l);
    gemmq_f16(F.moe.router, B.mixed16, d, b.rl, E, n);
    HIPCHECK(hipMemsetAsync(b.cnt_d, 0, route_count_words(D.max_e) * sizeof(int32_t), st));
    launch_route({b.rl, b.sel, b.sw, b.cnt_d, n, E, D.ne, D.max_e}, st);
    HIPCHECK(hipMemcpyAsync(M.pf_hostc_, b.cnt_d, E * sizeof(int32_t), hipMemcpyDeviceToHost, st));
    HIPCHECK(hipMemcpyAsync(M.pf_hostc_ + route_nan_word(D.max_e), b.cnt_d + route_nan_word(D.max_e), sizeof(int32_t),
                            hipMemcpyDeviceToHost, st));
    share_x8(true);
    gemmq(F.moe.sh_gate, B.mixed16, d, b.sg, sff, n);
    gemmq(F.moe.sh_up, B.mixed16, d, b.su, sff, n);
    share_x8(false);
    launch_swiglu16({b.sg, b.su, b.sa16, 0, n, sff}, st);
    gemmq(F.moe.sh_down, b.sa16, sff, b.sho, d, n);
    gemmq(F.moe.sh_inp, B.mixed16, d, b.shi, 1, n);
}

// Offsets of the experts' rows from the counts, rows of x grouped by expert; returns the experts with tokens
std::vector<int> PrefillRun::moe_group(uint32_t l, const MoeBufs& b, const _Float16* x, int width, int ne) {
    HIPCHECK(hipStreamSynchronize(st));
    const int32_t* cnt = M.pf_hostc_;
    int32_t* off = M.pf_hostc_ + route_off_word(D.max_e);
    const int E = (int)M.experts_.experts(l);
    int32_t acc = 0;
    for (int e = 0; e < E; e++) off[e] = acc, acc += cnt[e], M.use_[M.experts_.offset(l) + e] += cnt[e];
    report_nan_rows(l, M.pf_hostc_[route_nan_word(D.max_e)]);
    if (flags_.check) {
        if (!pf_counts_ok(cnt, E, (int64_t)n * ne)) check_bad++, log_check(l, "counts", 0, 0, 0);
        moe_check(l, b, 0, ne);
    }
    HIPCHECK(hipMemcpyAsync(b.cnt_d + route_off_word(D.max_e), off, E * sizeof(int32_t), hipMemcpyHostToDevice, st));
    HIPCHECK(hipMemsetAsync(b.cnt_d + route_fill_word(D.max_e), 0, E * sizeof(int32_t), st));  // fill level per expert
    launch_scatter(
        {b.sel, b.cnt_d + route_off_word(D.max_e), b.cnt_d + route_fill_word(D.max_e), b.rowmap, x, b.xg, n, ne, width},
        st);
    if (flags_.check) moe_check(l, b, 1, ne);
    std::vector<int> used;
    for (int e = 0; e < E; e++)
        if (cnt[e] > 0) used.push_back(e);
    return used;
}

// A NaN router logit means that something before the router is wrong (stale workspace, a race): the route takes it as
// -inf so that no address goes wild, but the weights of that token are wrong. Always reported, once per prefill
void PrefillRun::report_nan_rows(uint32_t l, int32_t rows) {
    if (rows <= 0 || nan_reported) return;
    nan_reported = true;
    std::fprintf(stderr,
                 "WARNING: %d of %d tokens with a NaN router logit at layer %u, chunk start %lld; plan: chunk %d, %s, "
                 "extra slots %s\n",
                 (int)rows, n, l, (long long)pos0, C, layered ? "layer by layer" : "chunk by chunk",
                 M.extras_on_ ? "on" : "off");
}

void PrefillRun::log_check(uint32_t l, const char* what, int a, int e, int r) const {
    std::fprintf(stderr,
                 "TONA_PF_CHECK: %s violated at layer %u, chunk start %lld, n %d: assignment %d expert %d row %d\n",
                 what, l, (long long)pos0, n, a, e, r);
}

// TONA_PF_CHECK: the router's experts (stage 0) and the rows of the scatter (stage 1) against the counts and offsets of
// the step. A violation is logged and counted, the run goes on. The flag words sit in the unused tail of cnt_d.
void PrefillRun::moe_check(uint32_t l, const MoeBufs& b, int stage, int ne) {
    int32_t* flag = b.cnt_d + route_flag_word(D.max_e);
    HIPCHECK(hipMemsetAsync(flag, 0, 4 * sizeof(int32_t), st));
    launch_moe_check(
        {b.sel, b.rowmap, b.cnt_d, b.cnt_d + route_off_word(D.max_e), flag, n, ne, (int)M.experts_.experts(l), stage},
        st);
    int32_t h[4];
    HIPCHECK(hipMemcpyAsync(h, flag, sizeof h, hipMemcpyDeviceToHost, st));
    HIPCHECK(hipStreamSynchronize(st));
    check_steps++;
    if (h[0] == 0) return;
    check_bad++;
    static constexpr const char* kWhat[] = {"expert range", "row range after the scatter", "row range before moe_sum"};
    log_check(l, kWhat[stage], h[1], h[2], h[3]);
}

// Weights of expert e as place s of the batch: layer buffer, cache slot, pinned RAM (copied to the staging buffer
// of the batch on the copy stream), or read from the file into the staging area
const uint8_t* PrefillRun::expert_weights(uint32_t l, int e, int s, const BatchPlan& bp, BatchState& bs) {
    const LayerCache& LC = M.cache_[l];
    const size_t sb = LC.slot_bytes;
    if (layered) return res[l & 1][e];
    if (auto it = LC.where.find(e); it != LC.where.end()) return LC.slots + (size_t)it->second.first * sb;
    if (!M.pin_.empty() && M.pin_[l].idx[e] >= 0) {
        if (!bs.copied) {
            HIPCHECK(hipStreamWaitEvent(M.pf_cs_, M.pf_ev_[bp.buf], 0));
            bs.copied = true;
        }
        HIPCHECK(hipMemcpyAsync(bp.stg[bp.buf] + (size_t)s * sb, M.pin_[l].host + (size_t)M.pin_[l].idx[e] * sb, sb,
                                hipMemcpyHostToDevice, M.pf_cs_));
        return bp.stg[bp.buf] + (size_t)s * sb;
    }
    if (bs.staged == 0) HIPCHECK(hipStreamSynchronize(st));  // the staging area is reused
    const MoeWeights& S = M.m_.moe_weights(l);
    const ExpertParts P = expert_parts(S, LC);
    for (int k = 0; k < P.n; k++) {
        const size_t per = P.t[k]->bytes / P.t[k]->dims[2];
        M.m_.gguf().read(*P.t[k], (size_t)e * per, M.staging_ + (size_t)s * sb + P.off[k], per);
    }
    bs.staged++;
    return M.staging_dev_ + (size_t)s * sb;
}

// Collects the weights of the experts of one batch; makes the compute stream wait for the copies
ExpertBatch PrefillRun::stage_batch(uint32_t l, const std::vector<int>& used, const BatchPlan& bp) {
    const LayerCache& LC = M.cache_[l];
    ExpertBatch eb;
    eb.m = bp.m;
    BatchState bs;
    for (int s = 0; s < bp.m; s++) {
        const uint8_t* w = expert_weights(l, used[bp.b0 + s], s, bp, bs);
        eb.wg[s] = w, eb.wu[s] = w + LC.off_up, eb.wd[s] = w + LC.off_down;
    }
    if (bs.copied) {
        HIPCHECK(hipEventRecord(M.pf_ev_[2 + bp.buf], M.pf_cs_));
        HIPCHECK(hipStreamWaitEvent(st, M.pf_ev_[2 + bp.buf], 0));
    }
    return eb;
}

// gate and up, SwiGLU, down for the experts of one batch; weights are read in their format and unpacked in the LDS
void PrefillRun::run_batch(uint32_t l, const MoeBufs& b, const std::vector<int>& used, const BatchPlan& bp,
                           const ExpertBatch& eb) {
    const MoeWeights& S = M.m_.moe_weights(l);
    if (!S.gate_exps) return run_batch_ungated(l, b, used, bp, eb);
    const int d = D.d, ff = D.ff, m = eb.m;
    const int32_t* cnt = M.pf_hostc_;
    const int32_t* off = M.pf_hostc_ + route_off_word(D.max_e);
    const int64_t rbg = row_bytes_of(S.gate_exps->type, d), rbu = row_bytes_of(S.up_exps->type, d);
    const int64_t rbd = row_bytes_of(S.down_exps->type, ff);
    GemmQSeg sgu[2 * MB], sd[MB];  // gate, then up
    for (int s = 0; s < m; s++) {
        const int e = used[bp.b0 + s];
        sgu[s] = {b.xg + (size_t)off[e] * d, eb.wg[s], b.G + (size_t)off[e] * ff, cnt[e], ff, d, d, ff, rbg};
        sgu[m + s] = {b.xg + (size_t)off[e] * d, eb.wu[s], b.U + (size_t)off[e] * ff, cnt[e], ff, d, d, ff, rbu};
        sd[s] = {b.A16 + (size_t)off[e] * ff, eb.wd[s], b.O + (size_t)off[e] * d, cnt[e], d, ff, ff, d, rbd};
    }
    if (S.gate_exps->type == S.up_exps->type) {  // one launch for both
        gemm_q(S.gate_exps->type, sgu, 2 * m, st);
    } else {
        gemm_q(S.gate_exps->type, sgu, m, st);
        gemm_q(S.up_exps->type, sgu + m, m, st);
    }
    const int e0 = used[bp.b0], e1 = used[bp.b0 + m - 1];
    const int64_t r0 = off[e0], rows = off[e1] + cnt[e1] - r0;
    launch_swiglu16({b.G, b.U, b.A16, r0, rows, ff}, st);
    gemm_q(S.down_exps->type, sd, m, st);
}

// The batches of the experts that have tokens. Pinned experts are copied to VRAM in one piece on the copy stream (two
// buffers inside w16) while the previous batch computes. Read directly over PCIe, gemm_q fetched 18-98 scattered
// bytes per row and K step: IQ4_NL 0.92 instead of 0.25 ms per batch.
void PrefillRun::moe_experts(uint32_t l, int j, const MoeBufs& b, const std::vector<int>& used) {
    const size_t sb = M.cache_[l].slot_bytes;
    if (layered && j == 0) HIPCHECK(hipStreamWaitEvent(st, M.pf_lev_[2 + (l & 1)], 0));  // layer buffer full
    ensure_copy_stream();
    // As many experts per batch as two copy buffers in w16 hold, at most MB: with 8 experts per batch gate and up had
    // only 40 blocks for 32 CUs
    const int mb = (int)std::min<size_t>(MB, B.w16_elems * sizeof(_Float16) / (2 * sb));
    if (mb < 1) throw std::runtime_error("gpu::Mega: prefill staging buffer too small for a batch of experts");
    uint8_t* w16 = reinterpret_cast<uint8_t*>(B.w16);
    HIPCHECK(hipEventRecord(M.pf_ev_[0], st));  // w16 free: the shared expert is done
    HIPCHECK(hipEventRecord(M.pf_ev_[1], st));
    for (size_t b0 = 0; b0 < used.size(); b0 += mb) {
        const BatchPlan bp{
            b0, (int)std::min<size_t>(mb, used.size() - b0), (int)(b0 / mb) & 1, {w16, w16 + (size_t)mb * sb}};
        run_batch(l, b, used, bp, stage_batch(l, used, bp));
        HIPCHECK(hipEventRecord(M.pf_ev_[bp.buf], st));  // buffer buf is free again
    }
}

// MoE: router, grouping by expert, shared expert, the routed experts, weighted sum
void PrefillRun::moe(uint32_t l, int j) {
    if (M.pf_[M.m_.moe_block(l)].moe.lat_down.w) return moe_latent(l, j);
    const MoeBufs b = moe_buffers();
    moe_route(l, b);
    const std::vector<int> used = moe_group(l, b, B.mixed16, D.d, D.ne);
    moe_experts(l, j, b, used);
    if (flags_.check) moe_check(l, b, 2, D.ne);  // the rowmap must survive the expert GEMMs
    launch_moe_sum({b.O, b.rowmap, b.sw, b.sho, b.shi, B.y, n, D.d, D.ne}, st);
}

}  // namespace tona::gpu
