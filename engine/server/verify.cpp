// Verify pass: accept or reject the drafts against the target model's logits, commit, emit tokens
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "backend/env.h"
#include "backend/host_trace.h"
#include "server/generator.h"
#include "server/generator_env.h"

namespace tona::server {

// Thinking budget: counts tokens inside <think>; when it runs out, the closing text follows as forced tokens
void Generator::count_think(Stream& s, int32_t token) {
    if (s.think_left < 0) return;
    if (token == think_end_)
        s.think_left = -1;
    else if (--s.think_left <= 0)
        s.think_left = -1, s.forced.assign(think_force_.begin(), think_force_.end());
}

// Sampling: accept draft d with probability p(d) / q(d), otherwise draw the residual (speculative sampling)
int Generator::accept_sampled(Stream& s, const float* lg, int32_t& next) {
    const size_t V = m_.dims().vocab;
    const std::vector<int32_t>& blk = s.blk;
    std::uniform_real_distribution<double> uni(0, 1);
    int a = 0;
    while (true) {
        const Dist D = make_dist(lg + a * V, V, s.smp);
        if (a + 1 >= (int)blk.size()) {
            next = sample(D, s.rng);
            return a;
        }
        const Dist* Q = a < (int)s.qd.size() ? &s.qd[a] : nullptr;
        if (uni(s.rng) * (Q ? Q->prob(blk[a + 1]) : 1.0) < D.prob(blk[a + 1])) {
            a++;
            continue;
        }
        next = Q ? sample_residual(D, s.rng, *Q) : sample(D, s.rng, blk[a + 1]);
        return a;
    }
}

// TONA_DIAG_SCAN=1 with TONA_HOST_TRACE (diagnosis, not a feature): is the scan slow because of the memory the logits
// arrive in? Cycles alternate: serial scan of the pinned buffer (cold), or a copy into ordinary memory and the serial
// and the parallel scan of the copy. The marks tell the times; the production scan follows and is not meant to be read
// in those cycles.
static void diag_scan(const float* lg, int rows, size_t V) {
    static int cycle = 0;
    const int kind = cycle++ % 3;
    volatile int sink = 0;
    if (kind == 0) {
        backend::HostTrace::mark("d_ser_orig_begin");
        for (int r = 0; r < rows; r++) sink = sink + argmax_scan(lg + (size_t)r * V, V).best;
        backend::HostTrace::mark("d_ser_orig_end");
    } else if (kind == 2) {
        static std::vector<float> copy;
        copy.resize((size_t)rows * V);
        backend::HostTrace::mark("d_cpy_begin");
        std::copy(lg, lg + (size_t)rows * V, copy.data());
        backend::HostTrace::mark("d_cpy_end");
        for (int r = 0; r < rows; r++) sink = sink + argmax_scan(copy.data() + (size_t)r * V, V).best;
        backend::HostTrace::mark("d_ser_copy_end");
#pragma omp parallel for schedule(static) if (rows >= 2)
        for (int r = 0; r < rows; r++) sink = sink + argmax_scan(copy.data() + (size_t)r * V, V).best;
        backend::HostTrace::mark("d_par_copy_end");
    }
}

// Number of accepted drafts; next is the token after them (from the logits of the last accepted position)
int Generator::accept_drafts(Stream& s, const float* lg, int32_t& next) {
    const size_t V = m_.dims().vocab;
    const std::vector<int32_t>& blk = s.blk;
    if (s.forcing) {
        const int a = (int)blk.size() - 1;
        if (!s.forced.empty())
            next = s.forced.front(), s.forced.pop_front();
        else
            next = s.greedy ? argmax(lg + a * V, V) : sample(make_dist(lg + a * V, V, s.smp), s.rng);
        return a;
    }
    if (!s.greedy) return accept_sampled(s, lg, next);
    // The rows of the block are independent: scan them all in parallel (the trace of a decode cycle showed 0.3 ms of
    // serial scans after the logits, research/measurements/2026-11-17-b1-cycle-breakdown.md), then accept in order.
    // Same tokens; only the rows that are used are reported for NaN, as before.
    const int rows = (int)blk.size();
    static const bool diag = backend::HostTrace::on() && tona::env::flag("TONA_DIAG_SCAN");
    if (diag) diag_scan(lg, rows, V);
    std::vector<ArgmaxRow> am((size_t)rows);
    backend::RegionTrace rt("scan_enter", "scan_worker_start", "scan_first_done", "scan_exit", rows);
    rt.enter();
#pragma omp parallel for schedule(static) if (rows >= 2)
    for (int r = 0; r < rows; r++) {
        rt.row_begin(r);
        am[(size_t)r] = argmax_scan(lg + (size_t)r * V, V);
        rt.row_end(r);
    }
    rt.leave();
    auto take = [&](int r) {
        if (am[(size_t)r].nan) tona::nan::report("greedy argmax", am[(size_t)r].nan);
        return am[(size_t)r].best;
    };
    int a = 0;
    while (a + 1 < rows && take(a) == blk[a + 1]) a++;
    next = take(a);
    return a;
}

void Generator::log_pass(const Stream& s, int a, double ms_fw) const {
    FILE* dl = draft_log();
    if (s.forcing || !dl || s.from_sfx) return;
    const std::vector<int32_t>& blk = s.blk;
    if (!dfl_) {
        std::fprintf(dl, "%d %d %.2f %.2f %d\n", (int)blk.size() - 1, a, ms_fw, s.ms_drf, (int)s.raw_q.size());
        std::fflush(dl);
        return;
    }
    std::fprintf(dl, "%d %d", (int)blk.size() - 1, a);
    for (int i = 0; i < n_draft_max(); i++) std::fprintf(dl, " %.4f", i < (int)s.raw_q.size() ? s.raw_q[i] : 0.0);
    std::fprintf(dl, " %.2f", ms_fw);
    for (double v : s.calib[s.greedy ? 0 : 1].params()) std::fprintf(dl, " %.3f", v);
    std::fprintf(dl, "\n");
    std::fflush(dl);
}

// Large block: after a small block with all drafts accepted (and their calibrated product >= TONA_DRAFT_ENTER_P), then
// stay as long as >= TONA_DRAFT_STAY are accepted
void Generator::update_block_size(Stream& s, int a) {
    static const int big_at = (int)tona::env::integer("TONA_DRAFT_BIG_AT", 0);
    static const int stay = (int)tona::env::integer("TONA_DRAFT_STAY", 0);
    static const double enter_p = tona::env::number("TONA_DRAFT_ENTER_P", 0);
    // TONA_DRAFT_BIG_ALWAYS: always the large block (with TONA_DRAFT_FULL like llama.cpp --spec-draft-n-max 7)
    static const bool big_always = tona::env::flag("TONA_DRAFT_BIG_ALWAYS");
    const size_t blk_size = s.blk.size();
    if (stay > 0 && !s.from_sfx) {
        double pp = 1;
        for (int i = 0; i < s.nd && i < (int)s.probs.size(); i++) pp *= s.probs[i];
        s.full = s.nd > N_SMALL ? a >= stay : a == s.nd && blk_size == (size_t)s.nd + 1 && pp >= enter_p;
    } else {
        // TONA_DRAFT_BIG_AT=k: large block after k accepted drafts (otherwise: all drafts of the pass)
        s.full = big_at > 0 ? a >= big_at : a + 1 == (int)blk_size && blk_size > 1;
    }
    if (big_always) s.full = true;
}

// Emits the accepted drafts; false if the stream ended (flagged done)
bool Generator::emit_accepted(Stream& s, int a) {
    for (int i = 1; i <= a; i++) {
        const int32_t t = s.blk[i];
        if (is_stop(t)) {
            s.why = Stop::eos, s.done = true;
            return false;
        }
        count_think(s, t);
        if (!s.on_token(t)) {
            s.why = Stop::callback, s.done = true;
            return false;
        }
        if (++s.n >= s.max_tokens) {
            s.why = Stop::length, s.done = true;
            return false;
        }
    }
    return true;
}

void Generator::post(Stream& s, const float* lg, double ms_fw) {
    int32_t next = 0;
    const int a = accept_drafts(s, lg, next);
    const int drafted = (int)s.blk.size() - 1;
    log_pass(s, a, ms_fw);
    if (!s.from_sfx && !s.forcing) (dfl_ ? s.calib : s.mcal)[s.greedy ? 0 : 1].observe(s.raw_q.data(), drafted, a);
    mg_.commit(a + 1);
    s.prev_n = (int)s.blk.size(), s.prev_e = a + 1;
    // TONA_TOKEN_LOG=<file>: accepted tokens per pass (coverage of shortened heads)
    static FILE* tl = tona::env::text("TONA_TOKEN_LOG") ? std::fopen(tona::env::text("TONA_TOKEN_LOG"), "a")
                                                        : nullptr;  // own stream, "a"
    if (tl) {
        for (int i = 1; i <= a; i++) std::fprintf(tl, "%d ", s.blk[i]);
        std::fprintf(tl, "%d\n", next), std::fflush(tl);
    }
    update_block_size(s, a);
    pending_ = dfl_ ? a + 1 : 0;
    if (!s.forcing) passes_++, gen_tokens_ += a + 1, s.passes++, s.gen_tokens += a + 1;
    if (s.from_sfx) sfx_passes_++, sfx_tokens_ += a + 1, s.sfx_passes++, s.sfx_tokens += a + 1;
    if (emit_accepted(s, a)) s.last = next;
}

}  // namespace tona::server
