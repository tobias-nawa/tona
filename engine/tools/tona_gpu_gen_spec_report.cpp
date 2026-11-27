// stderr report of the speculative run. scripts/bench-decode.sh greps "tokens per pass", "Experts:", "Misses per pass"
// and "per pass, N tok/s" from it; change a line and the pattern in the same commit.
#include <cstdio>

#include "backend/env.h"
#include "tona_gpu_gen_spec.h"

namespace tona::gen {

void SpecRun::report_summary() {
    const int gen = (int)out_.size() - 1;
    const double per_pass = std::max(1, passes_);
    std::fprintf(stderr, "Prompt %zu tokens in %.2f s; %d tokens in %d passes = %.2f tokens per pass\n", ids_.size(),
                 t_prompt_, gen, passes_, gen / per_pass);
    std::fprintf(stderr, "Passes per width:");
    for (int i = 1; i <= n_draft_ + 1; i++) std::fprintf(stderr, " %d: %d", i, width_[i]);
    std::fprintf(stderr, "\nAccepted drafts per pass:");
    for (int i = 0; i <= n_draft_; i++) std::fprintf(stderr, " %d: %d", i, hist_[i]);
    std::fprintf(stderr,
                 "\nTarget %.2f ms per pass (%.1f tok/s decode, target only), drafter (%s) %.2f ms per pass, %.2f "
                 "tok/s decode\n",
                 1e3 * t_target_ / per_pass, gen / t_target_, env_.cpu ? "CPU" : "GPU", 1e3 * t_draft_ / per_pass,
                 gen / (t_target_ + t_draft_));
    if (env_.check) std::fprintf(stderr, "GPU drafts equal to CPU drafts: %d of %d\n", same_draft_, cmp_draft_);
}

// Per pass: how many draft tokens agree with the output (only passes with `probe` tokens after them).
void SpecRun::report_probe() {
    const int probe = env_.probe;
    std::vector<int> matches(probe + 1, 0);
    int counted = 0;
    for (const auto& [k, draft] : probes_) {
        if (k + 1 + probe > out_.size()) continue;
        int mt = 0;
        while (mt < probe && mt < (int)draft.size() && draft[mt] == out_[k + 1 + mt]) mt++;
        matches[mt]++, counted++;
    }
    std::fprintf(stderr, "Draft of %d, matching (%d passes):", probe, counted);
    for (int i = 0; i <= probe; i++) std::fprintf(stderr, " %d: %d", i, matches[i]);
    std::fprintf(stderr, "\nTokens per pass at width w (draft w - 1 from this block):");
    for (int w : {2, 3, 4, 5, 6, 8, 12, 16}) {
        if (w - 1 > probe) continue;
        double e = 0;
        for (int i = 0; i <= probe; i++) e += matches[i] * (std::min(i, w - 1) + 1.0);
        std::fprintf(stderr, " %d: %.2f", w, e / std::max(1, counted));
    }
    std::fprintf(stderr, "\n");
}

void SpecRun::report_experts() {
    const auto& st = mg_.stats();
    const auto& st0 = stats0_;
    const double per_pass = std::max(1, passes_);
    const double hits = (double)(st.expert_hits - st0.expert_hits),
                 miss = (double)(st.expert_misses - st0.expert_misses);
    std::fprintf(stderr, "Experts: %.1f %% hits, %.0f MB reloaded, of which from disk %.0f MB (whole process)\n",
                 100.0 * hits / std::max(1.0, hits + miss), (st.upload_bytes - st0.upload_bytes) / 1e6,
                 disk_read_mb() - disk0_);
    std::fprintf(stderr, "CPU per pass: selection %.2f ms\n", 1e3 * (st.select_s - st0.select_s) / per_pass);
    std::fprintf(stderr, "CPU per pass: reading misses %.1f ms (%.1f GB/s), waiting for the GPU %.1f ms\n",
                 1e3 * (st.read_s - st0.read_s) / per_pass,
                 (st.upload_bytes - st0.upload_bytes) / 1e9 / std::max(1e-9, st.read_s - st0.read_s),
                 1e3 * (st.wait_s - st0.wait_s) / per_pass);
    if (st.cpu_experts > st0.cpu_experts || st.fills > st0.fills)
        std::fprintf(stderr,
                     "Misses per pass: %.1f GPU direct from RAM, %.1f CPU in %.1f ms, %.1f refilled in %.1f ms\n",
                     (st.gpu_direct - st0.gpu_direct) / per_pass, (st.cpu_experts - st0.cpu_experts) / per_pass,
                     1e3 * (st.cpu_s - st0.cpu_s) / per_pass, (st.fills - st0.fills) / per_pass,
                     1e3 * (st.fill_s - st0.fill_s) / per_pass);
    if (st.dropped > st0.dropped)
        std::fprintf(stderr, "Dropped per pass (TONA_MISS_DROP): %.1f\n", (st.dropped - st0.dropped) / per_pass);
    if (st.prefetched > st0.prefetched)
        std::fprintf(
            stderr, "Prefetch per pass: %.1f experts, of which needed %.1f %%, wait %.2f ms (%lld checked)\n",
            (st.prefetched - st0.prefetched) / per_pass,
            100.0 * (st.prefetch_used - st0.prefetch_used) / std::max<int64_t>(1, st.prefetched - st0.prefetched),
            1e3 * (st.prefetch_wait_s - st0.prefetch_wait_s) / per_pass,
            (long long)(st.prefetch_checked - st0.prefetch_checked));
}

void SpecRun::report() {
    report_summary();
    if (env_.probe > 0) report_probe();
    std::fprintf(stderr, "Projection without expert reloads (seed pass times): %.1f tok/s decode\n",
                 (double)(out_.size() - 1) / (model_ms_ / 1e3));
    if (env::flag("TONA_DFLASH_CALIB"))
        for (int i = 0; i < n_draft_; i++)
            for (const auto& [p, ok] : calib_[i]) std::fprintf(stderr, "calib %d %.4f %d\n", i, p, ok);
    if (!env_.cpu)
        std::fprintf(stderr, "Drafter: kernel %.2f ms, selector (CPU) %.2f ms per pass\n",
                     1e3 * t_draft_kernel_ / std::max(1, passes_), 1e3 * t_select_ / std::max(1, passes_));
    report_experts();
    if (env::flag("TONA_MEGA_PROF"))
        print_profile(stderr, "Time per pass in the megakernel (all passes averaged)", mg_);
}

}  // namespace tona::gen
