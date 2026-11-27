// Plain greedy generation without speculation, optionally checked step by step against the CPU reference (--check N).
// The generated text goes to stdout, the statistics to stderr.
#include <cstdio>

#include "backend/env.h"
#include "tona_gpu_gen.h"

namespace tona::gen {

namespace {

class PlainRun {
public:
    PlainRun(Session& s, const Options& o) : s_(s), o_(o), im_end_(s.tok.find("<|im_end|>")) {}

    int run(const std::vector<int32_t>& ids) {
        const auto t0 = Clock::now();
        feed_prompt(ids);
        const double t_prompt = seconds_since(t0);
        const StatsView st0 = s_.stats();
        std::printf("%s", o_.prompt);
        std::fflush(stdout);
        const double t_gen = generate();
        std::printf("\n");
        report(ids.size(), t_prompt, t_gen, st0);
        return 0;
    }

private:
    // Single steps, or passes of o_.prompt_nb tokens (all accepted). Only the logits after the last token are kept.
    void feed_prompt(const std::vector<int32_t>& ids) {
        if (o_.prompt_nb <= 1) {
            for (int32_t id : ids) {
                logits_ = s_.step(id);
                compare(logits_, id);
            }
            return;
        }
        const size_t vocab = s_.model.hp().vocab;
        for (size_t i = 0; i < ids.size();) {
            const int nb = (int)std::min<size_t>(o_.prompt_nb, ids.size() - i);
            const float* lg = s_.mg->forward(ids.data() + i, nb);
            logits_.assign(lg + (size_t)(nb - 1) * vocab, lg + (size_t)nb * vocab);
            s_.mg->commit(nb);
            i += nb;
        }
    }

    // Logits of the GPU against the reference for the first o_.check steps.
    void compare(const std::vector<float>& gpu, int32_t id) {
        if (!s_.reference || checked_ >= o_.check) return;
        const auto ref = s_.reference->step(id);
        const Deviation dev = deviation(gpu.data(), ref.data(), ref.size());
        worst_ = std::max(worst_, dev.relative());
        same_top1_ += argmax(gpu) == argmax(ref);
        std::fprintf(stderr, "  step %d: max |d| %.4f (relative %.2e), top-1 GPU %d CPU %d\n", checked_, dev.max_diff,
                     dev.relative(), argmax(gpu), argmax(ref));
        checked_++;
    }

    // Returns the time spent in decode steps.
    double generate() {
        double t_gen = 0;
        for (; generated_ < o_.n_gen; generated_++) {
            const int next = argmax(logits_);
            std::printf("%s", s_.tok.decode(next).c_str());
            std::fflush(stdout);
            if ((next == s_.tok.eos() || next == im_end_) && !o_.ignore_eos) break;
            const auto t1 = Clock::now();
            logits_ = s_.step(next);
            t_gen += seconds_since(t1);
            compare(logits_, next);
        }
        return t_gen;
    }

    void report(size_t prompt_tokens, double t_prompt, double t_gen, const StatsView& st0) {
        const StatsView st = s_.stats();
        const double hits = (double)(st.expert_hits - st0.expert_hits),
                     miss = (double)(st.expert_misses - st0.expert_misses);
        std::fprintf(stderr, "Prompt %zu tokens in %.2f s; generation %d tokens in %.2f s = %.2f tok/s decode\n",
                     prompt_tokens, t_prompt, generated_, t_gen, generated_ ? generated_ / t_gen : 0.0);
        std::fprintf(stderr, "Experts during generation: %.1f %% hits, %.0f MB reloaded\n",
                     100.0 * hits / std::max(1.0, hits + miss), (st.upload_bytes - st0.upload_bytes) / 1e6);
        if (s_.mg) report_megakernel();
        if (checked_)
            std::fprintf(stderr, "Check: %d steps, top-1 equal in %d, worst relative deviation %.2e\n", checked_,
                         same_top1_, worst_);
    }

    void report_megakernel() const {
        if (env::flag("TONA_MEGA_PROF"))
            print_profile(stderr, "Time per token in the megakernel (mean over all tokens, including the prompt)",
                          *s_.mg);
        const auto dg = s_.mg->diag();
        std::fprintf(stderr,
                     "Megakernel: %lld launches, barrier: %lld aborts, start skew max %.1f µs, wait max %.1f µs\n",
                     (long long)s_.mg->stats().launches, (long long)dg.aborts, dg.start_skew_us, dg.max_wait_us);
    }

    Session& s_;
    const Options& o_;
    const int32_t im_end_;
    std::vector<float> logits_;
    int checked_ = 0;
    int same_top1_ = 0;
    int generated_ = 0;
    double worst_ = 0;
};

}  // namespace

int run_plain(Session& s, const Options& o, const std::vector<int32_t>& ids) { return PlainRun(s, o).run(ids); }

}  // namespace tona::gen
