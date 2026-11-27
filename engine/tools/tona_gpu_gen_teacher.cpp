// --teacher: teacher-forced pass over a text file, reports negative log-likelihood and top-1 hit rate.
// TONA_TEACHER_NB=k (--mega): k tokens per pass, all accepted (the verify pass as in speculation).
// TONA_TEACHER_NLL=FILE with --prefill: one line "pos nll" per position (-log p of the next token) for paired
//   comparisons of prefill settings; the logits of the prefill are not kept, so no KL and no top-1.
// TONA_TEACHER_ACC=1: accept 1, 2, 3, 4 tokens in turn instead; the rest is discarded and recomputed, as with wrong
//   drafts.
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <stdexcept>

#include "backend/env.h"
#include "tona_gpu_gen.h"

namespace tona::gen {

namespace {

std::string read_text(const char* path) {
    FILE* f = std::fopen(path, "rb");
    if (!f) throw std::runtime_error(std::string("cannot read: ") + path);
    std::string text;
    char buf[65536];
    for (size_t n; (n = std::fread(buf, 1, sizeof buf, f)) > 0;) text.append(buf, n);
    std::fclose(f);
    return text;
}

struct Tally {
    double nll = 0;
    int hits = 0;
    int n = 0;

    void add(const float* logits, size_t vocab, int32_t next) {
        nll += neg_log_prob(logits, vocab, next);
        hits += argmax(logits, vocab) == next;
        n++;
    }
};

// --check N with --prefill: -log p of the first N positions from the CPU reference (f32, token by token) against
// the prefill's: mean signed and mean absolute difference and the largest; (the step-by-step --check never runs
// the prefill kernels)
void compare_prefill_with_reference(Session& s, const std::vector<int32_t>& ids, const std::vector<float>& v, int n) {
    n = std::min<int>(n, (int)v.size());
    double sum = 0, sum_abs = 0, worst = 0, nll_cpu = 0;
    for (int i = 0; i < n; i++) {
        const auto lg = s.reference->step(ids[i]);
        const double c = neg_log_prob(lg.data(), lg.size(), ids[i + 1]);
        const double d = v[i] - c;
        sum += d, sum_abs += std::fabs(d), worst = std::max(worst, std::fabs(d)), nll_cpu += c;
    }
    std::printf(
        "check against the CPU reference over %d positions: NLL prefill %.6f, CPU %.6f; -log p prefill - CPU: "
        "mean %+.6f, mean abs %.6f, max abs %.6f\n",
        n, std::accumulate(v.begin(), v.begin() + n, 0.0) / n, nll_cpu / n, sum / n, sum_abs / n, worst);
}

// Prefill path: one call yields the NLL of every position.
int teacher_prefill(Session& s, const std::vector<int32_t>& ids, int check) {
    if (!s.mg) throw std::runtime_error("--prefill needs --mega");
    const auto t0 = Clock::now();
    std::vector<float> v;
    s.mg->prefill(ids.data(), (int)ids.size(), &v);
    const double dt = seconds_since(t0);
    double nll = 0;
    for (float x : v) nll += x;
    const int n = (int)v.size();
    if (const char* path = env::text("TONA_TEACHER_NLL")) {
        FILE* f = std::fopen(path, "w");
        if (!f) throw std::runtime_error(std::string("cannot write: ") + path);
        for (int i = 0; i < n; i++) std::fprintf(f, "%d %.6f\n", i, v[i]);
        std::fclose(f);
    }
    if (check > 0 && s.reference) compare_prefill_with_reference(s, ids, v, check);
    std::printf("prefill (chunks of %d, %s): %d tokens, NLL %.9f, perplexity %.3f (%.1f tok/s prefill)\n",
                s.mg->prefill_last_chunk(), s.mg->prefill_last_group() ? "layer by layer" : "chunk by chunk", n,
                nll / n, std::exp(nll / n), ids.size() / dt);
    return 0;
}

// Passes of `width` tokens through the verify pass; `cycle` accepts 1..4 of them in turn.
void teacher_batched(Session& s, const std::vector<int32_t>& ids, int width, bool cycle, Tally& t) {
    const size_t vocab = s.model.hp().vocab;
    for (size_t i = 0, k = 0; i + 1 < ids.size(); k++) {
        const int nb = (int)std::min<size_t>(width, ids.size() - 1 - i);
        const float* logits = s.mg->forward(ids.data() + i, nb);
        const int accepted = cycle ? std::min(nb, 1 + (int)(k % 4)) : nb;
        for (int b = 0; b < accepted; b++) t.add(logits + b * vocab, vocab, ids[i + b + 1]);
        s.mg->commit(accepted);
        i += accepted;
    }
}

void teacher_stepwise(Session& s, const std::vector<int32_t>& ids, Tally& t) {
    for (size_t i = 0; i + 1 < ids.size(); i++) {
        const auto& logits = s.step(ids[i]);
        t.add(logits.data(), logits.size(), ids[i + 1]);
    }
}

const char* path_label(const Session& s) {
    if (s.mg) return env::flag("TONA_MEGA_F32") ? "mega float" : "mega int8";
    return env::flag("TONA_GPU_INT8") ? "int8" : "float";
}

}  // namespace

int run_teacher(Session& s, const Options& o) {
    const std::string text = read_text(o.teacher);
    // TONA_TEACHER_SPECIAL=1: control tokens in the text (<|im_start|>, <|im_end|>) become their IDs, so a rendered
    // chat prompt reaches the model as the server sends it
    auto ids = s.tok.encode(text, env::flag("TONA_TEACHER_SPECIAL"));
    if ((int)ids.size() > o.max_teacher + 1) ids.resize(o.max_teacher + 1);
    if (const int tail = (int)env::integer("TONA_TEACHER_TAIL", 0); tail > 0) return run_teacher_tail(s, ids, tail);
    const auto t0 = Clock::now();
    if (o.use_prefill) return teacher_prefill(s, ids, o.check);
    const int width = (int)env::integer("TONA_TEACHER_NB", 1);
    Tally t;
    if (s.mg && width > 1)
        teacher_batched(s, ids, width, env::flag("TONA_TEACHER_ACC"), t);
    else
        teacher_stepwise(s, ids, t);
    const double dt = seconds_since(t0);
    std::printf("%s: %d tokens, NLL %.9f, perplexity %.3f, top-1 %.1f %% (%.1f tok/s teacher-forced)\n", path_label(s),
                t.n, t.nll / t.n, std::exp(t.nll / t.n), 100.0 * t.hits / t.n, t.n / dt);
    return 0;
}

}  // namespace tona::gen
