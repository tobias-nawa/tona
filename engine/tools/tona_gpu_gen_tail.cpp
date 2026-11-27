// --teacher with TONA_TEACHER_TAIL=N (--mega): the text goes through the prefill path except for its last N tokens,
// which run through the megakernel (decode path) teacher-forced. A paired comparison of two builds or settings of the
// decode program (e.g. TONA_IDX_SHARE) needs the logits of the reference run:
//   TONA_TEACHER_DUMP=FILE: write per token -log p (float) and the logits row (float x vocab).
//   TONA_TEACHER_REF=FILE: read such a file; TONA_TEACHER_OUT=FILE gets one line per token
//   "0 pos nll_ref nll_here KL(ref||here) top1_equal", the format of scripts/nll-compare.py.
//   TONA_TEACHER_NLL=FILE: one line "pos target_id nll top1_hit" per token, no reference run needed (retrieval tails:
//   the NLL of the answer tokens only).
// TONA_TEACHER_NB=k: k tokens per pass, all accepted (the verify pass).
// TONA_TEACHER_SPLIT=p: the head is not computed in one prefill call but with a checkpoint at token p, as a request
// continues from a warm-up or a fork checkpoint (server/prefill.cpp): the first p tokens like a prompt of their own,
// the rest from there. Compared with the unsplit run it measures what the other partition of the prefill costs in
// quality.
// TONA_TEACHER_CUTS=p1,p2,...: the head goes through the prefill in pieces cut at these tokens (ascending), with no
// forward token at the cuts: what a prefix checkpoint does. The logits of the tail must equal those of the
// unsplit run bit for bit under TONA_DETERMINISTIC=1, whatever the pieces are.
#include <cstdio>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>

#include "backend/env.h"
#include "tona_gpu_gen.h"

namespace tona::gen {

namespace {

struct File {
    FILE* f = nullptr;
    File(const char* path, const char* mode) : f(path ? std::fopen(path, mode) : nullptr) {
        if (path && !f) throw std::runtime_error(std::string("cannot open: ") + path);
    }
    ~File() {
        if (f) std::fclose(f);
    }
    File(const File&) = delete;
    File& operator=(const File&) = delete;
};

struct Pair {
    double nll_ref = 0, kl = 0;
    bool same_top1 = false;
};

// KL(ref || here) in nats, from two logit rows
double kl_rows(const std::vector<float>& ref, const float* here, size_t n) {
    auto lse = [n](const float* g) {
        double m = *std::max_element(g, g + n), s = 0;
        for (size_t i = 0; i < n; i++) s += std::exp(g[i] - m);
        return m + std::log(s);
    };
    const double lr = lse(ref.data()), lh = lse(here);
    double kl = 0;
    for (size_t i = 0; i < n; i++) {
        const double lp = ref[i] - lr;
        kl += std::exp(lp) * (lp - (here[i] - lh));
    }
    return kl;
}

class Tail {
public:
    Tail(Session& s, const std::vector<int32_t>& ids, int tail)
        : s_(s),
          ids_(ids),
          head_((int)ids.size() - 1 - tail),
          vocab_(s.model.hp().vocab),
          dump_(env::text("TONA_TEACHER_DUMP"), "wb"),
          ref_(env::text("TONA_TEACHER_REF"), "rb"),
          out_(env::text("TONA_TEACHER_OUT"), "w"),
          nll_out_(env::text("TONA_TEACHER_NLL"), "w") {}

    int run() {
        if (head_ < 1) throw std::runtime_error("TONA_TEACHER_TAIL: the text is shorter than the tail");
        const auto t0 = Clock::now();
        const int split = (int)env::integer("TONA_TEACHER_SPLIT", 0);
        if (split > 0 && split < head_)
            prefill_split(split);
        else
            prefill_cuts();
        const double t_pre = seconds_since(t0);
        const auto t1 = Clock::now();
        const int width = (int)env::integer("TONA_TEACHER_NB", 1);
        for (int i = head_; i + 1 < (int)ids_.size();) {
            const int nb = std::min(width, (int)ids_.size() - 1 - i);
            const float* logits = s_.mg->forward(ids_.data() + i, nb);
            for (int b = 0; b < nb; b++) add(i + b, logits + b * vocab_);
            s_.mg->commit(nb);
            i += nb;
        }
        std::printf(
            "tail: prefill %d tokens in %.1f s, then %d tokens through the megakernel (%.1f tok/s): NLL %.6f, top-1 "
            "%.2f %%",
            head_, t_pre, n_, n_ / seconds_since(t1), nll_ / n_, 100.0 * hits_ / n_);
        if (ref_.f)
            std::printf(", vs reference: dNLL %+.6f, KL %.6f, top-1 equal %.2f %%", (nll_ - nll_ref_) / n_, kl_ / n_,
                        100.0 * same_ / n_);
        std::printf("\n");
        return 0;
    }

private:
    // The first `split` tokens as run_prompt does (all but the last through the prefill, the last through forward),
    // then the rest of the head from that state
    void prefill_split(int split) {
        s_.mg->prefill(ids_.data(), split - 1);
        s_.mg->forward(ids_.data() + split - 1, 1);
        s_.mg->commit(1);
        s_.mg->prefill(ids_.data() + split, head_ - split);
    }

    // The head in pieces cut at TONA_TEACHER_CUTS (none: one call)
    void prefill_cuts() {
        int at = 0;
        std::stringstream list(env::text("TONA_TEACHER_CUTS") ? env::text("TONA_TEACHER_CUTS") : "");
        for (std::string item; std::getline(list, item, ',');) {
            const int cut = std::stoi(item);
            if (cut <= at || cut >= head_) throw std::runtime_error("TONA_TEACHER_CUTS: ascending, inside the head");
            s_.mg->prefill(ids_.data() + at, cut - at);
            at = cut;
        }
        s_.mg->prefill(ids_.data() + at, head_ - at);
    }

    void add(int i, const float* logits) {
        const int32_t next = ids_[i + 1];
        const double nll = neg_log_prob(logits, vocab_, next);
        const bool hit = argmax(logits, vocab_) == next;
        nll_ += nll, hits_ += hit, n_++;
        if (nll_out_.f) std::fprintf(nll_out_.f, "%d %d %.6f %d\n", i, next, nll, (int)hit);
        if (dump_.f) {
            const float v = (float)nll;
            std::fwrite(&v, sizeof v, 1, dump_.f);
            std::fwrite(logits, sizeof(float), vocab_, dump_.f);
        }
        if (!ref_.f) return;
        float rn = 0;
        row_.resize(vocab_);
        if (std::fread(&rn, sizeof rn, 1, ref_.f) != 1 ||
            std::fread(row_.data(), sizeof(float), vocab_, ref_.f) != vocab_)
            throw std::runtime_error("TONA_TEACHER_REF: file ends early");
        const double kl = kl_rows(row_, logits, vocab_);
        const bool same = argmax(row_.data(), vocab_) == argmax(logits, vocab_);
        nll_ref_ += rn, kl_ += kl, same_ += same;
        if (out_.f) std::fprintf(out_.f, "0 %d %.7f %.7f %.8f %d\n", i, rn, nll, kl, (int)same);
    }

    Session& s_;
    const std::vector<int32_t>& ids_;
    const int head_;
    const size_t vocab_;
    File dump_, ref_, out_, nll_out_;
    std::vector<float> row_;
    double nll_ = 0, nll_ref_ = 0, kl_ = 0;
    int n_ = 0, hits_ = 0, same_ = 0;
};

}  // namespace

int run_teacher_tail(Session& s, const std::vector<int32_t>& ids, int tail) {
    if (!s.mg) throw std::runtime_error("TONA_TEACHER_TAIL needs --mega");
    return Tail(s, ids, tail).run();
}

}  // namespace tona::gen
