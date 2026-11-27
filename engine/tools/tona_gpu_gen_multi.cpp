// --multi-check N (--mega, TONA_SEQS >= 2): two conversations in separate slots, each first alone, then together in
// shared passes (OEA: experts both need are shared). Compares the logits row by row, also with swapped slots, and
// reports how much sharing costs in quality.
// TONA_MULTI_NB=k: up to k tokens per conversation and pass (default 2). TONA_MULTI_B=FILE: second conversation from
//   this file instead of the shifted prompt. TONA_MULTI_SWAPS=1: only slots 0/1 (the swapped run checks only that
//   the slots are equal). TONA_MULTI_HALF=1: keep only accepted rows, in f16 (automatic above 3000 tokens, else
//   ~36 GB at 10,000 tokens); the bitwise comparison is then meaningless. TONA_MULTI_DUMP=FILE: per accepted row
//   "conversation position nll_alone nll_together KL top1_equal" for scripts/nll-compare.py. The reference without
//   OEA is TONA_OEA_K0=0.
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include "backend/env.h"
#include "tona_gpu_gen.h"

namespace tona::gen {

namespace {

// Tokens fed and tokens accepted per conversation in one pass (0: the conversation is not in this pass).
struct PassPlan {
    int nt[2];
    int acc[2];
};

struct Alone {
    std::vector<std::vector<float>> rows;
    std::vector<std::vector<_Float16>> rows_half;
};

// One row of a shared pass next to its solo reference.
struct RowPair {
    int g;     // conversation
    int b;     // row within the conversation's part of the pass
    int swap;  // 0: slots 0/1, 1: swapped
    const PassPlan* plan;
    const float* together;
    const float* alone;
};

struct Metrics {
    size_t row[2] = {0, 0};
    int pos[2] = {0, 0};
    int same = 0, top = 0, cmp = 0, first = -1;
    double worst = 0;
    double nll_alone[2] = {0, 0}, nll_together[2] = {0, 0}, kl_sum[2] = {0, 0};
    int nll_count[2] = {0, 0}, top_accepted[2] = {0, 0};
};

// KL(p_alone || p_together) over the whole vocabulary
double kl_divergence(const float* a, const float* b, size_t n) {
    const float ma = *std::max_element(a, a + n), mb = *std::max_element(b, b + n);
    double sa = 0, sb = 0;
    for (size_t i = 0; i < n; i++) sa += std::exp((double)(a[i] - ma)), sb += std::exp((double)(b[i] - mb));
    const double la = std::log(sa) + ma, lb = std::log(sb) + mb;
    double k = 0;
    for (size_t i = 0; i < n; i++) {
        const double pa = std::exp((double)a[i] - la);
        k += pa * (((double)a[i] - la) - ((double)b[i] - lb));
    }
    return std::max(0.0, k);
}

class MultiCheck {
public:
    MultiCheck(Session& s, const Options& o, const std::vector<int32_t>& ids)
        : s_(s), mg_(*s.mg), vocab_(s.model.hp().vocab), n_((int)std::min<size_t>(ids.size(), o.multi_check)) {
        load_conversations(ids);
        make_plan();
        half_ = n_ > 3000 || env::flag("TONA_MULTI_HALF");
        if (const char* path = env::text("TONA_MULTI_DUMP")) dump_ = std::fopen(path, "w");
    }
    ~MultiCheck() {
        if (dump_) std::fclose(dump_);
    }
    MultiCheck(const MultiCheck&) = delete;
    MultiCheck& operator=(const MultiCheck&) = delete;

    int run() {
        const int swaps = (int)env::integer("TONA_MULTI_SWAPS", 2);
        for (int swap = 0; swap < swaps; swap++) run_swap(swap);
        return 0;
    }

private:
    void load_conversations(const std::vector<int32_t>& ids) {
        if (mg_.seqs() < 2) throw std::runtime_error("--multi-check needs TONA_SEQS >= 2");
        x_[0].assign(ids.begin(), ids.begin() + n_);
        if (const char* path = env::text("TONA_MULTI_B")) {
            std::ifstream in(path);
            const std::string txt((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
            x_[1] = s_.tok.encode(txt);
            if ((int)x_[1].size() < n_) throw std::runtime_error("TONA_MULTI_B: too short");
            x_[1].resize(n_);
        } else {
            for (int i = 0; i < n_; i++) x_[1].push_back(ids[(i + n_ / 2) % n_]);
        }
    }

    void make_plan() {
        const int w = (int)env::integer("TONA_MULTI_NB", 2);
        for (int k = 0, p0 = 0, p1 = 0; p0 < n_ || p1 < n_; k++) {
            PassPlan p{};
            p.nt[0] = std::min(1 + k % w, n_ - p0), p.nt[1] = std::min(w - k % w, n_ - p1);
            for (int g = 0; g < 2; g++) p.acc[g] = p.nt[g] ? std::min(p.nt[g], 1 + (k / 3 + g) % p.nt[g]) : 0;
            p0 += p.acc[0], p1 += p.acc[1];
            plan_.push_back(p);
        }
    }

    // Conversation g alone in slot `slot`: logits per pass and row.
    Alone run_alone(int g, int slot) {
        Alone out;
        mg_.use_slot(slot);
        mg_.reset();
        int pos = 0;
        for (const PassPlan& p : plan_) {
            if (!p.nt[g]) continue;
            const float* lg = mg_.forward(x_[g].data() + pos, p.nt[g]);
            for (int b = 0; b < p.nt[g]; b++) {
                if (!half_)
                    out.rows.emplace_back(lg + b * vocab_, lg + (b + 1) * vocab_);
                else if (b < p.acc[g])
                    out.rows_half.emplace_back(lg + b * vocab_, lg + (b + 1) * vocab_);
            }
            mg_.commit(p.acc[g]);
            pos += p.acc[g];
        }
        return out;
    }

    const float* reference_row(const Alone& a, size_t& row) {
        if (!half_) return a.rows[row++].data();
        const auto& h = a.rows_half[row++];
        for (size_t i = 0; i < vocab_; i++) rbuf_[i] = (float)h[i];
        return rbuf_.data();
    }

    void score_quality(Metrics& m, const RowPair& r) {
        const int g = r.g;
        const int32_t next = x_[g][m.pos[g] + r.b + 1];
        const double na = neg_log_prob(r.alone, vocab_, next), nm = neg_log_prob(r.together, vocab_, next);
        const double k = kl_divergence(r.alone, r.together, vocab_);
        const bool t1 = argmax(r.together, vocab_) == argmax(r.alone, vocab_);
        m.nll_alone[g] += na, m.nll_together[g] += nm, m.nll_count[g]++, m.kl_sum[g] += k, m.top_accepted[g] += t1;
        if (dump_ && r.swap == 0)
            std::fprintf(dump_, "%d %d %.6f %.6f %.8f %d\n", g, m.pos[g] + r.b, na, nm, k, (int)t1);
    }

    void score_row(Metrics& m, const RowPair& r) {
        if (m.pos[r.g] + r.b + 1 < n_ && r.b < r.plan->acc[r.g]) score_quality(m, r);
        const bool eq = !std::memcmp(r.together, r.alone, vocab_ * sizeof(float));
        m.same += eq;
        if (!eq && m.first < 0) m.first = m.cmp;
        m.worst = std::max(m.worst, relative_deviation(r.together, r.alone, vocab_));
        m.top += argmax(r.together, vocab_) == argmax(r.alone, vocab_);
        m.cmp++;
    }

    // One shared pass: the rows of both conversations that take part, compared with their solo logits.
    void run_pass(Metrics& m, const PassPlan& p, const int slot[2], const Alone* ref[2], int swap) {
        int slots[2], nt[2], k = 0;
        std::vector<int32_t> tokens;
        for (int g = 0; g < 2; g++)
            if (p.nt[g]) {
                slots[k] = slot[g], nt[k++] = p.nt[g];
                tokens.insert(tokens.end(), x_[g].begin() + m.pos[g], x_[g].begin() + m.pos[g] + p.nt[g]);
            }
        const float* lg = mg_.forward_multi(slots, nt, k, tokens.data());
        size_t r = 0;
        for (int g = 0; g < 2; g++) {
            for (int b = 0; b < p.nt[g]; b++, r++) {
                if (half_ && b >= p.acc[g]) continue;
                const float* rf = reference_row(*ref[g], m.row[g]);
                score_row(m, {g, b, swap, &p, lg + r * vocab_, rf});
            }
            if (p.nt[g]) {
                mg_.use_slot(slot[g]);
                mg_.commit(p.acc[g]);
                m.pos[g] += p.acc[g];
            }
        }
    }

    void run_swap(int swap) {
        const int slot[2] = {swap, 1 - swap};
        const auto t_alone0 = Clock::now();
        const Alone alone0 = run_alone(0, slot[0]);
        const Alone alone1 = run_alone(1, slot[1]);
        const Alone* ref[2] = {&alone0, &alone1};
        rbuf_.assign(vocab_, 0.f);
        const double t_alone = seconds_since(t_alone0);
        for (int sl : slot) mg_.use_slot(sl), mg_.reset();
        Metrics m;
        const auto st0 = mg_.stats();
        const auto t_multi0 = Clock::now();
        for (const PassPlan& p : plan_) run_pass(m, p, slot, ref, swap);
        report(m, slot, st0, t_alone, seconds_since(t_multi0));
        if (dump_ && swap == 0) std::fflush(dump_);
    }

    void report(const Metrics& m, const int slot[2], const tona::backend::Stats& st0, double t_alone, double t_multi) {
        std::printf(
            "Conversations in slots %d/%d: %zu passes, %d rows, bitwise equal %d, top-1 equal %d, first "
            "deviation row %d, worst relative deviation %.2e; alone %.2f s, together %.2f s\n",
            slot[0], slot[1], plan_.size(), m.cmp, m.same, m.top, m.first, m.worst, t_alone, t_multi);
        const auto& st1 = mg_.stats();
        const double passes_layers = (double)std::max<int64_t>(1, st1.uni_multi_n - st0.uni_multi_n);
        const int n0 = std::max(1, m.nll_count[0]), n1 = std::max(1, m.nll_count[1]);
        std::printf(
            "  -log p per token alone / together: A %.4f / %.4f, B %.4f / %.4f (%d, %d tokens); distinct experts "
            "per layer in the joint pass %.1f; OEA: %.2f foreign, %.2f empty slots per pass and layer\n",
            m.nll_alone[0] / n0, m.nll_together[0] / n0, m.nll_alone[1] / n1, m.nll_together[1] / n1, m.nll_count[0],
            m.nll_count[1], (st1.uni_multi - st0.uni_multi) / passes_layers,
            (st1.oea_swapped - st0.oea_swapped) / passes_layers, (st1.oea_short - st0.oea_short) / passes_layers);
        std::printf("  KL(alone || together) per token: A %.5f, B %.5f; top-1 equal A %.2f %%, B %.2f %%\n",
                    m.kl_sum[0] / n0, m.kl_sum[1] / n1, 100.0 * m.top_accepted[0] / n0, 100.0 * m.top_accepted[1] / n1);
    }

    Session& s_;
    tona::gpu::Mega& mg_;
    const size_t vocab_;
    const int n_;
    std::vector<int32_t> x_[2];
    std::vector<PassPlan> plan_;
    bool half_ = false;
    FILE* dump_ = nullptr;
    std::vector<float> rbuf_;
};

}  // namespace

int run_multi_check(Session& s, const Options& o, const std::vector<int32_t>& ids) {
    return MultiCheck(s, o, ids).run();
}

}  // namespace tona::gen
