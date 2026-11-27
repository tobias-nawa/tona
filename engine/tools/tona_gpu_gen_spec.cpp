// --dflash: speculation with the DFlash2 drafter on the GPU (or as CPU reference), greedy, N draft tokens per pass.
// Prints the generated text on stdout, statistics on stderr.
// Variables: see SpecEnv in tona_gpu_gen_spec.h. The draft length per pass follows the expected tokens per time
// (selector probabilities, pass times without reloading); TONA_DFLASH_FIXED=1 always uses N. TONA_DFLASH_EMA=1
// measures pass times with a moving average; that picks worse when experts reload, because one expensive pass
// deters the choice for good.
#include "tona_gpu_gen_spec.h"

#include <stdexcept>

#include "backend/env.h"

namespace tona::gen {

namespace {

// Pass time per width in ms, without expert reloads (research/measurements/2026-10-23-verify-pass.md); widths 5-8
// relative to width 4 from research/measurements/2026-10-25-width-8.md (5 int8, 6-8 k_wide with TONA_WIDE_MIN=6).
constexpr double kSeedPassMs[9] = {0, 22.9, 31.3, 37.0, 41.7, 50.6, 57.3, 62.6, 67.6};
constexpr double kSeedDrafterMs = 6.3;
constexpr int kMaxDraft = 7;
constexpr double kEmaWeight = 0.1;

}  // namespace

SpecEnv SpecEnv::read() {
    SpecEnv e{};
    e.cpu = env::flag("TONA_DFLASH_CPU");
    e.check = !e.cpu && env::flag("TONA_DFLASH_CHECK");
    e.ema = env::flag("TONA_DFLASH_EMA");
    e.adaptive = !env::flag("TONA_DFLASH_FIXED");
    e.prompt_width = (int)env::integer("TONA_PROMPT_NB", 4);
    e.probe = (int)env::integer("TONA_DRAFT_PROBE", 0);
    // TONA_DRAFT_SMALL=k: the drafter block of n_draft + 1 only after a pass that accepted all drafts, else k + 1
    // (the block of 8 tokens costs 8.8 instead of 5.3 ms, research/measurements/2026-10-25-width-8.md)
    e.small_block = (int)env::integer("TONA_DRAFT_SMALL", 0);
    // TONA_DRAFT_BIG_AT=k: the big block already when the last pass accepted k drafts (default: all)
    e.big_at = (int)env::integer("TONA_DRAFT_BIG_AT", 0);
    return e;
}

SpecRun::SpecRun(Session& s, const Options& o, const std::vector<int32_t>& ids)
    : s_(s),
      mg_(*s.mg),
      drafter_(*s.drafter),
      o_(o),
      ids_(ids),
      env_(SpecEnv::read()),
      vocab_(s.model.hp().vocab),
      d_(s.model.hp().d),
      feature_size_(s.drafter->target_layers().size() * s.model.hp().d),
      im_end_(s.tok.find("<|im_end|>")),
      n_draft_(o.n_draft),
      last_logits_(vocab_),
      embeddings_((size_t)(o.n_draft + 1) * d_),
      drafter_ms_(kSeedDrafterMs) {
    if (n_draft_ > kMaxDraft) throw std::runtime_error("--draft at most 7");
    std::copy(kSeedPassMs, kSeedPassMs + kWidths, pass_ms_);
}

// The prompt in passes of prompt_width tokens; the drafter context is written along.
void SpecRun::feed_prompt() {
    const auto t0 = Clock::now();
    size_t i0 = 0;
    if (o_.use_prefill && ids_.size() > 1) mg_.prefill(ids_.data(), (int)ids_.size() - 1), i0 = ids_.size() - 1;
    for (size_t i = i0; i < ids_.size();) {
        const int nb = (int)std::min<size_t>(env_.prompt_width, ids_.size() - i);
        const float* lg = mg_.forward(ids_.data() + i, nb);
        std::copy(lg + (size_t)(nb - 1) * vocab_, lg + (size_t)nb * vocab_, last_logits_.begin());
        mg_.commit(nb);
        if (env_.cpu || env_.check)
            for (int k = 0; k < nb; k++) drafter_.inject(mg_.features() + k * feature_size_, (int64_t)(i + k));
        if (!env_.cpu) mg_.draft(nb, nullptr, 0);
        i += nb;
    }
    t_prompt_ = seconds_since(t0);
}

void SpecRun::draft_on_cpu(std::vector<int32_t>& block) {
    for (int32_t t : drafter_.draft(last_, pos_, n_draft_ + 1)) block.push_back(t);
    // TONA_DRAFT_PROBE=n: an extra draft of n tokens (own block, attention within the block is not causal), compared
    // with the output at the end: acceptance at wider verify passes
    if (env_.probe > 0) probes_.emplace_back(out_.size() - 1, drafter_.draft(last_, pos_, env_.probe + 1));
}

// k drafts give E = 1 + sum_j prod_{i<=j} p_i tokens in a pass of width k + 1 plus the drafter; take the k with the
// best tokens per time.
void SpecRun::shorten_by_expected_rate(std::vector<int32_t>& block, int n_drafts) {
    int best = 0;
    double e = 1, run = 1, best_rate = 1 / (pass_ms_[1] + drafter_ms_);
    for (int k = 1; k <= n_drafts; k++) {
        run *= probs_[k - 1], e += run;
        const double rate = e / (pass_ms_[k + 1] + drafter_ms_);
        if (rate > best_rate) best_rate = rate, best = k;
    }
    block.resize(1 + best);
}

void SpecRun::draft_on_gpu(std::vector<int32_t>& block, int n_drafts) {
    drafter_.embed(last_, embeddings_.data());
    for (int i = 1; i <= n_drafts; i++) drafter_.embed(drafter_.mask_token(), embeddings_.data() + i * d_);
    const auto ta = Clock::now();
    mg_.draft(pending_, embeddings_.data(), n_drafts + 1);
    const auto tb = Clock::now();
    pending_ = 0;
    for (int32_t t : drafter_.select(last_, mg_.draft_logits(), mg_.draft_gate(), n_drafts + 1, &probs_))
        block.push_back(t);
    t_draft_kernel_ += seconds_between(ta, tb);
    t_select_ += seconds_since(tb);
    if (env_.adaptive) shorten_by_expected_rate(block, n_drafts);
    if (env_.check) {
        const auto ref = drafter_.draft(last_, pos_, n_drafts + 1);
        for (int i = 0; i + 1 < (int)block.size(); i++) same_draft_ += ref[i] == block[i + 1], cmp_draft_++;
    }
}

int SpecRun::count_accepted(const float* logits, const std::vector<int32_t>& block) const {
    int a = 0;
    while (a + 1 < (int)block.size()) {
        if (argmax(logits + a * vocab_, vocab_) != block[a + 1]) break;
        a++;
    }
    return a;
}

void SpecRun::record_calibration(int accepted) {
    for (int i = 0; i < (int)probs_.size() && i <= accepted && i < kCalibPositions; i++)
        calib_[i].emplace_back(probs_[i], i < accepted);
}

void SpecRun::inject_features(int count) {
    for (int i = 0; i < count; i++) drafter_.inject(mg_.features() + i * feature_size_, pos_ + i);
}

// Accepted drafts up to the end of the turn; what the pass accepted after it is not printed. False: the turn ended.
bool SpecRun::emit(const std::vector<int32_t>& block, int accepted, int32_t next, size_t width) {
    bool ended = false;
    for (int i = 1; i <= accepted && !ended; i++) {
        out_.push_back(block[i]), std::printf("%s", s_.tok.decode(block[i]).c_str());
        ended = !o_.ignore_eos && (block[i] == s_.tok.eos() || block[i] == im_end_);
    }
    if (ended) {
        passes_++;
        return false;
    }
    out_.push_back(next);
    std::printf("%s", s_.tok.decode(next).c_str());
    std::fflush(stdout);
    pos_ += accepted + 1;
    last_ = next;
    passes_++;
    hist_[accepted]++;
    width_[width]++;
    return true;
}

bool SpecRun::one_pass() {
    const auto t1 = Clock::now();
    const int n_drafts = env_.small_block > 0 && !full_ ? std::min(env_.small_block, n_draft_) : n_draft_;
    std::vector<int32_t> block{last_};
    if (n_draft_ > 0 && env_.cpu)
        draft_on_cpu(block);
    else if (n_draft_ > 0)
        draft_on_gpu(block, n_drafts);
    const auto t2 = Clock::now();
    const float* lg = mg_.forward(block.data(), (int)block.size());
    const size_t width = block.size();
    const int a = count_accepted(lg, block);
    full_ = env_.big_at > 0 ? a >= env_.big_at : a + 1 == (int)block.size() && block.size() > 1;
    record_calibration(a);
    const int32_t next = argmax(lg + a * vocab_, vocab_);
    mg_.commit(a + 1);
    const auto t3 = Clock::now();
    if (env_.cpu || env_.check) inject_features(a + 1);
    if (!env_.cpu) pending_ = a + 1;
    const auto t4 = Clock::now();
    const double drafter_s = seconds_between(t1, t2) + seconds_between(t3, t4);
    t_draft_ += drafter_s;
    t_target_ += seconds_between(t2, t3);
    model_ms_ += kSeedPassMs[width] + (n_draft_ > 0 ? kSeedDrafterMs : 0);
    if (env_.ema) {
        pass_ms_[width] += kEmaWeight * (1e3 * seconds_between(t2, t3) - pass_ms_[width]);
        if (!env_.cpu && n_draft_ > 0) drafter_ms_ += kEmaWeight * (1e3 * drafter_s - drafter_ms_);
    }
    return emit(block, a, next, width);
}

int SpecRun::run() {
    feed_prompt();
    stats0_ = mg_.stats();
    disk0_ = disk_read_mb();
    last_ = argmax(last_logits_);
    pos_ = (int64_t)ids_.size();
    out_.push_back(last_);
    std::printf("%s%s", o_.prompt, s_.tok.decode(last_).c_str());
    std::fflush(stdout);
    while ((int)out_.size() < o_.n_gen && (o_.ignore_eos || (last_ != s_.tok.eos() && last_ != im_end_)))
        if (!one_pass()) break;
    std::printf("\n");
    report();
    return 0;
}

int run_speculative(Session& s, const Options& o, const std::vector<int32_t>& ids) { return SpecRun(s, o, ids).run(); }

}  // namespace tona::gen
