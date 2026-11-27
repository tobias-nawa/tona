// Prompt processing: pick the best source to continue from, compute the rest, create intermediate checkpoints
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <stdexcept>

#include "backend/env.h"
#include "backend/nan_guard.h"
#include "backend/stall.h"
#include "server/generator.h"
#include "server/run_split.h"

namespace tona::server {

namespace {

using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// p is the start of the prompt or the prompt itself
bool prefix_of(const std::vector<int32_t>& p, const std::vector<int32_t>& prompt) {
    return !p.empty() && p.size() <= prompt.size() && std::equal(p.begin(), p.end(), prompt.begin());
}

bool strict_prefix_of(const std::vector<int32_t>& p, const std::vector<int32_t>& prompt) {
    return p.size() < prompt.size() && std::equal(p.begin(), p.end(), prompt.begin());
}

}  // namespace

// The first m tokens through the prefill kernels, in slices when other work waits (yield_) or a warm-up may give way
// (abort_); returns how many it computed (fewer only when abort_ stopped it)
size_t Generator::prefill_slices(const int32_t* ids, size_t m) {
    const size_t slice = (yield_ || abort_) && pf_slice_ > 0 ? (size_t)pf_slice_ : m;
    size_t i = 0;
    while (i < m) {
        const size_t k = std::min(slice, m - i);
        mg_.prefill(ids + i, (int)k);
        i += k;
        if (i < m && yield_) {
            const int s = mg_.slot();
            yield_();
            mg_.use_slot(s);
        }
        // A warm-up gives way at a slice boundary
        if (i < m && abort_ && abort_()) {
            aborted_ = true;
            break;
        }
    }
    return i;
}

// Computes the n tokens by split_run: the prefill kernels for the part it names, the rest through forward. Without
// `final` (a checkpoint inside the prompt) everything goes through the prefill and no logits remain: the caller stores
// the state without them. After an interrupted warm-up the state also has no logits: the computed slices are the
// checkpoint, no forward token is added
Generator::RunResult Generator::run_prompt(const int32_t* ids, size_t n, bool final, bool force) {
    const size_t V = m_.dims().vocab;
    static const bool dbg = tona::env::flag("TONA_PF_DEBUG");
    auto t0 = Clock::now();
    auto ms = [&] {
        const double v = ms_since(t0);
        t0 = Clock::now();
        return v;
    };
    const RunSplit sp = split_run(n, final, force, (size_t)pf_min_, mg_.prefill_chunk() > 0);
    size_t done = 0;
    if (sp.prefill) {
        done = prefill_slices(ids, sp.prefill);
        if (dbg)
            std::fprintf(stderr, "run_prompt: prefill %zu tokens %.1f ms (MTP included: %.1f ms)\n", done, ms(),
                         mg_.stats().mtp_pf_s * 1e3);
        if (done < sp.prefill) return {done, false};
    }
    ids += done;
    for (size_t i = 0; i < sp.forward;) {
        const int nb = (int)std::min<size_t>(4, sp.forward - i);
        const float* lg = mg_.forward(ids + i, nb);
        std::copy(lg + (nb - 1) * V, lg + nb * V, last_logits_.begin());
        mg_.commit(nb);
        if (dfl_)
            mg_.draft(nb, nullptr, 0);
        else if (mg_.has_mtp())
            mg_.mtp_fill(-1);
        i += nb;
    }
    if (dbg) std::fprintf(stderr, "run_prompt: remainder through forward %.1f ms\n", ms());
    return {done + sp.forward, sp.forward > 0};
}

// Candidates: running tokens, checkpoint, stashed conversations; the longest match wins
Generator::Source Generator::find_source(const std::vector<int32_t>& prompt) const {
    Source src;
    if (!mg_.tokens().empty() && strict_prefix_of(mg_.tokens(), prompt)) src.pick = 0, src.len = mg_.tokens().size();
    const size_t snap_len = mg_.has_state() ? mg_.snap_tokens().size() : 0;
    if (snap_len > src.len && prefix_of(mg_.snap_tokens(), prompt) &&
        usable(snap_len, !snap_logits_[mg_.slot()].empty(), prompt))
        src.pick = 1, src.len = snap_len;
    for (size_t i = 0; i < stash_.size(); i++)
        if (stash_[i].snap.tokens.size() > src.len && prefix_of(stash_[i].snap.tokens, prompt) &&
            usable(stash_[i].snap.tokens.size(), !stash_[i].logits.empty(), prompt))
            src.pick = 2, src.len = stash_[i].snap.tokens.size(), src.stash = i;
    return src;
}

// Two kinds of intermediate checkpoint besides the one at the start of the last message.
// Fixed end behind growing content (the auto-mode reviewer: rules, action log, then the same question; new actions
// come before the question): if the prompt deviates from a stashed one only far behind what can be reused, the rest
// of the old one (tail) is probably the fixed end. A checkpoint `tail` tokens before the end fits next time. Before,
// the state (it cannot be rewound) was stashed only at the start of the last message and the whole log was
// recomputed every time (~11,500 tokens, 9 s).
// Fork: a long common start with another conversation and separate ways afterwards (Claude Code in another project:
// same system prompt up to the working directory; a reset conversation). A checkpoint there saves the next new
// conversation its start, also across a restart (SSD).
Generator::CheckpointPlan Generator::plan_checkpoints(const std::vector<int32_t>& prompt, size_t best_len) const {
    size_t lcp = 0, tail = 0;
    auto consider = [&](const std::vector<int32_t>& t) {
        const size_t n = std::min(t.size(), prompt.size());
        const size_t l = std::mismatch(t.begin(), t.begin() + n, prompt.begin()).first - t.begin();
        if (l > lcp && l < t.size()) lcp = l, tail = t.size() - l;
    };
    for (const Stashed& e : stash_) consider(e.snap.tokens);
    for (int s = 0; s < mg_.seqs(); s++) consider(mg_.snap_tokens_of(s));  // also conversations in other slots
    constexpr size_t margin = 16;  // tokens at the seam can be merged differently
    CheckpointPlan plan;
    // From 1,024 lost tokens (~0.8 s) a new checkpoint pays off (stashing takes ~60-180 ms); so it moves along
    if (slots_ >= 2 && lcp >= best_len + 1024 && tail <= 8192 && prompt.size() > tail + margin)
        plan.tail = prompt.size() - tail - margin;
    if (slots_ >= 2 && lcp >= best_len + 2048 && lcp > margin) plan.fork = lcp - margin;
    if (plan.tail && plan.fork + 1024 > plan.tail) plan.tail = 0;  // same place: the one at the fork stays
    return plan;
}

// A checkpoint from the SSD, if it matches longer than everything in RAM: take it in as a stashed conversation
void Generator::load_disk_checkpoint(const std::vector<int32_t>& prompt, Source& src, PrefillInfo& info) {
    const OnDisk* od = nullptr;
    for (const OnDisk& d : disk_)
        if (d.tokens.size() > src.len && prefix_of(d.tokens, prompt) && usable(d.tokens.size(), d.logits, prompt))
            od = &d, src.len = d.tokens.size();
    if (!od) return;
    const auto ts = Clock::now();
    Stashed got;
    if (disk_read(*od, got)) {
        got.ckpt = true, got.used = ++clock_;
        stash_.push_back(std::move(got));
        src.pick = 2, src.stash = stash_.size() - 1;
        info.disk_ms = ms_since(ts);
        return;
    }
    // Broken or foreign: delete it, otherwise it is read again with every request
    fprintf(stderr, "Checkpoint unusable, deleted: %s\n", od->path.c_str());
    std::error_code ec;
    std::filesystem::remove(od->path, ec);
    const std::string bad = od->path;
    std::erase_if(disk_, [&](const OnDisk& x) { return x.path == bad; });
}

// Brings the megakernel to the chosen source; returns how many prompt tokens it already holds. A restore whose copy
// stalled (backend/stall.h) leaves the slot invalid: the prompt is computed from the start instead of failing.
size_t Generator::restore_source(const Source& src, PrefillInfo& info) {
    try {
        return restore_chosen(src, info);
    } catch (const backend::CopyStalled& e) {
        fprintf(stderr, "Restore failed: %s; the prompt is computed from the start\n", e.what());
        keep_ = 0;
        info.slot = -1;
        mg_.reset();  // the stalled copy has ended (CopyStalled is thrown only then), so this cannot hang on it
        return 0;
    }
}

size_t Generator::restore_chosen(const Source& src, PrefillInfo& info) {
    if (src.pick == 0) return mg_.tokens().size();
    if (src.pick == 1) {
        mg_.load_state();  // end of the previous prompt
        return mg_.tokens().size();
    }
    const auto ts = Clock::now();
    if (src.pick == 2) {
        // The chosen entry stays stashed (checkpoints are needed again); marked as used last, so trim_stash evicts it
        // last
        keep_ = stash_[src.stash].used = ++clock_;  // mark: stash_current moves and deletes entries
        stash_[src.stash].hits++;
        stash_current();
        const Stashed& got =
            *std::find_if(stash_.begin(), stash_.end(), [&](const Stashed& s) { return s.used == keep_; });
        mg_.unstash(got.snap);
        snap_logits() = got.logits;
        keep_ = 0;
        trim_stash();
        info.slot = 1;
        info.swap_ms = ms_since(ts);
        return mg_.tokens().size();
    }
    stash_current();
    trim_stash();
    info.swap_ms = ms_since(ts);
    info.slot = -1;
    if (mg_.pos() > 0 || mg_.has_state()) mg_.reset();
    return 0;
}

// Start of the last message: the second-to-last <|im_start|> (the last one opens the answer); 0: none
size_t Generator::last_message_start(const std::vector<int32_t>& prompt, size_t from) const {
    if (im_start_ < 0 || slots_ < 2) return 0;
    int seen = 0;
    for (size_t i = prompt.size(); i-- > from;)
        if (prompt[i] == im_start_ && ++seen == 2) return i;
    return 0;
}

// Computes the prompt up to `at`, stashes a checkpoint there and returns the new `from`. All of it goes through the
// prefill, as in a cold run of the whole prompt, so a hit on the checkpoint computes the same bits as that run.
// The state has no logits: it continues longer prompts only
size_t Generator::checkpoint_at(const std::vector<int32_t>& prompt, size_t from, size_t at, bool tail,
                                PrefillInfo& info) {
    run_prompt(prompt.data() + from, at - from, false);
    mg_.save_state();
    snap_logits().clear();
    const auto ts = Clock::now();
    stash_current(true, tail);
    if (!tail)  // the one before the fixed end moves with every request, it does not go to the SSD
        for (const Stashed& e : stash_)
            if (e.ckpt && e.snap.tokens.size() == at) disk_write(e);
    trim_stash();
    info.swap_ms += ms_since(ts);
    info.ckpt = at;
    return at;
}

// Ascending, otherwise a later checkpoint prevents the earlier ones (the state cannot be rewound)
size_t Generator::write_checkpoints(const std::vector<int32_t>& prompt, size_t from, const CheckpointPlan& plan,
                                    PrefillInfo& info) {
    struct Ck {
        size_t at, min;
        bool tail;
    };
    std::array<Ck, 3> cks{
        {{plan.fork, 2048, false}, {last_message_start(prompt, from), 4096, false}, {plan.tail, 1024, true}}};
    std::sort(cks.begin(), cks.end(), [](const Ck& a, const Ck& b) { return a.at < b.at; });
    for (const Ck& c : cks)
        if (c.at && c.at >= from + c.min) {
            from = checkpoint_at(prompt, from, c.at, c.tail, info);
            if (c.tail) info.tail = true;
        }
    return from;
}

PrefillInfo Generator::prefill(const std::vector<int32_t>& prompt) {
    tona::nan::new_epoch();  // a request: every NaN site may report once more
    const auto t0 = Clock::now();
    if ((int64_t)prompt.size() >= max_ctx_)
        throw std::runtime_error(std::format("prompt of {} tokens, context {}", prompt.size(), max_ctx_));
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (pending_ && dfl_) mg_.draft(pending_, nullptr, 0), pending_ = 0;
    auto secs = [&] { return std::chrono::duration<double>(Clock::now() - t0).count(); };
    Source src = find_source(prompt);
    const CheckpointPlan plan = plan_checkpoints(prompt, src.len);
    PrefillInfo info;
    load_disk_checkpoint(prompt, src, info);
    // Behind a checkpoint the rest stays in the prefill however short it is: a cold run of the whole prompt prefilled
    // it
    const bool after_ckpt = src.pick == 2 && stash_[src.stash].ckpt;
    size_t from = restore_source(src, info);
    info.reused = from;
    if (from == prompt.size()) {  // same prompt again: nothing to compute
        last_logits_ = snap_logits();
        info.seconds = secs();
        return info;
    }
    const size_t computed = prompt.size() - from;
    const size_t written = write_checkpoints(prompt, from, plan, info);
    run_prompt(prompt.data() + written, prompt.size() - written, true, after_ckpt || written > from);
    mg_.save_state();
    snap_logits() = last_logits_;
    info.computed = computed;
    info.seconds = secs();
    return info;
}

}  // namespace tona::server
