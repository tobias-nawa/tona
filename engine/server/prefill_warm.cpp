// Warm-up prefill (max_tokens 0): compute a prompt like a request would, store the checkpoints, generate nothing
#include <chrono>
#include <format>
#include <stdexcept>

#include "backend/budget.h"
#include "server/generator.h"

namespace tona::server {

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

}  // namespace

// The follow-up request continues the messages of the warm-up, not its end: the template closes the last message
// (<|im_end|>\n) and opens the answer (<|im_start|>assistant\n...), tokens the follow-up does not have. So the
// head ends before the closing of the last message, minus the seam margin of the fork checkpoint (16 tokens: the
// last words of the head can be merged differently with what follows). 0: no such place.
size_t Generator::warm_cut(const std::vector<int32_t>& prompt) const {
    constexpr size_t margin = 16, closing = 2;
    if (im_start_ < 0) return 0;
    for (size_t i = prompt.size(); i-- > 0;)
        if (prompt[i] == im_start_) return i > closing + margin ? i - closing - margin : 0;
    return 0;
}

// Stores the state at the end of the computed part. On the SSD only: a RAM copy of every warm-up would push the
// conversations in use out of the stash. Without an SSD directory the stash is the only place for it.
void Generator::store_warm(PrefillInfo& info, bool logits) {
    mg_.save_state();
    snap_logits() = logits ? last_logits_ : std::vector<float>{};
    const auto ts = std::chrono::steady_clock::now();
    if (disk_dir_.empty()) {
        stash_current(true);
        trim_stash();
    } else if (mg_.snap_tokens().size() >= 1024) {  // as stash_current: shorter ones are recomputed in under 1 s
        // The copy is transient but as large as a stash: same rule
        if (!backend::stash_fits(mg_.stash_bytes(), ram_gb_, backend::free_ram_bytes(), 0)) return;
        Stashed s;
        s.ckpt = true;
        mg_.stash(s.snap);
        s.logits = snap_logits();
        disk_write(s);
    } else {
        return;
    }
    info.swap_ms += ms_since(ts);
    info.ckpt = mg_.snap_tokens().size();
}

// Stages: the head (warm_cut) and the end of the prompt. A real request that arrives meanwhile stops the
// computation at the next slice boundary (abort_); what was computed so far is kept if it is at least one slice,
// so a later warm-up of the same prompt continues there.
PrefillInfo Generator::prefill_warm(const std::vector<int32_t>& prompt, std::function<bool()> stop) {
    const auto t0 = std::chrono::steady_clock::now();
    if ((int64_t)prompt.size() >= max_ctx_)
        throw std::runtime_error(std::format("prompt of {} tokens, context {}", prompt.size(), max_ctx_));
    if (prompt.empty()) throw std::runtime_error("empty prompt");
    if (pending_ && dfl_) mg_.draft(pending_, nullptr, 0), pending_ = 0;
    struct Reset {
        Generator& g;
        ~Reset() { g.abort_ = nullptr, g.aborted_ = false; }
    } reset{*this};
    abort_ = std::move(stop), aborted_ = false;
    Source src = find_source(prompt);
    PrefillInfo info;
    load_disk_checkpoint(prompt, src, info);
    const size_t from = restore_source(src, info);
    info.reused = from;
    size_t pos = from;
    const size_t cut = warm_cut(prompt);
    bool logits = true;  // the state has the logits of its last token
    for (const size_t at : {cut > from ? cut : 0, prompt.size()}) {
        if (!at) continue;
        if (at > pos) {
            // The head is a checkpoint inside the prompt (prefill only, see checkpoint_at); the end is a whole prompt.
            // After a checkpoint even a short rest stays in the prefill, as in a cold run
            const RunResult r = run_prompt(prompt.data() + pos, at - pos, at == prompt.size(), pos > 0);
            pos += r.done, logits = r.logits;
        } else {
            last_logits_ = snap_logits();  // the whole prompt was already there
            logits = !last_logits_.empty();
        }
        if (aborted_ && pos - from < (size_t)pf_slice_) break;
        store_warm(info, logits);
        if (aborted_) break;
    }
    info.done = pos, info.computed = pos - from, info.aborted = aborted_;
    info.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    return info;
}

}  // namespace tona::server
