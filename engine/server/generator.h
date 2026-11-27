// Generation for the server: megakernel with speculation (greedy or sampled), prompt in passes of 4 tokens.
// Reuse across requests:
// - if the new prompt continues the accepted tokens, generation continues there;
// - otherwise the checkpoint from the end of the previous prompt is loaded if the new prompt starts with it (agents
//   resend the whole history, and the answer is often tokenized differently the second time);
// - otherwise a conversation stashed in RAM whose prompt end the new prompt continues; the running conversation is
//   stashed first. Up to TONA_PROMPT_SLOTS conversations (default 16 including the running one; usually bounded by
//   TONA_PROMPT_RAM_GB). Eviction takes what was never fetched again first, then the longest unused (after Marconi,
//   arXiv 2411.19379);
// - otherwise from scratch (the running conversation is stashed).
// Intermediate checkpoint: if at least 4,096 tokens would have to be computed before the start of the last message, a
// checkpoint is stashed there. With TONA_PROMPT_CACHE=<dir> checkpoints also go to the SSD and survive a restart
// (TONA_PROMPT_CACHE_GB, default 16; the oldest drop out). Meant for requests with a long fixed start and a changing
// end: Claude Code's auto-mode reviewer sends ~30K tokens of rules, then the growing action log in one message.
#pragma once

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "backend/backend.h"
#include "backend/env.h"
#include "cpu/dflash.h"
#include "model/model_base.h"
#include "model/tokenizer.h"
#include "server/chat_format.h"
#include "server/sampling.h"
#include "server/suffix.h"
#include "spec/draft_calib.h"

namespace tona::server {

enum class Stop { eos, length, callback };

struct PrefillInfo {
    size_t reused = 0;    // tokens that were not recomputed
    size_t computed = 0;  // freshly computed tokens
    double seconds = 0;
    int slot = 0;          // 0: running conversation, 1: fetched from RAM, -1: new
    double swap_ms = 0;    // stashing and fetching
    size_t ckpt = 0;       // position of the checkpoint created by this request (0: none)
    bool tail = false;     // the last one of them is the one before the fixed end of the prompt
    double disk_ms = 0;    // checkpoint read from the SSD (0: not)
    size_t done = 0;       // warm-up: prompt tokens the state covers (all unless a real request interrupted it)
    bool aborted = false;  // warm-up: interrupted at a slice boundary
};

// A running generation of one conversation: state between passes, so that several conversations can run in one pass
// (Generator::advance). The fields belong to the generator.
struct Stream {
    int slot = 0;  // slot in the megakernel
    int max_tokens = 0;
    std::function<bool(int32_t)> on_token;
    Sampling smp;
    double weight = 1;  // priority when drafts are shared across conversations
    bool greedy = true, qs = false, done = false;
    Stop why = Stop::length;
    std::mt19937_64 rng;
    int32_t last = -1;
    int n = 0, think_left = -1;
    std::deque<int32_t> forced;
    // Pass: block (last + drafts), distributions q of the drafts, raw and calibrated probabilities
    std::vector<int32_t> blk, sfx;
    std::vector<Dist> qd;
    std::vector<double> raw_q, pc;
    std::vector<float> probs;
    bool from_sfx = false, forcing = false, fixed = false;  // fixed: do not shorten the block (history, forcing, probe)
    int nd = 0;
    double ms_drf = 0;
    spec::DraftCalib<7> calib[2], mcal[2];
    int64_t probe = 0;
    bool full = true;
    int64_t passes = 0, gen_tokens = 0, sfx_passes = 0, sfx_tokens = 0;
    int prev_n = 3;  // block size and accepted + 1 in the last pass: expectation for the other conversations
    double prev_e = 2.5;
    // Running MTP head chain (chain_*): proposal, expectation, best rate and length, step, limit
    backend::MtpPick ch_pk;
    double ch_e = 1, ch_run = 1, ch_best_rate = 0, ch_w = 1, ch_eo = 0;
    int ch_best = 0, ch_i = 0, ch_nmax = 0, ch_no = 0;
    bool ch_probe = false, ch_multi = false;
    std::chrono::steady_clock::time_point ch_t0;
};

class Generator {
public:
    // dfl: DFlash2 drafter; nullptr: drafts come from the MTP head in mg
    Generator(const ModelBase& m, backend::Backend& mg, ref::DFlash* dfl, const Tokenizer& tok, const ChatFormat& chat);

    // Compute the prompt to the end; afterwards the logits of the last token are ready
    PrefillInfo prefill(const std::vector<int32_t>& prompt);
    // Warm-up (max_tokens 0): the same, but nothing is generated. Checkpoints at the head and at the end of the prompt
    // go to the SSD. stop is asked at every slice boundary; true ends the computation (a real request is waiting).
    PrefillInfo prefill_warm(const std::vector<int32_t>& prompt, std::function<bool()> stop);
    // Up to max_tokens tokens, on_token(id) for each; false ends. End tokens are not reported.
    Stop generate(int max_tokens, const std::function<bool(int32_t)>& on_token, const Sampling& smp = {});
    // The same in steps, for several conversations: begin right after prefill (in the same slot), then advance over
    // all running streams until each is done. One pass per advance; with several streams each drafts in its own slot,
    // the drafts are shared across all (at most BMAX tokens per pass) and verified in one pass.
    std::unique_ptr<Stream> begin(int max_tokens, std::function<bool(int32_t)> on_token, const Sampling& smp = {},
                                  double weight = 1);
    void advance(const std::vector<Stream*>& live);
    // Pick and activate a slot for a new prompt: free (not in busy), with the longest matching start (running tokens
    // or checkpoint), otherwise empty, otherwise the longest unused
    void select_slot(const std::vector<int32_t>& prompt, const std::vector<int>& busy);
    int seqs() const { return mg_.seqs(); }
    void use_slot(int s) { mg_.use_slot(s); }
    bool idle_tick() { return mg_.vram_tick(); }  // engine thread, no request running
    // Sliced prefill: if yield is set, prefill computes long prompts in slices of TONA_PREFILL_SLICE tokens (default
    // 4096) and calls yield in between so other conversations keep running; afterwards the same slot is active
    void set_yield(std::function<void()> f) { yield_ = std::move(f); }

    int64_t max_ctx() const { return max_ctx_; }
    // After an exception: reset the megakernel; the next prefill computes from scratch
    void recover() {
        mg_.recover();
        pending_ = 0;
    }
    double tokens_per_pass() const { return passes_ ? (double)gen_tokens_ / passes_ : 0; }
    // Counters since start: passes, generated tokens, passes with a draft from the history and their tokens
    int64_t passes() const { return passes_; }
    int64_t gen_tokens() const { return gen_tokens_; }
    int64_t suffix_passes() const { return sfx_passes_; }
    int64_t suffix_tokens() const { return sfx_tokens_; }
    // for /status, only from the engine thread
    const backend::Backend& mega() const { return mg_; }
    backend::Backend& backend() { return mg_; }
    int stashed() const { return (int)stash_.size(); }
    int on_disk() const { return (int)disk_.size(); }

private:
    // One stashed conversation (RAM) and one checkpoint file (SSD)
    struct Stashed {
        backend::HostSnap snap;
        std::vector<float> logits;
        uint64_t used = 0;
        bool ckpt = false;  // checkpoint (start of the last message or before the fixed end)
        bool tail = false;  // checkpoint before the fixed end: a longer one that continues it replaces it
        int hits = 0;       // times fetched; an end that continues an older one inherits its count
    };
    // Checkpoints on the SSD: head and tokens of each file stay in memory, the rest is read on demand
    struct OnDisk {
        std::string path;
        std::vector<int32_t> tokens;
        bool logits = true;  // false: a checkpoint inside a prompt, useless for exactly this prompt again
    };
    // Where a new prompt continues from: pick -1 scratch, 0 running tokens, 1 checkpoint, 2 stashed conversation
    struct Source {
        int pick = -1;
        size_t stash = 0, len = 0;
    };
    // Positions for intermediate checkpoints (0: none)
    struct CheckpointPlan {
        size_t tail = 0, fork = 0;
    };

    // --- setup (generator.cpp)
    void init_stop_tokens(const Tokenizer& tok, const ChatFormat& chat);
    void read_env();
    void reset_calib();  // calibration back to the pre-fitted start values

    // --- prompt cache and prefill (prompt_cache.cpp, prefill.cpp)
    // How many tokens run_prompt computed (fewer only when abort_ ended a warm-up) and whether the state has the logits
    // of the last one (a checkpoint inside the prompt has none)
    struct RunResult {
        size_t done;
        bool logits;
    };
    // The n tokens like split_run (run_split.h) says: prefill, then passes of 4 tokens through forward writing the
    // drafter context along. final = false: a checkpoint inside the prompt, no forward token
    RunResult run_prompt(const int32_t* ids, size_t n, bool final = true, bool force = false);
    size_t prefill_slices(const int32_t* ids, size_t m);
    size_t warm_cut(const std::vector<int32_t>& prompt) const;
    void store_warm(PrefillInfo& info, bool logits);
    Source find_source(const std::vector<int32_t>& prompt) const;
    // The logits of the last token of a state: without them it can only continue a longer prompt
    bool usable(size_t len, bool logits, const std::vector<int32_t>& prompt) const {
        return logits || len < prompt.size();
    }
    CheckpointPlan plan_checkpoints(const std::vector<int32_t>& prompt, size_t best_len) const;
    void load_disk_checkpoint(const std::vector<int32_t>& prompt, Source& src, PrefillInfo& info);
    size_t restore_source(const Source& src, PrefillInfo& info);
    size_t restore_chosen(const Source& src, PrefillInfo& info);
    size_t last_message_start(const std::vector<int32_t>& prompt, size_t from) const;
    size_t checkpoint_at(const std::vector<int32_t>& prompt, size_t from, size_t at, bool tail, PrefillInfo& info);
    size_t write_checkpoints(const std::vector<int32_t>& prompt, size_t from, const CheckpointPlan& plan,
                             PrefillInfo& info);
    void stash_current(bool ckpt = false, bool tail = false);
    void trim_stash();
    bool evict_one();  // the least valuable entry except keep_; false if there is none
    double stash_held_bytes(bool evictable = false) const;
    bool can_stash(double bytes);  // false: a stash of this size must not be made (backend::stash_fits), logged once
    void make_room(double bytes);  // drops older entries until it fits
    void disk_scan();
    void disk_write(const Stashed& s);
    void disk_prune(const std::string& keep_path);
    bool disk_read(const OnDisk& d, Stashed& s) const;
    std::vector<float>& snap_logits() { return snap_logits_[mg_.slot()]; }

    // --- drafting (draft.cpp)
    // Block for the next pass (at most cap tokens); with several conversations: n_o tokens and w * E of the others.
    // defer: false if s began an MTP head chain (chain_*) that advance finishes
    bool draft(Stream& s, int cap, bool multi, int n_o = 0, double e_o = 0, bool defer = false);
    void draft_forced(Stream& s, int n_draft);
    void draft_from_history(Stream& s, int n_draft);
    bool draft_mtp(Stream& s, int n_draft, bool multi, int n_o, double e_o, bool defer);
    void draft_dflash(Stream& s);
    void catch_up_drafter();  // feed the drafter the features of the last pass
    // DFlash2 or an MTP head is present; a model without either runs one token per pass and feeds nothing
    bool has_drafter() const { return dfl_ || mg_.has_mtp(); }
    // MTP head chain: begin after the catch-up, push evaluates the proposal (true: continue with s.blk.back()), end
    // shortens. Several conversations take their steps together (mtp_step_multi)
    void chain_begin(Stream& s, bool multi, int n_o, double e_o);
    bool chain_push(Stream& s);
    void chain_draw(Stream& s);
    void chain_end(Stream& s);
    void allocate(const std::vector<Stream*>& go);  // share drafts across the conversations

    // --- passes (verify.cpp, generator.cpp)
    bool pre(Stream& s);  // emit last; false: done
    bool is_stop(int32_t t) const { return std::find(stops_.begin(), stops_.end(), t) != stops_.end(); }
    void post(Stream& s, const float* lg, double ms_fw);  // verify, commit, emit
    int accept_drafts(Stream& s, const float* lg, int32_t& next);
    int accept_sampled(Stream& s, const float* lg, int32_t& next);
    void log_pass(const Stream& s, int accepted, double ms_fw) const;
    void update_block_size(Stream& s, int accepted);
    bool emit_accepted(Stream& s, int accepted);
    void finish(Stream& s);
    void others_expectation(const std::vector<Stream*>& go, size_t i, int& n_o, double& e_o) const;
    void run_chains(const std::vector<Stream*>& chains, size_t n_go, int cap,
                    std::chrono::steady_clock::time_point t_start);
    void forward_and_verify(const std::vector<Stream*>& go);
    void trim_fixed_block(Stream& s, int cap);
    std::vector<Stream*> begin_pass(const std::vector<Stream*>& live);
    std::vector<Stream*> draft_all(const std::vector<Stream*>& go, int cap);
    void count_think(Stream& s, int32_t token);

    const ModelBase& m_;
    backend::Backend& mg_;
    ref::DFlash* dfl_;
    int64_t max_ctx_;
    std::vector<float> last_logits_, snap_logits_[backend::SMAX];  // snap: belongs to the checkpoint, per slot
    std::vector<int32_t> stops_;                                   // end tokens
    std::vector<int32_t> think_force_;  // closing text when the thinking budget is reached, ends with </think>
    uint64_t slot_used_[backend::SMAX] = {};
    // Pass time per width (ms) and drafter; start values from research/measurements/2026-10-23-verify-pass.md, used to
    // pick the width; widths 5-8 relative to 4 from research/measurements/2026-10-25-width-8.md (5 int8, 6-8 k_wide)
    double t_pass_[9] = {0, 22.9, 31.3, 37.0, 41.7, 50.6, 57.3, 62.6, 67.6}, t_drf_ = 6.3;
    int64_t probe_ = 0;
    // Acceptance probabilities of the drafts, calibrated online (TONA_DRAFT_CALIB=0: raw). One per mode, because with
    // sampling a draft is accepted only with p(d) instead of always when it is the argmax: [0] greedy, [1] sampling
    spec::DraftCalib<7> calib_[2];
    // MTP: own calibration (start: raw q), cost per MTP step and of the catch-up in ms (TONA_MTP_STEP_MS,
    // TONA_MTP_FILL_MS; fixed, so the width depends on the request only)
    spec::DraftCalib<7> mcal_[2];
    // Verify pass per width with MTP in ms, 1-8 tokens: batch bench with real experts, mean of 3 rounds
    // (research/measurements/2026-11-12-verify-pass.md, Results 3; before: 2026-10-26-mtp.md)
    double t_step_ = 1.15, t_fill_ = 1.3, t_mpass_[9] = {0, 23.7, 28.9, 32.5, 35.9, 41.7, 46.1, 49.6, 52.6};
    // Verify pass with two conversations (more distinct experts): widths 4-8 measured
    // (research/measurements/2026-10-26-parallel.md), 2-3 as above * 1.12
    double t_xpass_[9] = {0, 24.2, 33.0, 37.1, 41.7, 48.9, 54.0, 58.4, 63.0};
    int64_t passes_ = 0, gen_tokens_ = 0, sfx_passes_ = 0, sfx_tokens_ = 0;
    SuffixIndex sfx_[backend::SMAX];  // per slot
    std::vector<Stashed> stash_;      // stashed conversations
    // At most this much RAM for stashed conversations (backend::prompt_ram_gb): with a 256K context up to ~3 GB each,
    // next to 25 GB of pinned experts
    double ram_gb_ = 12;
    uint64_t clock_ = 0, keep_ = 0;  // keep_: mark (used) of the entry stash_current must not delete
    std::string disk_dir_;
    double disk_gb_ = 16;
    uint64_t fingerprint_ = 0;
    std::vector<OnDisk> disk_;
    std::function<void()> yield_;
    std::function<bool()> abort_;  // warm-up only: true ends the computation at the next slice boundary
    // The 4-byte and 1-byte members last, so the class has no padding between members (clang-analyzer Padding)
    int32_t think_end_ = -1;  // </think>
    int pending_ = 0;         // tokens of the last pass the drafter has not been fed yet
    // Slots including the running one (TONA_PROMPT_SLOTS; 1: only the running one). 4 were too few: 6 conversations in
    // turn took 97 s instead of 4.7 s of prefill (research/measurements/2026-10-26-conversations-16.md)
    int slots_ = 16;
    int32_t im_start_ = -1;  // token <|im_start|>: message boundaries
    // Draft from the history instead of the drafter if the end of the text occurred before with this many tokens
    // (TONA_SUFFIX_MIN, default 8; 0: off)
    int sfx_min_ = 8;
    int pf_min_ = 32;  // prompts with more new tokens go through the prefill (TONA_PREFILL_MIN)
    int pf_slice_ = 4096;
    bool full_ = true;  // the last pass accepted all drafts: next drafter block is large
    // Measured without gain, off by default (TONA_MTP_MULTI=1 turns it on)
    bool mtp_multi_ = tona::env::flag("TONA_MTP_MULTI");
    bool stash_skip_logged_ = false;  // the "not stashed" reason is logged once
    bool aborted_ = false;
};

}  // namespace tona::server
