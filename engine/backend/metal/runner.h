// Sequences (slots) and their passes: the pass width nb, commit with rollback, the save point and stashes in RAM.
// commit(m < nb) restores the recurrent state from the backup taken before the pass and computes the m accepted tokens
// again without the LM head; KV rows behind the position are simply overwritten later.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <vector>

#include "backend/backend.h"
#include "backend/metal/gpu_gaps.h"
#include "backend/metal/mtp_vocab.h"
#include "backend/metal/pass.h"
#include "backend/metal/profile.h"
#include "backend/metal/seq.h"
#include "backend/metal/stop_wait.h"
#include "backend/mtp_track.h"

namespace tona::metal {

struct Slot {
    SeqState st;
    std::vector<int32_t> tokens, pass, snap_tokens;  // accepted, of the pass awaiting commit, of the save point
    int64_t pos = 0, snap_pos = 0;
    bool snap = false;
    // MTP: which entries are computed, the last pass and the scan of the dynamic rows (backend/mtp_track.h); its last
    // h (hout); hlast: the target's h at mtp.hlast_pos, kept for the start of the next fill
    backend::MtpTrack mtp;
    std::vector<float> hout, hlast;
    std::vector<int32_t> ahead;  // tokens of the next target pass known so far (the fill's next, the drafts stepped)
};

struct RunnerOptions {
    int64_t max_ctx = 0;
    int seqs = 1, chunk = 0;       // slots; tokens per prefill pass (0: prefill through forward)
    const MtpHead* mtp = nullptr;  // the MTP head (its tensors already in the Weights) or nullptr
};

struct MtpCounts {
    int64_t passes = 0, stale = 0;
};

class Runner {
public:
    Runner(Device& dev, Weights& w, const Qwen4ExpModel& m, const RunnerOptions& o);
    ~Runner();
    Runner(const Runner&) = delete;
    Runner& operator=(const Runner&) = delete;
    // Passes of the given slots one after the other (tokens back to back), logits [sum n][vocab]
    const float* forward(const int* slots, const int* n, int k, const int32_t* tok, bool backup);
    void commit(int m);
    // n tokens of the current slot through the prefill kernels, accepted at once; nll: -log p of each next token
    void prefill(const int32_t* tok, int n, std::vector<float>* nll);
    int chunk() const { return chunk_; }
    void reset();
    void save_state();
    void load_state();
    void stash(backend::HostSnap& s) const;
    size_t stash_bytes() const;
    void unstash(const backend::HostSnap& s);
    bool snap_fits(const backend::HostSnap& s) const;

    Slot& cur() { return *slots_[cur_]; }
    const Slot& cur() const { return *slots_[cur_]; }
    const Slot& at(int s) const { return *slots_[s]; }
    int seqs() const { return (int)slots_.size(); }
    int slot() const { return cur_; }
    void use(int s);
    const SeqLayout& layout() const { return layout_; }
    uint64_t router_nans() const { return nans_; }
    bool nan_reported() const { return nan_reported_; }

    // MTP head (runner_mtp.mm), semantics of backend::Backend
    bool has_mtp() const { return mvocab_ != nullptr; }
    backend::MtpPick mtp_fill(int32_t next);
    backend::MtpPick mtp_step(int32_t tok);
    const float* mtp_logits() const { return pass_.mtp_logits(); }
    const MtpVocab& mtp_vocab() const { return *mvocab_; }
    const MtpCounts& mtp_counts() const { return mcount_; }

private:
    void run(Slot& s, const int32_t* tok, int nb, const Buf& logits, bool backup);
    // Commits and waits; with TONA_GPU_GAPS the command buffer goes to the probe, prepared from host time `prepared`
    void finish_timed(id<MTLCommandBuffer> cb, double prepared, const char* what);
    void run_split(Slot& s, id<MTLCommandBuffer> first, const PassIn& in, double prepared);
    void run_groups(Slot& s, id<MTLCommandBuffer> first, const PassIn& in);
    // With the arena: a pass that stops after each arena layer's router for its cold experts
    void run_cold(Slot& s, id<MTLCommandBuffer> cb, const PassIn& in);
    // At the part "cold": commits and waits for the command buffer up to the router, reads the cold experts, opens the
    // next command buffer
    id<MTLComputeCommandEncoder> cold_stop(id<MTLCommandBuffer> __strong& cb, id<MTLComputeCommandEncoder> enc, int nb);
    void prefill_chunk(Slot& s, const int32_t* tok, int nb, std::vector<float>* nll, size_t at, int64_t total);
    void nll_of(const int32_t* next, int t0, int n, float* out);
    void prefetch_after(const Slot& s, const int32_t* tok, int skip, int n);
    void profiled(SeqState& st, const PassIn& in);
    void by_layer(SeqState& st, const PassIn& in);
    struct ChunkTimes {
        double enter, first, encoded;  // host clock: by_layer entered, first command buffer committed, all
    };
    void wait_chunk(const std::vector<id<MTLCommandBuffer>>& cbs, const std::vector<int>& kinds, int nb,
                    const ChunkTimes& t);
    void blit(const Recurrent& from, Recurrent& to);
    void check_idle(const char* what) const;
    const float* trunk_h(const Slot& s, int64_t q, bool& stale) const;
    void mtp_pass(Slot& s, const int32_t* toks, const float* const* hs, const MtpPassIn& in);
    void mtp_catch_up(Slot& s, int64_t end_pos, int32_t next);
    void mtp_prefill(Slot& s, const int32_t* tok, int nb, const float* h0);
    void log_routes(int nb);
    backend::MtpPick mtp_pick() const;
    backend::MtpPick host_pick() const;

    Device& dev_;
    Pass pass_;
    SeqLayout layout_;
    int64_t max_ctx_;
    uint32_t vocab_;
    Buf logits_, head_;  // head_: logits of NLL_ROWS prefill tokens, allocated on the first nll request
    int chunk_;
    bool slow_cb_warned_ = false;
    std::vector<std::unique_ptr<Slot>> slots_;
    std::unique_ptr<Profiler> prof_;  // TONA_METAL_PROFILE
    std::unique_ptr<MtpVocab> mvocab_;
    const Qwen4ExpModel& m_;
    MtpCounts mcount_;
    GpuGaps gaps_;          // TONA_GPU_GAPS
    GpuGaps::Parts parts_;  // of the next command buffer's inputs (TONA_GPU_GAPS)
    // TONA_METAL_GPU_PICK: the head's pick on the GPU (kernels/mtp_pick.metal) instead of the host scan;
    // TONA_METAL_PICK_CHECK: both, and a line per 200 picks with the token mismatches, q's relative deviation and a
    // hash
    // On by default: same tokens, the decode cycle 0.4 to 0.7 ms shorter, decode +0.2 to +0.5 % (undecided after 6
    // rounds); the MSL 3.2 library it needs costs nothing measured in 8 rounds (research/measurements/
    // 2026-11-14-mac-mtp-gpu-pick.md)
    bool gpu_pick_ = env::integer("TONA_METAL_GPU_PICK", 1) != 0;
    bool pick_check_ = env::flag("TONA_METAL_PICK_CHECK");
    // Layers in the first command buffer of a pass (0: one command buffer). 2 by default: the target's encoding leaves
    // the critical path (0.83 -> 0.11 ms per decode cycle), GPU idle -0.56 ms, decode text +1.25 % (decided), code and
    // @8000 +0.3 and +0.5 % (open), same answers; k = 1 to 8 leave the same GPU idle between the halves, larger k
    // encode more before the first commit (research/measurements/2026-11-15-mac-early-commit.md)
    int early_commit_ = (int)env::integer("TONA_METAL_EARLY_COMMIT", 2);
    int split_every_ = (int)env::integer("TONA_METAL_SPLIT_EVERY", 0);
    bool split_wait_ = env::integer("TONA_METAL_SPLIT_WAIT", 0) != 0;
    mutable struct {
        int64_t picks = 0, mismatches = 0;
        uint64_t hash = 1469598103934665603ULL;  // FNV-1a over the GPU's token and q bits: equal input, equal hash
        std::vector<double> rel;
    } checked_;
    std::FILE* route_log_ = nullptr;  // TONA_MTP_ROUTE_STATS
    Arena* arena_ = nullptr;          // TONA_METAL_ARENA_GB
    id cold_ev_ = nil;                // id<MTLSharedEvent> of the arena stops (runner_cold.mm)
    Buf cold_flag_;                   // the word the GPU writes at a stop
    uint64_t cold_val_ = 0;           // its last value
    // TONA_METAL_STOP_SLEEP=f (0 to 0.9, 0 off): at a stop of a decode pass, sleep up to f times the stop's expected
    // wait before spinning; the spin alone kept a core busy for the whole pass (97.6 % of the pass time). Default
    // 0.25: our CPU 77 to 58 % of a core at no measured decode loss, 0.5 lost 1.3 to 4.2 % in most rounds
    // (research/measurements/2026-11-16-mac-arena-stop-cpu.md). TONA_METAL_STOP_SLEEP_SIGMAS=k: end the sleep k
    // deviations before the mean; TONA_METAL_STOP_SLEEP_MIN_US: no sleep at shorter stops (stop_wait.h)
    SleepRule sleep_rule_{std::clamp(env::number("TONA_METAL_STOP_SLEEP", 0.25), 0.0, 0.9),
                          env::number("TONA_METAL_STOP_SLEEP_SIGMAS", -1.0),
                          env::number("TONA_METAL_STOP_SLEEP_MIN_US", 0.0) * 1e-6};
    std::vector<StopPace> stop_pace_;
    struct {
        double planned = 0, slept = 0;
        uint64_t sleeps = 0, early = 0;  // early: the flag was there on waking
    } sleep_stats_;
    double sleep_before(size_t i) const;
    void pace_stop(size_t i, size_t n, std::chrono::steady_clock::time_point t0, bool spun);
    struct {
        const Slot* s = nullptr;
        int64_t pos = 0;
        int n = 0;
    } fw_;  // the rows of the target's residual after the last pass: its h at [pos, pos + n) of slot s
    int cur_ = 0;
    uint64_t nans_ = 0;
    bool nan_reported_ = false;  // a NaN warning of this request was printed (cleared by reset)
    // Counts the last pass's NaN router logits and reports them (backend/nan_guard.h: once per request and site)
    void note_nans(const char* site, int64_t pos);
};

}  // namespace tona::metal
