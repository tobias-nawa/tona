// Interface of a backend: everything the generator and the server need from a forward pass. One call computes a
// whole pass (no single ops), so every backend can exploit its platform (megakernel on gfx1200, Metal, CUDA, ...).
// The reference backend is gpu::Mega; it gets no abstraction in the hot path (one indirect call per pass; checked
// with bench-set before and after). A backend that cannot do something (e.g. no DFlash drafter) throws when called;
// the generator asks has_mtp() first and uses DFlash only if it is loaded.
//
// Sequence of a pass: forward(tokens, nb) or forward_multi over several slots, then exactly one commit(m) per
// participating slot. Slots: sequences with their own state; use_slot selects the active one and all other calls
// apply to it.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "backend/env.h"

namespace tona::backend {

constexpr int BMAX = 8;  // tokens per pass
constexpr int SMAX = 4;  // sequences with their own state (slots)

// TONA_DETERMINISTIC: every width computes bit-identically (passes up to 4 tokens in the server, no k_wide); otherwise
// passes up to 8 with k_wide from 6, where the dense part rounds differently (f16 instead of int8)
inline bool deterministic() {
    static const bool on = env::flag("TONA_DETERMINISTIC");
    return on;
}

// Counters; every backend fills what it has (the rest stay 0)
struct Stats {
    int64_t expert_hits = 0, expert_misses = 0;
    int64_t early_hits = 0, early_misses = 0;  // of those, in the first 10 passes after a prefill (extra slots refill)
    double upload_bytes = 0;
    int64_t launches = 0;
    int64_t yields = 0;  // kernel pauses because the CPU took too long (then restart from that step)
    double read_s = 0, wait_s = 0, select_s = 0;  // CPU: read misses; wait for the router message; selection
    int64_t cpu_experts = 0, fills = 0, gpu_direct = 0,
            dropped =
                0;  // experts computed by the CPU (counted per pass and layer); refilled; dropped (TONA_MISS_DROP)
    double cpu_s = 0, fill_s = 0;                                     // CPU: compute experts; read for refilling
    int64_t prefetched = 0, prefetch_used = 0, prefetch_checked = 0;  // prefetched; of those used; checked
    double prefetch_wait_s = 0;                                       // CPU: wait for prefetching
    int64_t mtp_passes = 0, mtp_stale = 0;  // MTP passes; entries without a known h (computed with 0)
    int64_t mtp_dyn = 0;                    // rows loaded into the truncated head on demand
    double mtp_pf_s = 0;                    // prefill: MTP entries of the prompt
    int64_t multi_passes = 0;               // passes over several sequences
    int64_t mtp_multi = 0,
            mtp_multi_diff = 0;  // MTP steps of several sequences; of those different from running alone (check mode)
    // Union of the experts per layer in passes with several tokens (sum, count), split by sequences
    double uni_single = 0, uni_multi = 0;
    int64_t uni_single_n = 0, uni_multi_n = 0;
    int64_t oea_swapped = 0, oea_short = 0;  // OEA: experts beyond the own top-k taken; slots left empty
    // TONA_TOKHIST: union experts of single-sequence passes by pass width and by number of tokens using them
    int64_t tokhist[9][9] = {};
    int64_t nan_rows = 0;       // tokens whose router logits held a NaN or inf (decode selection)
    bool nan_reported = false;  // the warning of this request has been printed
};

// Proposal of the MTP head: token and its probability
struct MtpPick {
    int32_t token = -1;
    double q = 0;
};

// A vector whose resize leaves new bytes uninitialized. Writing zeros first costs a second pass over up to 16 GB
// (measured at 30K tokens: 80 ms of the 140 ms of a stash, research/measurements/2026-11-01-stash-hang.md).
// Safety: a HostSnap is complete or empty. Its bytes are written in full by exactly two places: Mega::stash (all of
// state, KV and pool in one copy; on a stall, an exception or any other way out it resets the snapshot to empty) and
// read_checkpoint (all bytes read from the file, else it resets the snapshot). The checkpoint writer and unstash see
// only snapshots those two returned successfully; stash_current drops the entry if stash throws.
template <class T>
struct DefaultInitAllocator : std::allocator<T> {
    template <class U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };
    using std::allocator<T>::allocator;
    template <class U>
    void construct(U* p) noexcept(std::is_nothrow_default_constructible_v<U>) {
        ::new (static_cast<void*>(p)) U;
    }
    template <class U, class... A>
    void construct(U* p, A&&... a) {
        ::new (static_cast<void*>(p)) U(std::forward<A>(a)...);
    }
};
using Bytes = std::vector<uint8_t, DefaultInitAllocator<uint8_t>>;

// State of a sequence in RAM (stash/unstash); the content belongs to the backend
struct HostSnap {
    std::vector<int32_t> tokens;
    int64_t pos = 0;
    int commit = 0, parity = 0;
    Bytes state, kv, pool, dring;
    size_t bytes() const { return state.size() + kv.size() + pool.size() + dring.size(); }
};

// Memory and layout for /status and the log
struct Info {
    std::string name;  // e.g. "hip-gfx1200"
    size_t vram_bytes = 0, pinned_bytes = 0, kv_bytes = 0;
    bool kv_host = false;  // KV in RAM instead of VRAM
    int expert_slots = 0;  // experts per layer in fast memory
};

// Costs (ms) the server's width choice of speculative decoding weighs against the expected tokens: a pass of width 1 to
// BMAX of one conversation with the MTP head, a pass of N tokens across conversations, a chained head step and the
// head's fill. Fixed per backend, so the width depends on the request alone. A backend without its own numbers returns
// nullptr from pass_costs() and the server keeps the reference's (engine/server/generator.h).
struct PassCosts {
    double pass[BMAX + 1] = {}, multi[BMAX + 1] = {};
    double step = 0, fill = 0;
};

class Backend {
public:
    virtual ~Backend() = default;
    virtual Info info() const = 0;
    virtual const Stats& stats() const = 0;

    // Pass: nb tokens from pos(); logits [nb][vocab] until the next call. forward_multi: k slots, n[i] tokens per
    // slot back to back in tok, logits in the same order. Then commit(m) per slot, 1 <= m <= n
    virtual const float* forward(const int32_t* tokens, int nb) = 0;
    virtual const float* forward_multi(const int* slots, const int* n, int k, const int32_t* tok) = 0;
    virtual void commit(int m) = 0;
    // Prompt: n tokens from pos() in chunks (prefill_chunk() > 0: own path; 0: forward only); nll: -log p per token
    virtual void prefill(const int32_t* tokens, int n, std::vector<float>* nll = nullptr) = 0;
    virtual int prefill_chunk() const = 0;
    // Between passes, from the engine thread: housekeeping of the backend (the VRAM governor). True: call again every
    // 250 ms while idle; false: nothing to do ever (the default)
    virtual bool vram_tick() { return false; }
    virtual void reset() = 0;    // back to position 0
    virtual void recover() = 0;  // after an error in a pass: make the state usable again
    virtual int64_t pos() const = 0;
    virtual int64_t max_ctx() const = 0;
    virtual const std::vector<int32_t>& tokens() const = 0;  // accepted tokens up to pos()

    // Save point in the backend and stashed sequences in RAM
    virtual void save_state() = 0;
    virtual void load_state() = 0;
    virtual bool has_state() const = 0;
    virtual const std::vector<int32_t>& snap_tokens() const = 0;
    virtual const std::vector<int32_t>& snap_tokens_of(int slot) const = 0;
    virtual void stash(HostSnap& s) const = 0;
    // Bytes stash() will allocate for the current restore point (0 without one); lets the caller decide before the copy
    virtual size_t stash_bytes() const = 0;
    virtual void unstash(const HostSnap& s) = 0;
    virtual bool snap_fits(const HostSnap& s) const = 0;
    virtual size_t state_bytes() const = 0;
    virtual size_t dring_bytes() const = 0;
    virtual size_t kv_row_bytes() const = 0;

    // Slots
    virtual int seqs() const = 0;
    virtual int slot() const = 0;
    virtual void use_slot(int s) = 0;

    // DFlash drafter: catch up m committed features, draft a block of nb with emb
    virtual void draft(int m, const float* emb, int nb) = 0;
    virtual const float* draft_logits() const = 0;
    virtual const float* draft_gate() const = 0;

    // MTP head: fill catches up the entries to pos() - 1, with next >= 0 also next (proposal behind it);
    // step: tok behind the last entry; step_multi: one per slot, in one pass
    virtual bool has_mtp() const = 0;
    virtual const PassCosts* pass_costs() const { return nullptr; }
    virtual MtpPick mtp_fill(int32_t next) = 0;
    virtual MtpPick mtp_step(int32_t tok) = 0;
    virtual void mtp_step_multi(const int* slots, const int32_t* toks, int k, MtpPick* out) = 0;
    virtual void mtp_keep_logits(bool on) = 0;  // proposal logits to the host (sampling)
    virtual const float* mtp_logits() const = 0;
    virtual size_t mtp_rows() const = 0;
    virtual int32_t mtp_token(size_t r) const = 0;  // token of row r of mtp_logits (< 0: free)

    // The server is idle and expects a request soon (on), or a request is coming (off; returns when the hold has
    // stopped). A backend whose GPU needs no help to keep its clock ignores it (server/hold_controller.h)
    virtual void clock_hold(bool on) { (void)on; }
};

}  // namespace tona::backend
