#include "backend/metal/runner.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/env.h"
#include "backend/host_logits.h"
#include "backend/nan_guard.h"
#include "backend/snap_layout.h"
#include "model/ple.h"

namespace tona::metal {

namespace {

constexpr int NLL_ROWS = 64;  // prefill tokens per LM head slice: 64 × vocab floats (64 MB for 248k)

void finish(id<MTLCommandBuffer> cb) {
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
}

}  // namespace

Runner::Runner(Device& dev, Weights& w, const Qwen4ExpModel& m, const RunnerOptions& o)
    : dev_(dev),
      pass_(dev, w, m, o.mtp, o.max_ctx, o.chunk),
      layout_(seq_layout(m.hp())),
      max_ctx_(o.max_ctx),
      vocab_(m.hp().vocab),
      chunk_(o.chunk),
      m_(m),
      arena_(w.arena()) {
    const MtpHead* mtp = o.mtp;
    logits_ = dev.floats((uint64_t)BMAX * backend::SMAX * vocab_);
    if (mtp) {
        mvocab_ = std::make_unique<MtpVocab>(dev, m);
        pass_.mtp_bufs().head = mvocab_->head();
        pass_.mtp_bufs().logits = dev.floats(mvocab_->rows());
        if (gpu_pick_) {
            pass_.mtp_bufs().rows = mvocab_->row_tokens();
            pass_.mtp_bufs().pick_part = dev.floats(4ULL * 256), pass_.mtp_bufs().pick_out = dev.floats(4);
            pass_.mtp_bufs().pick_done = dev.floats(1);
        }
        if (const char* f = env::text("TONA_MTP_ROUTE_STATS")) {
            route_log_ = std::fopen(f, "a");
            if (!route_log_) throw std::runtime_error(std::string("Metal: cannot open ") + f);
        }
    }
    if (env::flag("TONA_METAL_PROFILE") && arena_)
        throw std::runtime_error("Metal: TONA_METAL_PROFILE does not work with the arena yet");
    if (env::flag("TONA_METAL_PROFILE")) {
        prof_ = std::make_unique<Profiler>(dev);
        pass_.split = [this](const char* name) { return prof_->next(name); };
    }
    for (int i = 0; i < o.seqs; i++) {
        slots_.push_back(std::make_unique<Slot>());
        slots_.back()->st = make_seq(dev, m.hp(), o.max_ctx, mtp ? 1 : 0);
    }
}

Runner::~Runner() {
    if (route_log_) std::fclose(route_log_);
}

void Runner::check_idle(const char* what) const {
    if (!cur().pass.empty()) throw std::logic_error(std::string("Metal backend: ") + what + " before commit");
}

void Runner::use(int s) {
    if (s < 0 || s >= seqs()) throw std::out_of_range("Metal backend: no slot " + std::to_string(s));
    cur_ = s;
}

// One pass of one slot in its own command buffer: the scratch inputs are written on the CPU before encoding, so two
// passes cannot share a command buffer
// A whole pass with every part in its own command buffer (Pass::split); layer() and finish() switch to their first
// part's encoder before they dispatch anything
void Runner::profiled(SeqState& st, const PassIn& in) {
    pass_.begin(prof_->next("embed"), in);
    for (uint32_t l = 0; l < pass_.layers(); l++) pass_.layer(nil, st, in, l);
    pass_.finish(nil, st, in);
    prof_->end();
}

void Runner::finish_timed(id<MTLCommandBuffer> cb, double prepared, const char* what) {
    [cb commit];
    const double committed = GpuGaps::enabled() ? GpuGaps::now() : 0;
    [cb waitUntilCompleted];
    if (cb.error)
        throw std::runtime_error(std::string("Metal: ") + what + " failed: " + cb.error.description.UTF8String);
    if (GpuGaps::enabled())
        gaps_.add({prepared, committed, cb.GPUStartTime, cb.GPUEndTime, GpuGaps::now(), parts_, 0, 0});
    parts_ = {};
}

void Runner::run(Slot& s, const int32_t* tok, int nb, const Buf& logits, bool backup) {
    const double prepared = GpuGaps::enabled() ? GpuGaps::now() : 0;
    fw_ = {&s, s.pos, nb};
    std::vector<int32_t> history(s.tokens);
    history.insert(history.end(), tok, tok + nb);
    if (GpuGaps::enabled()) parts_.history = GpuGaps::now() - prepared;
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [dev_.queue() commandBuffer];
        if (backup) copy(cb, s.st.live, s.st.backup);
        if (prof_) {
            finish(cb);
            profiled(s.st, PassIn{tok, nb, s.pos, &history, logits});
            note_nans("router logits", s.pos);
            return;
        }
        const PassIn in{tok, nb, s.pos, &history, logits};
        if (arena_) {
            run_cold(s, cb, in);
        } else if (split_every_ > 0) {
            run_groups(s, cb, in);
        } else if (early_commit_ > 0 && early_commit_ < (int)pass_.layers()) {
            run_split(s, cb, in, prepared);
        } else {
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            pass_.encode(enc, s.st, in);
            [enc endEncoding];
            parts_.rows = pass_.rows_s(), parts_.positions = pass_.positions_s();
            finish_timed(cb, prepared, "pass");
        }
    }
    note_nans("router logits", s.pos);
}

// Probe TONA_METAL_SPLIT_EVERY=n: a command buffer per n layers, each binding only the weights of its layers, for a
// model whose weights exceed one command buffer's working set (the Coder on 32 GB: 29.6 GB against 26.8,
// research/measurements/2026-11-15-mac-coder-baseline.md); with TONA_METAL_SPLIT_WAIT=1 each is waited on before the
// next is committed. The same kernels in the same order: bit-identical.
void Runner::run_groups(Slot& s, id<MTLCommandBuffer> first, const PassIn& in) {
    const auto n = (uint32_t)split_every_, layers = pass_.layers();
    std::vector<id<MTLCommandBuffer>> cbs;
    for (uint32_t l0 = 0; l0 < layers; l0 += n) {
        id<MTLCommandBuffer> cb = cbs.empty() ? first : [dev_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        if (l0 == 0) pass_.begin(enc, in);
        for (uint32_t l = l0; l < std::min(layers, l0 + n); l++) pass_.layer(enc, s.st, in, l);
        if (l0 + n >= layers) pass_.finish(enc, s.st, in);
        [enc endEncoding];
        [cb commit];
        if (split_wait_) [cb waitUntilCompleted];
        cbs.push_back(cb);
    }
    parts_.rows = pass_.rows_s(), parts_.positions = pass_.positions_s();
    [cbs.back() waitUntilCompleted];
    for (id<MTLCommandBuffer> cb : cbs)
        if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
}

// TONA_METAL_EARLY_COMMIT=k: the inputs and the first k layers go to the GPU at once, the rest is encoded while they
// run (encoding a target pass costs about 0.8 ms of host time, research/measurements/2026-11-14-mac-gpu-gaps.md; as
// llama.cpp's n_main, ggml-metal-context.m). The same kernels in the same order on the same queue: bit-identical.
void Runner::run_split(Slot& s, id<MTLCommandBuffer> first, const PassIn& in, double prepared) {
    const auto k = (uint32_t)early_commit_;
    id<MTLComputeCommandEncoder> enc = [first computeCommandEncoder];
    pass_.begin(enc, in);
    for (uint32_t l = 0; l < k; l++) pass_.layer(enc, s.st, in, l);
    [enc endEncoding];
    [first commit];
    const double committed = GpuGaps::enabled() ? GpuGaps::now() : 0;
    parts_.rows = pass_.rows_s(), parts_.positions = pass_.positions_s();
    id<MTLCommandBuffer> rest = [dev_.queue() commandBuffer];
    enc = [rest computeCommandEncoder];
    for (uint32_t l = k; l < pass_.layers(); l++) pass_.layer(enc, s.st, in, l);
    pass_.finish(enc, s.st, in);
    [enc endEncoding];
    [rest commit];
    const double rest_committed = GpuGaps::enabled() ? GpuGaps::now() : 0;
    [rest waitUntilCompleted];
    [first waitUntilCompleted];
    for (id<MTLCommandBuffer> cb : {first, rest})
        if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
    if (!GpuGaps::enabled()) return;
    const double inner = std::max(0.0, rest.GPUStartTime - first.GPUEndTime);
    gaps_.add({prepared, committed, first.GPUStartTime, rest.GPUEndTime, GpuGaps::now(), parts_, inner,
               std::clamp(rest_committed - first.GPUEndTime, 0.0, inner)});
    parts_ = {};
}

const float* Runner::forward(const int* slots, const int* n, int k, const int32_t* tok, bool backup) {
    const GpuGaps::Scope scope(gaps_, GpuGaps::kTarget);
    const Arena::PassScope arena_pass(arena_);  // until the pass's command buffers completed
    dev_.touch();
    if (k < 1 || k > seqs()) throw std::logic_error("Metal backend: forward over no or too many slots");
    for (int i = 0; i < k; i++) {
        const Slot& s = at(slots[i]);
        if (!s.pass.empty()) throw std::logic_error("Metal backend: forward without commit");
        if (n[i] < 1 || n[i] > BMAX) throw std::logic_error("Metal backend: pass width outside 1..8");
    }
    uint64_t row = 0;
    for (int i = 0; i < k; i++) {
        Slot& s = *slots_[slots[i]];
        run(s, tok, n[i], logits_.at(row * vocab_ * 4), backup && n[i] > 1);
        s.pass.assign(tok, tok + n[i]);
        tok += n[i], row += n[i];
    }
    return logits_.f32();
}

void Runner::commit(int m) {
    const GpuGaps::Scope scope(gaps_, GpuGaps::kCommit);
    Slot& s = cur();
    const int nb = (int)s.pass.size();
    if (m < 1 || m > nb) throw std::logic_error("Metal backend: commit outside 1..nb of the pass");
    // m < nb: the state from the backup and the DeltaNet recurrence of the m tokens from the inputs the pass kept, as
    // the reference does (Mega::commit, k_dn_recur); the projections' bits do not depend on the width, so this equals a
    // pass of m tokens. KV cache and indexer rows behind the new position are never read.
    if (m < nb) {
        @autoreleasepool {
            const double prepared = GpuGaps::enabled() ? GpuGaps::now() : 0;
            id<MTLCommandBuffer> cb = [dev_.queue() commandBuffer];
            restore_after(cb, s.st, m, nb, (m_.hp().ple_conv - 1) * m_.hp().ple_ngram);
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            pass_.replay_dn(enc, s.st, m);
            [enc endEncoding];
            finish_timed(cb, prepared, "commit");
        }
        if (fw_.s == &s) fw_.n = m;
    }
    s.tokens.insert(s.tokens.end(), s.pass.begin(), s.pass.begin() + m);
    s.pos += m;
    s.pass.clear();
}

// One command buffer for the inputs and the embedding, one per layer, one for the end; committed together, waited on
// once. Each must stay at most 0.5 s (engine/backend/metal/AGENTS.md): a longer one is reported once on stderr;
// TONA_METAL_CB_LOG=1 prints the longest of every chunk and the GPU time by kind of command buffer.
void Runner::by_layer(SeqState& st, const PassIn& in) {
    const uint32_t layers = pass_.layers();
    std::vector<id<MTLCommandBuffer>> cbs;
    std::vector<int> kinds;  // per command buffer: 0 the start of a layer (mixer), 1 attention, 2 experts, 3 the ends
    id<MTLCommandBuffer> cb = nil;
    id<MTLComputeCommandEncoder> enc = nil;
    int kind = 0;
    const double t_enter = GpuGaps::now();
    double t_first = 0;
    // The long parts of a layer start a command buffer of their own: each attention slice and the routed experts
    pass_.split = [&](const char* name) {
        if (std::strcmp(name, "cold") == 0) {  // the arena: wait, read the cold experts, go on as for "experts"
            cbs.push_back(cb), kinds.push_back(kind);
            enc = cold_stop(cb, enc, in.nb);
            kind = 2;
            return enc;
        }
        if (std::strcmp(name, "attend") != 0 && std::strcmp(name, "experts") != 0) return enc;
        [enc endEncoding];
        [cb commit];
        cbs.push_back(cb), kinds.push_back(kind);
        kind = name[0] == 'a' ? 1 : 2;
        cb = [dev_.queue() commandBuffer];
        return enc = [cb computeCommandEncoder];
    };
    for (uint32_t part = 0; part < layers + 2; part++) {
        cb = [dev_.queue() commandBuffer];
        enc = [cb computeCommandEncoder];
        kind = part == 0 || part > layers ? 3 : 0;
        if (part == 0)
            pass_.begin(enc, in);
        else if (part <= layers)
            pass_.layer(enc, st, in, part - 1);
        else
            pass_.finish(enc, st, in);
        [enc endEncoding];
        [cb commit];
        cbs.push_back(cb), kinds.push_back(kind);
        if (part == 0) t_first = GpuGaps::now();
    }
    pass_.split = nullptr;
    wait_chunk(cbs, kinds, in.nb, {t_enter, t_first, GpuGaps::now()});
}

// Waits on a prefill chunk's command buffers; the 0.5 s check and TONA_METAL_CB_LOG
void Runner::wait_chunk(const std::vector<id<MTLCommandBuffer>>& cbs, const std::vector<int>& kinds, int nb,
                        const ChunkTimes& t) {
    double longest = 0, idle = 0, gap = 0, prev_end = 0, by_kind[4] = {};
    for (size_t c = 0; c < cbs.size(); c++) {
        id<MTLCommandBuffer> cb = cbs[c];
        [cb waitUntilCompleted];
        if (cb.error)
            throw std::runtime_error(std::string("Metal: prefill failed: ") + cb.error.description.UTF8String);
        longest = std::max(longest, cb.GPUEndTime - cb.GPUStartTime);
        by_kind[kinds[c]] += cb.GPUEndTime - cb.GPUStartTime;
        if (prev_end > 0)
            idle += std::max(0.0, cb.GPUStartTime - prev_end), gap = std::max(gap, cb.GPUStartTime - prev_end);
        prev_end = cb.GPUEndTime;
    }
    if (env::flag("TONA_METAL_CB_LOG"))
        std::fprintf(
            stderr,
            "Metal: prefill chunk of %d tokens, %zu command buffers, longest %.0f ms, GPU idle between them "
            "%.2f ms (longest %.2f); GPU ms: layer starts %.1f, attention %.1f, experts %.1f, ends %.1f; "
            "host: encoding %.1f ms, first committed %.1f ms, until it starts %.1f ms, after the last ends %.1f ms\n",
            nb, cbs.size(), longest * 1e3, idle * 1e3, gap * 1e3, by_kind[0] * 1e3, by_kind[1] * 1e3, by_kind[2] * 1e3,
            by_kind[3] * 1e3, (t.encoded - t.enter) * 1e3, (t.first - t.enter) * 1e3,
            (cbs.front().GPUStartTime - t.first) * 1e3, (GpuGaps::now() - prev_end) * 1e3);
    if (longest > 0.5 && !slow_cb_warned_) {
        slow_cb_warned_ = true;
        std::fprintf(stderr,
                     "Metal: a prefill command buffer took %.0f ms, over the 0.5 s limit; lower TONA_PREFILL_CHUNK\n",
                     longest * 1e3);
    }
}

// One chunk through the prefill kernels; then the -log p of its tokens if asked
void Runner::prefill_chunk(Slot& s, const int32_t* tok, int nb, std::vector<float>* nll, size_t at, int64_t total) {
    const double t_chunk = GpuGaps::now();
    const Arena::PassScope arena_pass(arena_);
    std::vector<int32_t> history(s.tokens);
    history.insert(history.end(), tok, tok + nb);
    const PassIn in{tok, nb, s.pos, &history, Buf{}, true, nll != nullptr};
    // The target's h before the chunk for its first MTP entry, before the chunk overwrites the residual
    std::vector<float> h0;
    bool stale = false;
    if (const float* h = has_mtp() ? trunk_h(s, s.pos - 1, stale) : nullptr)
        h0.assign(h, h + (size_t)m_.hp().n_hc * m_.hp().d);
    mcount_.stale += stale;
    fw_ = {&s, s.pos, nb};
    @autoreleasepool {
        if (prof_)
            profiled(s.st, in);
        else
            by_layer(s.st, in);
    }
    note_nans("router logits", s.pos);
    s.tokens.insert(s.tokens.end(), tok, tok + nb);
    s.pos += nb;
    for (int t0 = 0; nll && t0 < nb && (int64_t)at + t0 + 1 < total; t0 += NLL_ROWS) {
        const int n = (int)std::min<int64_t>({NLL_ROWS, nb - t0, total - 1 - (int64_t)at - t0});
        nll_of(tok + t0 + 1, t0, n, nll->data() + at + t0);  // tok runs on past the chunk to the end of the prompt
    }
    const double t_mtp = GpuGaps::now();
    if (has_mtp()) mtp_prefill(s, tok, nb, h0.empty() ? nullptr : h0.data());  // after nll_of: both use the scratch
    if (env::flag("TONA_METAL_CB_LOG"))
        std::fprintf(stderr, "Metal: prefill chunk: MTP %.1f ms, whole chunk %.1f ms\n", (GpuGaps::now() - t_mtp) * 1e3,
                     (GpuGaps::now() - t_chunk) * 1e3);
}

// -log p of the next tokens for rows [t0, t0 + n) of the last prefill chunk
void Runner::nll_of(const int32_t* next, int t0, int n, float* out) {
    gaps_.cut();
    if (!head_.b) head_ = dev_.floats((uint64_t)NLL_ROWS * vocab_);
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [dev_.queue() commandBuffer];
        id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
        pass_.head(enc, t0, n, head_);
        [enc endEncoding];
        finish(cb);
    }
    for (int t = 0; t < n; t++) {
        const float* lg = head_.f32() + (size_t)t * vocab_;
        out[t] = (float)backend::neg_log_p(lg, vocab_, next[t]);
    }
}

// Read-ahead of the n-gram rows of tok[skip, n), the tokens tok[0, n) following the slot's accepted ones
void Runner::prefetch_after(const Slot& s, const int32_t* tok, int skip, int n) {
    const size_t keep = std::min<size_t>(s.tokens.size(), ple::MAX_NGRAM - 1);  // the n-grams' predecessors
    std::vector<int32_t> seq(s.tokens.end() - (std::ptrdiff_t)keep, s.tokens.end());
    seq.insert(seq.end(), tok, tok + n);
    pass_.prefetch_rows(seq.data(), (int64_t)keep + skip, (int64_t)seq.size());
}

void Runner::prefill(const int32_t* tok, int n, std::vector<float>* nll) {
    check_idle("prefill");
    gaps_.cut();
    dev_.touch();
    if (nll) nll->assign(n > 1 ? n - 1 : 0, 0.0f);
    Slot& s = cur();
    for (int at = 0; at < n; at += chunk_) prefill_chunk(s, tok + at, std::min(chunk_, n - at), nll, at, n);
}

void Runner::reset() {
    gaps_.cut();
    Slot& s = cur();
    clear(s.st);
    s.tokens.clear(), s.pass.clear(), s.snap_tokens.clear();
    s.pos = s.snap_pos = 0;
    s.mtp.on_reset(), s.mtp.nb = 0, fw_.n = 0;
    s.snap = false;
    nan_reported_ = false;
}

void Runner::note_nans(const char* site, int64_t pos) {
    const uint64_t n = pass_.router_nans();
    nans_ += n;
    if (nan::report(site, n, "position " + std::to_string(pos))) nan_reported_ = true;
}

void Runner::blit(const Recurrent& from, Recurrent& to) {
    gaps_.cut();
    @autoreleasepool {
        id<MTLCommandBuffer> cb = [dev_.queue() commandBuffer];
        copy(cb, from, to);
        finish(cb);
    }
}

// The save point keeps only the recurrent part: KV rows, raw keys and pool rows before the position are never written
// again by later passes (a block's pool row is written when the block completes, always at or after the save point)
void Runner::save_state() {
    check_idle("save_state");
    Slot& s = cur();
    blit(s.st.live, s.st.saved);
    s.snap_pos = s.pos, s.snap_tokens = s.tokens, s.snap = true;
}

void Runner::load_state() {
    Slot& s = cur();
    if (!s.snap || !s.pass.empty())
        throw std::runtime_error("Metal backend: load_state without a save point or before commit");
    blit(s.st.saved, s.st.live);
    s.pos = s.snap_pos, s.tokens = s.snap_tokens;
    s.mtp.valid = std::min(s.mtp.valid, s.pos), s.mtp.nb = 0, s.mtp.scan = 0, fw_.n = 0;
}

void Runner::stash(backend::HostSnap& out) const {
    const Slot& s = cur();
    if (!s.snap) throw std::runtime_error("Metal backend: stash without a save point");
    metal::stash(s.st, layout_, s.snap_pos, out);
    out.tokens = s.snap_tokens, out.pos = s.snap_pos, out.commit = 0, out.parity = 0;
}

size_t Runner::stash_bytes() const {
    const Slot& s = cur();
    if (!s.snap) return 0;
    return backend::snap_total_bytes(layout_.recurrent + 4, s.snap_pos * layout_.kv_row(),
                                     layout_.n_qsa * layout_.pool_rows(s.snap_pos) * layout_.pool);
}

bool Runner::snap_fits(const backend::HostSnap& in) const {
    return in.commit == 0 && in.parity == 0 && metal::snap_fits(layout_, in, max_ctx_);
}

void Runner::unstash(const backend::HostSnap& in) {
    check_idle("unstash");
    if (!snap_fits(in)) throw std::runtime_error("Metal backend: stash of another layout");
    Slot& s = cur();
    metal::unstash(s.st, layout_, in);
    s.snap_tokens = in.tokens, s.snap_pos = in.pos, s.snap = true;
    load_state();
}

}  // namespace tona::metal
