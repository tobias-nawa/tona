// Passes with the expert arena (arena.h): after each arena layer's router the GPU stops until the CPU has read the
// layer's cold experts into the staging slots.
#include <mach/mach.h>
#include <time.h>

#include <atomic>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include "backend/metal/runner.h"
#include "backend/metal/stop_wait.h"

namespace tona::metal {

namespace {

double since(std::chrono::steady_clock::time_point t0) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

double thread_cpu_s() { return (double)clock_gettime_nsec_np(CLOCK_THREAD_CPUTIME_ID) * 1e-9; }

int64_t page_ins() {
    task_events_info_data_t ev{};
    mach_msg_type_number_t n = TASK_EVENTS_INFO_COUNT;
    return task_info(mach_task_self(), TASK_EVENTS_INFO, (task_info_t)&ev, &n) == KERN_SUCCESS ? ev.pageins : -1;
}

// A stop that waits longer than a second: why, for the cases the old 4 s deadline failed (held process, paging)
void note_long_wait(const char* what, double wall, double cpu, int64_t pageins, uint32_t seen, uint32_t want) {
    std::fprintf(stderr, "Metal: an arena stop %s %.1f s (waiting thread's CPU %.1f s, page-ins %lld, flag %u of %u)\n",
                 what, wall, cpu, (long long)pageins, seen, want);
}

// Until the GPU has written `value` into the flag; gives up as stop_wait.h decides. Sleeps `sleep_s` first; returns
// whether the flag came after it (false: it was there on waking, the sleep may have been too long)
bool wait_flag(const Buf& flag, uint32_t value, id<MTLCommandBuffer> cb, double sleep_s = 0, double* slept = nullptr) {
    const auto* word = reinterpret_cast<const std::atomic<uint32_t>*>(flag.i32());
    // At or past `value`, wrap-safe: the GPU may already have written the next stop's value into the word (the
    // prediction word is written again before the next stop's wait), and an equality test then never comes true:
    // the GPU waited for an event the CPU never signaled until the watchdog aborted the pass (2026-11-16, a stop sleep
    // of 0.5 with all experts resident)
    const auto there = [&] { return (int32_t)(word->load(std::memory_order_acquire) - value) >= 0; };
    if (sleep_s > 0 && !there()) {
        const auto s0 = std::chrono::steady_clock::now();
        std::this_thread::sleep_for(std::chrono::duration<double>(sleep_s));
        if (slept) *slept = since(s0);
        if (there()) return false;
    }
    const auto t0 = std::chrono::steady_clock::now();
    const double c0 = thread_cpu_s();
    const int64_t p0 = page_ins();
    bool noted = false;
    // The clock and the status only every 1024 rounds: asked every round, the clock was 95 % of the spin
    for (uint32_t n = 1; !there(); n++) {
        if (n % 1024) continue;
        const MTLCommandBufferStatus st = cb.status;
        const StopWait w{since(t0), thread_cpu_s() - c0, st == MTLCommandBufferStatusError,
                         st == MTLCommandBufferStatusCompleted && !there()};
        switch (stop_verdict(w)) {
            case StopVerdict::wait:
                break;
            case StopVerdict::gpu_failed:
                throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
            case StopVerdict::missed:
                throw std::runtime_error("Metal: the pass completed without reaching an arena stop");
            case StopVerdict::hung:
                throw std::runtime_error("Metal: no arena stop within " + std::to_string((int)w.wall_s) +
                                         " s while this thread ran; the GPU made no progress");
        }
        if (!noted && w.wall_s > 1.0)
            noted = true, note_long_wait("waits", w.wall_s, w.cpu_s, page_ins() - p0, word->load(), value);
    }
    if (noted) note_long_wait("came after", since(t0), thread_cpu_s() - c0, page_ins() - p0, value, value);
    return true;
}

}  // namespace

// Prefill: a chunk's layers are command buffers of their own already (by_layer), so the stop commits and waits
id<MTLComputeCommandEncoder> Runner::cold_stop(id<MTLCommandBuffer> __strong& cb, id<MTLComputeCommandEncoder> enc,
                                               int nb) {
    [enc endEncoding];
    const auto t0 = std::chrono::steady_clock::now();
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
    arena_->note_stop(since(t0), cb.GPUEndTime - cb.GPUStartTime);
    arena_->read_cold(pass_.cold_layer(), pass_.route_ids(), nb * (int)m_.hp().n_expert_used, false);
    cb = [dev_.queue() commandBuffer];
    return [cb computeCommandEncoder];
}

// Decode and verify passes: one command buffer that waits on a shared event at every stop. A commit and wait per stop
// cost 2 ms of GPU idle each (48 per pass, research/measurements/2026-11-15-mac-coder-arena.md); an event reaches the
// waiting GPU in about 55 µs (research/measurements/2026-11-11-mac-event.md).
void Runner::run_cold(Slot& s, id<MTLCommandBuffer> cb, const PassIn& in) {
    if (!cold_ev_) cold_ev_ = [dev_.dev() newSharedEvent], cold_flag_ = dev_.floats(2);
    id<MTLSharedEvent> ev = cold_ev_;
    std::vector<std::pair<uint32_t, uint64_t>> stops;  // layer, the value the CPU signals after its reads
    std::vector<size_t> ahead;                         // stops with a prediction of the next layer
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
    pass_.split = [&](const char* name) {
        if (std::strcmp(name, "cold") == 0) {
            const uint64_t v = cold_val_ += 2;
            const auto at = (uint32_t)(v - 1);
            Dispatch{enc, dev_}.run("stop_flag", {cold_flag_}, at, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
            stops.emplace_back(pass_.cold_layer(), v);
        } else if (std::strcmp(name, "ahead") == 0) {  // the prediction is written: word 1
            const auto at = (uint32_t)(stops.back().second - 1);
            Dispatch{enc, dev_}.run("stop_flag", {cold_flag_.at(4)}, at, MTLSizeMake(1, 1, 1), MTLSizeMake(1, 1, 1));
            ahead.push_back(stops.size() - 1);
        } else if (std::strcmp(name, "cold_wait") == 0) {
            [enc endEncoding];
            [cb encodeWaitForEvent:ev value:stops.back().second];
            enc = [cb computeCommandEncoder];
        }
        return enc;
    };
    pass_.encode(enc, s.st, in);
    pass_.split = nullptr;
    [enc endEncoding];
    parts_.rows = pass_.rows_s(), parts_.positions = pass_.positions_s();
    [cb commit];
    const int n = in.nb * (int)m_.hp().n_expert_used;
    try {
        size_t next_ahead = 0;
        for (size_t i = 0; i < stops.size(); i++) {
            const auto [l, v] = stops[i];
            const auto t0 = std::chrono::steady_clock::now();
            const double plan = sleep_before(i);
            double slept = 0;
            const bool spun = wait_flag(cold_flag_, (uint32_t)(v - 1), cb, plan, &slept);
            if (plan > 0) {
                auto& st = sleep_stats_;
                st.planned += plan, st.slept += slept, st.sleeps++, st.early += !spun;
                if (st.sleeps % 5000 == 0)  // planned against actual sleep: late wakes cost decode
                    std::fprintf(
                        stderr,
                        "Metal: stop sleeps %llu, planned %.3f ms, slept %.3f ms, flag there on waking %.1f %%\n",
                        (unsigned long long)st.sleeps, st.planned * 1e3 / (double)st.sleeps,
                        st.slept * 1e3 / (double)st.sleeps, 100.0 * (double)st.early / (double)st.sleeps);
            }
            pace_stop(i, stops.size(), t0, spun);
            arena_->note_stop(since(t0), 0);
            arena_->read_cold(l, pass_.route_ids(), n, true);
            std::atomic_thread_fence(std::memory_order_release);  // the table before the GPU goes on
            ev.signaledValue = v;
            if (next_ahead < ahead.size() && ahead[next_ahead] == i) {
                next_ahead++;
                wait_flag(cold_flag_.at(4), (uint32_t)(v - 1), cb);
                arena_->prefetch(l + 1, pass_.ahead_ids(), in.nb * (int)pass_.ahead_k());
            }
        }
    } catch (...) {
        ev.signaledValue = cold_val_;  // release the GPU's remaining waits
        [cb waitUntilCompleted];
        throw;
    }
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(std::string("Metal: pass failed: ") + cb.error.description.UTF8String);
}

double Runner::sleep_before(size_t i) const {
    return i < stop_pace_.size() ? pace_sleep(stop_pace_[i], sleep_rule_) : 0.0;
}

void Runner::pace_stop(size_t i, size_t n, std::chrono::steady_clock::time_point t0, bool spun) {
    if (sleep_rule_.fraction <= 0) return;
    if (stop_pace_.size() != n) stop_pace_.assign(n, StopPace{});
    pace_update(stop_pace_[i], since(t0), spun);
}

}  // namespace tona::metal
