// TONA_METAL_KEEPALIVE_MS=N (default 500 on macOS, 0 off): after every pass, for TONA_METAL_KEEPALIVE_S seconds
// (default 180), one command buffer every N ms on a queue of its own that reads 16 bytes of every weight buffer. After
// a pause of 1 to 2 s or more the first command buffer of an agent's tool step waited 330 to 380 ms for the GPU to
// start it (none after 0.5 s); residency sets did not change it, a command buffer that uses the weight buffers did: 9
// to 10 ms at 500 ms, the step 1.04 to 1.07 s instead of 1.35 to 1.43 s, no measurable idle cost at 250 ms,
// prefill@8000 +2.05 % (research/measurements/2026-11-15-mac-cache-hit.md). Nothing is sent while passes come faster
// than N ms. Work first: nothing is sent outside the window after a pass, so an idle server sends nothing.
#pragma once
#import <Metal/Metal.h>

#include <chrono>
#include <condition_variable>
#include <mutex>
#include <thread>
#include <vector>

#include "backend/env.h"

namespace tona::metal {

// The window after a pass (TONA_METAL_KEEPALIVE_S, default 180 s): the keepalive's, and how long the purgeable arena
// keeps an unused block non-volatile (arena_purge.mm); one setting, so both end together
inline double keepalive_window_s() { return (double)env::integer("TONA_METAL_KEEPALIVE_S", 180); }

class Keepalive {
public:
    using Clock = std::chrono::steady_clock;

    // every_ms <= 0: off, no thread
    Keepalive(id<MTLDevice> dev, double every_ms, double window_s);
    ~Keepalive();
    Keepalive(const Keepalive&) = delete;
    Keepalive& operator=(const Keepalive&) = delete;

    // Before a pass: the window restarts
    void touch();
    // The buffers each keepalive reads 16 bytes of (the weights')
    void reads(const std::vector<id<MTLBuffer>>& bufs);
    // Whether a keepalive is due at now: inside the window after the last pass, and no pass for an interval (the
    // passes themselves use the weight buffers)
    bool due(Clock::time_point now) const;
    long long sent() const;

private:
    void run();
    bool inside(Clock::time_point now) const { return touched_ && now - last_ < window_; }   // under mu_
    bool idle(Clock::time_point now) const { return inside(now) && now - last_ >= every_; }  // under mu_
    void send();

    id<MTLCommandQueue> queue_ = nil;
    id<MTLBuffer> buf_ = nil;
    std::vector<id<MTLBuffer>> reads_;  // under mu_
    std::chrono::duration<double> every_, window_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    Clock::time_point last_{};
    bool touched_ = false, stop_ = false;
    long long sent_ = 0;
    std::thread thread_;
};

}  // namespace tona::metal
