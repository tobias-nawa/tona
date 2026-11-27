// Clock hold of the server (TONA_CLOCK_HOLD=1 or 2, off by default): when the last request has finished and nothing is
// queued, the backend keeps the GPU clock up for as long as the rule of hold_policy.h says, and stops at the next
// request. The next request's prefill then starts at full clock instead of paying the ramp.
#pragma once

#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <thread>

#include "backend/backend.h"
#include "server/hold_policy.h"

namespace tona::server {

class HoldController {
public:
    // nullptr unless TONA_CLOCK_HOLD is 1 or 2
    static std::unique_ptr<HoldController> from_env(backend::Backend& b);
    HoldController(backend::Backend& b, double seconds_per_save);
    ~HoldController();

    // A request reached the server (HTTP thread): stops the hold before the request is admitted. warm: a warm-up job,
    // not an agent's pause, so it is not recorded.
    void arrival(bool warm);
    // The server is idle now: queue empty, no request running
    void idle();

    struct Totals {
        long holds = 0, covered = 0;  // holds started; requests that came while one ran
        double held_s = 0;
        double last_hold_s = 0;  // the length the rule chose last time
    };
    Totals totals() const;

private:
    using Clock = std::chrono::steady_clock;
    void loop();
    void stop_locked();  // holding_ -> false and the backend off; under mu_

    backend::Backend& be_;
    const double per_save_;
    mutable std::mutex mu_;
    std::condition_variable cv_;
    PauseWindow window_;
    uint64_t seq_ = 0;  // counts idle periods
    bool idle_ = false, holding_ = false, quit_ = false;
    Clock::time_point idle_since_, hold_start_;
    Totals tot_;
    std::thread th_;
};

}  // namespace tona::server
