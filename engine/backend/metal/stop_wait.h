// When the wait at an arena stop gives up (runner_cold.mm). A wall-clock deadline of 4 s failed passes that were only
// slow: the process held by a profiler, and first passes under memory pressure
// (research/measurements/2026-11-16-mac-arena-fill-steps.md). Under pressure the engine must get slower, not fail. A
// GPU that hangs ends in macOS's watchdog: a command buffer that runs or waits for about 5 s is aborted and its status
// turns to error (engine/backend/metal/AGENTS.md). So the wait fails at once on that error or on a buffer that
// completed without reaching the stop, and otherwise only when the waiting thread itself ran `limit_s` of CPU time
// (it was not held or paging) in as much wall time: a stop that never comes without the watchdog's error is a bug.
#pragma once

#include <algorithm>
#include <cmath>

namespace tona::metal {

struct StopWait {
    double wall_s = 0;       // since the wait began
    double cpu_s = 0;        // CPU time of the waiting thread since then
    bool gpu_error = false;  // the command buffer's status is error
    bool gpu_done = false;   // it completed, and the flag is still not there
};

enum class StopVerdict { wait, gpu_failed, missed, hung };

// Above the watchdog's 5 s, so a GPU hang reports through its error first
constexpr double STOP_LIMIT_S = 10;

inline StopVerdict stop_verdict(const StopWait& w, double limit_s = STOP_LIMIT_S) {
    if (w.gpu_error) return StopVerdict::gpu_failed;
    if (w.gpu_done) return StopVerdict::missed;
    return w.cpu_s >= limit_s && w.wall_s >= limit_s ? StopVerdict::hung : StopVerdict::wait;
}

// How long to sleep at an arena stop before spinning (TONA_METAL_STOP_SLEEP). Per stop index of a pass: a moving mean
// and variance of the waits that spun to the flag. A fixed fraction of the mean cost 1.3 to 4.2 % decode (late wakes,
// research/measurements/2026-11-16-mac-arena-stop-cpu.md), so the sleep can end `sigmas` deviations before the mean
// instead; stops shorter than `min_s` are not slept at all (a sleep wakes late by tens of microseconds)
struct StopPace {
    double mean = 0, var = 0;
    bool seen = false;
};

struct SleepRule {
    double fraction = 0;  // of the mean; 0: off
    double sigmas = -1;   // >= 0: mean minus sigmas deviations, capped by the fraction
    double min_s = 0;
};

// A wait whose flag was there on waking says only that the GPU came earlier than the sleep ended: the mean drops, so
// the sleep approaches the arrival from below and never feeds on its own overshoot
inline void pace_update(StopPace& p, double wait_s, bool spun) {
    constexpr double a = 0.2;
    if (!spun) {
        p.mean *= 0.8;
        return;
    }
    if (!p.seen) {
        p.mean = wait_s, p.var = 0, p.seen = true;
        return;
    }
    const double d = wait_s - p.mean;
    p.mean += a * d, p.var = (1 - a) * (p.var + a * d * d);
}

inline double pace_sleep(const StopPace& p, const SleepRule& r) {
    if (r.fraction <= 0 || !p.seen || p.mean < r.min_s) return 0;
    double s = r.fraction * p.mean;
    if (r.sigmas >= 0) {
        s = std::min(s, p.mean - r.sigmas * std::sqrt(p.var > 0 ? p.var : 0));
    }
    return s > 0 ? s : 0;
}

}  // namespace tona::metal
