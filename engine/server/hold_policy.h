// How long the GPU clock is held after a response: derived from the pauses this server has seen, not a fixed time
// (TONA_CLOCK_HOLD, research/60-gpu-clock-ramp.md). Free of the GPU and of threads so a unit test runs it.
#pragma once

#include <algorithm>
#include <cstddef>
#include <deque>
#include <vector>

namespace tona::server {

// The rule. gaps: the last pauses (seconds) between the end of the last response and the next request, the server being
// idle. Holding for T costs T seconds of 27 W in each pause that lasts at least T (shorter pauses cost their length)
// and saves the ~190 ms of the clock ramp in each pause that is at most T. For every T among the observed pauses the
// hold-seconds per saved request are
//     cost(T) = mean(min(gap, T)) / (share of gaps <= T)
// The hold is the largest T whose cost is at most seconds_per_save: the exchange rate between holding and a saved ramp.
// A long absence (lunch) among short pauses makes the cost of covering it exceed the rate, so the hold stops at the
// session's usual pause; where pauses are all alike it covers them all. With fewer than kMinGaps pauses there is no
// basis and no hold (0). margin: slack on T so a pause just like the longest covered one still gets its clock.
constexpr size_t kMinGaps = 4, kWindow = 32;

inline double hold_seconds(std::vector<double> gaps, double seconds_per_save, double margin = 1.1) {
    if (gaps.size() < kMinGaps || seconds_per_save <= 0) return 0;
    std::sort(gaps.begin(), gaps.end());
    const double n = (double)gaps.size();
    double best = 0, sum_below = 0;
    for (size_t k = 0; k < gaps.size(); k++) {
        // mean(min(gap, T)) with T = gaps[k]: the k+1 smallest pauses count fully, the others count T
        sum_below += gaps[k];
        const double mean_min = (sum_below + (n - (double)(k + 1)) * gaps[k]) / n;
        const double saved = (double)(k + 1) / n;
        if (mean_min / saved <= seconds_per_save) best = gaps[k];
    }
    return best * margin;
}

// The pauses of one server, newest last, kWindow of them
class PauseWindow {
public:
    void add(double s) {
        g_.push_back(s);
        if (g_.size() > kWindow) g_.pop_front();
    }
    std::vector<double> gaps() const { return {g_.begin(), g_.end()}; }

private:
    std::deque<double> g_;
};

}  // namespace tona::server
