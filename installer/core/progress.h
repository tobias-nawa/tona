// Progress of a running step (model download, SHA-256 check) for the screen: structured numbers instead of text lines,
// a moving-average rate and the remaining time. The producers (model_fetch.cpp) report through a sink that the
// screen sets for its worker thread; without a sink (command line, tests) nothing happens and the text lines stay.
#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <string>
#include <utility>

namespace tona::install {

enum class Phase { Download, Verify };

struct Progress {
    Phase phase = Phase::Download;
    uint64_t done = 0, total = 0;  // bytes of the whole model in this phase
};

using ProgressFn = std::function<void(const Progress&)>;

// The sink of the calling thread; an empty function removes it
void set_progress_sink(ProgressFn sink);
void report_progress(const Progress& p);

// Rate as a moving average over the last kWindow seconds. Time is passed in, so tests need no clock.
class RateMeter {
public:
    static constexpr double kWindow = 5.0;  // s: smooths the bursts of a network line, still follows a slowdown
    static constexpr double kStable = 3.0;  // s of samples before the rate and the remaining time are shown

    void add(double seconds, uint64_t bytes);
    void reset() { samples_.clear(); }
    double rate() const;  // bytes per second, 0 until stable
    // Seconds until `total` at the current rate, negative while the rate is not stable
    double remaining(uint64_t total) const;

private:
    std::deque<std::pair<double, uint64_t>> samples_;
};

// "45.2 MB/s" (decimal)
std::string rate_text(double bytes_per_second);
// "2 h 03 min", "4 min 07 s", "35 s"
std::string remaining_text(double seconds);
// "12.3 GB of 58.0 GB  45.2 MB/s  2 h 03 min left"; rate and time are left out while the meter is not stable
std::string progress_text(const Progress& p, const RateMeter& m);
int percent_of(const Progress& p);

}  // namespace tona::install
