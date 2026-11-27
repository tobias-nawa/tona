#include "core/progress.h"

#include <cmath>
#include <format>

#include "core/space.h"

namespace tona::install {

namespace {
thread_local ProgressFn g_sink;
}

void set_progress_sink(ProgressFn sink) { g_sink = std::move(sink); }

void report_progress(const Progress& p) {
    if (g_sink) g_sink(p);
}

void RateMeter::add(double seconds, uint64_t bytes) {
    if (!samples_.empty() && bytes < samples_.back().second) samples_.clear();  // a restart from zero
    samples_.emplace_back(seconds, bytes);
    while (samples_.size() > 2 && samples_.front().first < seconds - kWindow) samples_.pop_front();
}

double RateMeter::rate() const {
    if (samples_.size() < 2) return 0;
    const auto& [t0, b0] = samples_.front();
    const auto& [t1, b1] = samples_.back();
    if (t1 - t0 < kStable) return 0;
    return static_cast<double>(b1 - b0) / (t1 - t0);
}

double RateMeter::remaining(uint64_t total) const {
    const double r = rate();
    if (r <= 0 || samples_.empty()) return -1;
    const uint64_t have = samples_.back().second;
    return have >= total ? 0 : static_cast<double>(total - have) / r;
}

std::string rate_text(double bps) { return std::format("{:.1f} MB/s", bps / 1e6); }

std::string remaining_text(double seconds) {
    const long s = std::lround(seconds);
    if (s >= 3600) return std::format("{} h {:02} min", s / 3600, s % 3600 / 60);
    if (s >= 60) return std::format("{} min {:02} s", s / 60, s % 60);
    return std::format("{} s", s);
}

int percent_of(const Progress& p) {
    return p.total ? static_cast<int>(static_cast<double>(p.done) * 100 / static_cast<double>(p.total)) : 0;
}

std::string progress_text(const Progress& p, const RateMeter& m) {
    std::string t = size_text(static_cast<double>(p.done)) + " of " + size_text(static_cast<double>(p.total));
    if (m.rate() <= 0) return t;
    t += "  " + rate_text(m.rate());
    const double left = m.remaining(p.total);
    if (left >= 0) t += "  " + remaining_text(left) + " left";
    return t;
}

}  // namespace tona::install
