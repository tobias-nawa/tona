#include "client/format.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <format>

#include "client/term.h"

namespace tona::client {

int to_int(const std::string& text, int fallback) {
    int value = 0;
    const char* end = text.data() + text.size();
    const auto [p, ec] = std::from_chars(text.data(), end, value);
    return ec == std::errc() && p == end ? value : fallback;
}

double to_double(const std::string& text, double fallback) {
    char* end = nullptr;
    const double value = std::strtod(text.c_str(), &end);
    return !text.empty() && end == text.c_str() + text.size() ? value : fallback;
}

std::string meter(double fraction, int w) { return meter2(fraction, 0, w); }

std::string meter2(double a, double b, int w) {
    w = std::max(w, 1);
    const int fa = std::clamp((int)std::lround(std::clamp(a, 0.0, 1.0) * w), 0, w);
    const int fb = std::clamp((int)std::lround(std::clamp(b, 0.0, 1.0) * w), 0, w - fa);
    return top::repeat("█", fa) + top::repeat("▒", fb) + top::repeat("░", w - fa - fb);
}

std::string sparkline(const std::vector<double>& values, int w, double top_value) {
    static const char* levels[9] = {" ", "▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"};
    std::string out;
    const size_t n = values.size();
    for (int x = 0; x < w; x++) {
        // column x covers the values [x * n / w, (x + 1) * n / w): their mean
        const size_t lo = n * x / w, hi = std::max(lo + 1, n * (x + 1) / w);
        double sum = 0;
        size_t count = 0;
        for (size_t i = lo; i < hi && i < n; i++) sum += values[i], count++;
        const double mean = count ? sum / (double)count : 0;
        const int k = top_value > 0 ? std::clamp((int)std::lround(mean / top_value * 8), 0, 8) : 0;
        out += levels[mean > 0 && k == 0 ? 1 : k];
    }
    return out;
}

std::string age(double seconds) {
    if (seconds < 90) return std::format("{:.0f} s", seconds);
    if (seconds < 3600 * 1.5) return std::format("{:.0f} min", seconds / 60);
    return std::format("{}:{:02d} h", (int)seconds / 3600, (int)seconds / 60 % 60);
}

std::string clock_hm(long long epoch_seconds) {
    const std::time_t t = epoch_seconds;
    char b[8];
    std::strftime(b, sizeof b, "%H:%M", std::localtime(&t));
    return b;
}

std::string project_label(const std::string& project) { return project.empty() ? "–" : project; }

std::string cell(const std::string& s, int w) {
    if (top::width(s) <= w) return top::fit(s, w);
    return top::fit(s, w - 1) + "…";
}

}  // namespace tona::client
