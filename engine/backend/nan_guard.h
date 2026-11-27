// NaN and infinity in logits that choose something (an expert, a token, a block). Every comparison with NaN is false,
// so a maximum or a sort over a row with NaN either keeps its start index (an index far outside the array) or
// breaks the strict weak order that std::partial_sort needs (undefined behaviour). The rules:
//   - NaN counts as the smallest value, +-inf as the largest and smallest finite value (clean), so that the
//     result of every selection is a valid index and nothing divides infinity by infinity;
//   - for finite input nothing changes;
//   - every NaN is counted and reported (report), once per request or prefill and site, never only under TONA_DIAG.
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>

namespace tona::nan {

// A logit as the selections see it: NaN -> -FLT_MAX, +-inf -> +-FLT_MAX, finite values unchanged
inline float clean_one(float x) { return x != x ? -FLT_MAX : std::clamp(x, -FLT_MAX, FLT_MAX); }

// In place; returns how many values were NaN (counted for the report; +-inf are clamped silently, they are legal)
inline size_t clean(float* v, size_t n) {
    size_t nan = 0;
    for (size_t i = 0; i < n; i++) {
        nan += v[i] != v[i];
        v[i] = clean_one(v[i]);
    }
    return nan;
}

// Number of NaN values, without changing them
inline size_t count(const float* v, size_t n) {
    size_t nan = 0;
    for (size_t i = 0; i < n; i++) nan += v[i] != v[i];
    return nan;
}

namespace detail {
struct State {
    std::mutex mu;
    std::set<std::string> seen;  // sites reported in this epoch
    uint64_t total = 0;          // NaN values counted since the start of the process
};
inline State& state() {
    static State s;
    return s;
}
}  // namespace detail

// A new request or prefill begins: every site may report once more
inline void new_epoch() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.seen.clear();
}

// Values counted since the start of the process (for tests and the statistics)
inline uint64_t total() {
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mu);
    return s.total;
}

// Always prints (stderr), but each site only once per epoch. `where` names the context (layer, position, ...).
// Returns true if the line was printed.
inline bool report(const char* site, uint64_t n, const std::string& where = {}) {
    if (n == 0) return false;
    auto& s = detail::state();
    std::lock_guard<std::mutex> lock(s.mu);
    s.total += n;
    if (!s.seen.insert(site).second) return false;
    std::fprintf(stderr, "WARNING: %llu NaN value(s) in %s%s%s; treated as the smallest value\n", (unsigned long long)n,
                 site, where.empty() ? "" : ", ", where.c_str());
    return true;
}

}  // namespace tona::nan
