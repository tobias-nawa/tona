// Host-side numbers of a logit row that every backend needs the same way.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace tona::backend {

// -log p of token next in a row of n logits: maximum, sum of exp in double, log. Used per prompt token of a
// perplexity run and by the checks, not per decoded token.
inline double neg_log_p(const float* lg, size_t n, int32_t next) {
    const float mx = *std::max_element(lg, lg + n);
    double sum = 0;
    for (size_t i = 0; i < n; i++) sum += std::exp((double)lg[i] - mx);
    return std::log(sum) + mx - lg[next];
}

}  // namespace tona::backend
