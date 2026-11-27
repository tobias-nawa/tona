// Sampling distributions for the generator. Drafts (drafter argmax, history) are fixed tokens, which makes
// speculative sampling exact and simple: accept a draft d with probability p(d), otherwise draw from p without d
// (Leviathan et al., arXiv 2211.17192, with q as a point mass). The output distribution stays the model's.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "backend/nan_guard.h"

namespace tona::server {

// temperature <= 0: greedy
struct Sampling {
    float temperature = 0, top_p = 1;
    int top_k = 0;
    uint64_t seed = 0;  // 0: random
    // Thinking budget: the answer starts inside the <think> block; after this many tokens in it the engine appends
    // Qwen's closing sentence and </think> (budget forcing, as llama.cpp --reasoning-budget). -1: unlimited
    int think_budget = -1;
};

// Distribution over the candidates after temperature, top_k and top_p
struct Dist {
    std::vector<int32_t> tok;
    std::vector<double> p;  // descending, sums to 1
    double prob(int32_t t) const {
        for (size_t i = 0; i < tok.size(); i++)
            if (tok[i] == t) return p[i];
        return 0;
    }
};

using Candidates = std::vector<std::pair<float, int32_t>>;  // (logit, token)

// The scan of argmax: the first index of the largest value (NaN skipped, std::max_element returns the index of a
// leading NaN; 0 for a row without a number) and the NaN count, which the caller reports. A pure function, so rows
// can be scanned in parallel.
struct ArgmaxRow {
    int32_t best;
    size_t nan;
};

inline ArgmaxRow argmax_scan(const float* r, size_t n) {
    size_t best = n, nan = 0;
    for (size_t i = 0; i < n; i++) {
        if (r[i] != r[i]) {
            nan++;
            continue;
        }
        if (best == n || r[i] > r[best]) best = i;
    }
    return {best == n ? 0 : (int32_t)best, nan};
}

// NaN is reported (once per request), never swallowed.
inline int32_t argmax(const float* r, size_t n) {
    const ArgmaxRow a = argmax_scan(r, n);
    if (a.nan) tona::nan::report("greedy argmax", a.nan);
    return a.best;
}

// Distribution from candidates; mx is the largest logit, the softmax is taken relative to it (sampling.cpp)
Dist dist_from_candidates(Candidates& c, float mx, const Sampling& s);

// Rows r of the logits are tokens tok_of(r) (< 0: row unused). Tokens more than 30 * T below the maximum drop out
// (together less than V * e^-30 = 2e-8 of the mass).
template <class F>
Dist make_dist_rows(const float* lg, size_t V, F tok_of, const Sampling& s) {
    // NaN is the smallest value and +-inf the largest and smallest finite one (nan_guard.h): an empty candidate list
    // or exp(inf - inf) would leave the sampler with no token or a NaN distribution
    float mx = -INFINITY;
    size_t nan = 0;
    for (size_t i = 0; i < V; i++) {
        nan += lg[i] != lg[i];
        const float v = tona::nan::clean_one(lg[i]);
        if (v > mx && tok_of(i) >= 0) mx = v;
    }
    if (nan) tona::nan::report("sampler logits", nan);
    const float cut = mx - 30.0f * s.temperature;
    Candidates c;
    for (size_t i = 0; i < V; i++) {
        const float v = tona::nan::clean_one(lg[i]);
        if (v >= cut) {
            const int32_t t = tok_of(i);
            if (t >= 0) c.emplace_back(v, t);
        }
    }
    if (c.empty())  // no row with a token that has a value: the first token, as a certain draw
        for (size_t i = 0; i < V && c.empty(); i++)
            if (tok_of(i) >= 0) c.push_back({0.0f, tok_of(i)}), mx = 0.0f;
    return dist_from_candidates(c, mx, s);
}

inline Dist make_dist(const float* lg, size_t V, const Sampling& s) {
    return make_dist_rows(lg, V, [](size_t i) { return (int32_t)i; }, s);
}

// Draw a token, skipping `skip` (-1: none)
template <class R>
int32_t sample(const Dist& d, R& rng, int32_t skip = -1) {
    double tot = 0;
    for (size_t i = 0; i < d.tok.size(); i++)
        if (d.tok[i] != skip) tot += d.p[i];
    if (d.tok.empty()) return 0;
    if (tot <= 0) return d.tok[0];
    double u = std::uniform_real_distribution<double>(0, tot)(rng);
    for (size_t i = 0; i < d.tok.size(); i++) {
        if (d.tok[i] == skip) continue;
        u -= d.p[i];
        if (u <= 0) return d.tok[i];
    }
    for (size_t i = d.tok.size(); i-- > 0;)
        if (d.tok[i] != skip) return d.tok[i];
    return d.tok[0];
}

// After rejecting a draft drawn from q: draw from max(0, p - q), normalized (Leviathan et al., arXiv 2211.17192)
template <class R>
int32_t sample_residual(const Dist& d, R& rng, const Dist& q) {
    std::vector<double> r(d.tok.size());
    double tot = 0;
    for (size_t i = 0; i < d.tok.size(); i++) tot += r[i] = std::max(0.0, d.p[i] - q.prob(d.tok[i]));
    if (tot <= 0) return sample(d, rng);
    double u = std::uniform_real_distribution<double>(0, tot)(rng);
    for (size_t i = 0; i < d.tok.size(); i++) {
        if (r[i] <= 0) continue;
        u -= r[i];
        if (u <= 0) return d.tok[i];
    }
    for (size_t i = d.tok.size(); i-- > 0;)
        if (r[i] > 0) return d.tok[i];
    return d.tok[0];
}

}  // namespace tona::server
