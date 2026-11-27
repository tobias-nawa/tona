// Softmax and ranking of the router logits of one token, on the CPU while the kernel waits at the router. Part of
// ExpertSelect (mega_select.cpp); a header of its own so that a unit test can hold it against the plain version.
#pragma once

#include <algorithm>
#include <cfloat>
#include <cmath>

#include "backend/route_policy.h"

namespace tona::backend {

// p[e] = softmax(r)[e] as float (sum in double, in the order of e). Returns the number of logits that are not usable:
// a NaN counts as -inf (probability 0) and +inf as the largest float, so that no probability is NaN and the ranking
// below keeps a strict total order (partial_sort on NaN is undefined). A row without any usable logit becomes
// uniform. For rows of finite logits (and -inf) the result is the same bits as the plain softmax.
inline int route_softmax(const float* r, int E, float* p) {
    constexpr int kMax = 512;  // E <= 512 as in op_route
    float clean[kMax];
    int bad = 0;
    for (int e = 0; e < E; e++) bad += !(r[e] == r[e]) || r[e] == INFINITY;
    if (bad > 0) {
        for (int e = 0; e < E; e++) clean[e] = r[e] != r[e] ? -INFINITY : std::min(r[e], FLT_MAX);
        r = clean;
    }
    const float mx = *std::max_element(r, r + E);
    if (mx == -INFINITY) {  // all -inf: nothing to rank by
        std::fill(p, p + E, 1.0f / (float)E);
        return bad;
    }
    double sum = 0;
    for (int e = 0; e < E; e++) sum += (p[e] = std::exp(r[e] - mx));
    for (int e = 0; e < E; e++) p[e] = (float)(p[e] / sum);
    return bad;
}

// Sigmoid policy (route_policy.h): score[e] and choice key[e] = score + bias[e] (bias null: none) of one token. Returns
// the number of unusable logits as route_softmax does; they get score 0, so the ranking stays a total order. The
// scores are the same bits as the GPU computes.
inline int route_sigmoid(const float* r, const float* bias, int E, float* score, float* key) {
    int bad = 0;
    for (int e = 0; e < E; e++) {
        bad += !(r[e] == r[e]) || r[e] == INFINITY;
        score[e] = sigmoid_route::score(r[e]);
        key[e] = sigmoid_route::key(score[e], bias ? bias[e] : 0.0f);
    }
    return bad;
}

// The k best experts by p, descending; on a tie the smaller index first (the order partial_sort gave with that
// comparison, which is a strict total order, so the result is the same). One pass over p: an expert enters only when
// it beats the current k-th, which most do not; k >= 1. idx[k..E) is not written. p must not contain NaN (route_softmax
// guarantees that).
inline void route_top(const float* p, int E, int k, int* idx) {
    int n = 0;
    for (int e = 0; e < E; e++) {
        const float v = p[e];
        // Scanning e upwards, a later expert with an equal p ranks behind the earlier one: strict > is the tie rule
        if (n == k && !(v > p[idx[k - 1]])) continue;
        int j = n < k ? n++ : k - 1;
        while (j > 0 && v > p[idx[j - 1]]) idx[j] = idx[j - 1], j--;
        idx[j] = e;
    }
}

}  // namespace tona::backend
