#include "server/sampling.h"

namespace tona::server {

namespace {

// At T = 1 almost the whole vocabulary is inside the window: a full sort costs 7.8 ms per position, the best top_k
// only 0.5 ms (strict ordering, so the same selection and order)
void order_candidates(Candidates& c, int top_k) {
    const auto cmp = [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    };
    if (top_k > 0 && c.size() > (size_t)top_k) {
        std::partial_sort(c.begin(), c.begin() + top_k, c.end(), cmp);
        c.resize(top_k);
    } else {
        std::sort(c.begin(), c.end(), cmp);
    }
}

void apply_top_p(Dist& d, float top_p) {
    double acc = 0;
    size_t keep = 0;
    while (keep < d.p.size() && (keep == 0 || acc < top_p)) acc += d.p[keep++];
    d.tok.resize(keep), d.p.resize(keep);
    for (auto& v : d.p) v /= acc;
}

}  // namespace

Dist dist_from_candidates(Candidates& c, float mx, const Sampling& s) {
    order_candidates(c, s.top_k);
    Dist d;
    double sum = 0;
    for (const auto& [l, t] : c) {
        d.tok.push_back(t);
        d.p.push_back(std::exp((double)(l - mx) / s.temperature));
        sum += d.p.back();
    }
    for (auto& v : d.p) v /= sum;
    if (s.top_p < 1) apply_top_p(d, s.top_p);
    return d;
}

}  // namespace tona::server
