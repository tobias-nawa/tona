// NaN and inf in the logits of the CPU selections: backend/nan_guard.h and the sampler (server/sampling.h).
// Rows of NaN, one NaN, +-inf and all +inf must give valid tokens and finite distributions; finite rows must give what
// the code before the change gave (the old argmax and make_dist are copied below).
#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include "backend/nan_guard.h"
#include "server/sampling.h"
#include "tests/test_support.h"

using tona::server::Dist;
using tona::server::Sampling;

namespace {

constexpr float kNan = std::numeric_limits<float>::quiet_NaN(), kInf = INFINITY;  // constant: cannot throw

// The sampler as it was before the change
Dist old_make_dist(const float* lg, size_t V, const Sampling& s) {
    float mx = -INFINITY;
    for (size_t i = 0; i < V; i++) mx = std::max(lg[i], mx);
    const float cut = mx - 30.0f * s.temperature;
    tona::server::Candidates c;
    for (size_t i = 0; i < V; i++)
        if (lg[i] >= cut) c.emplace_back(lg[i], (int32_t)i);
    return tona::server::dist_from_candidates(c, mx, s);
}

Sampling smp(float t, int top_k = 0) {
    Sampling s;
    s.temperature = t, s.top_k = top_k;
    return s;
}

std::vector<float> row(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 4.0f);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

bool valid(const Dist& d, int32_t V) {
    double s = 0;
    for (size_t i = 0; i < d.tok.size(); i++) {
        if (d.tok[i] < 0 || d.tok[i] >= V || !std::isfinite(d.p[i]) || d.p[i] < 0) return false;
        s += d.p[i];
    }
    return !d.tok.empty() && std::fabs(s - 1.0) < 1e-9;
}

void test_clean() {
    float v[6] = {1.0f, kNan, kInf, -kInf, -0.0f, 3.5f};
    EXPECT(tona::nan::count(v, 6) == 1);
    EXPECT(tona::nan::clean(v, 6) == 1);
    EXPECT(v[0] == 1.0f && v[1] == -FLT_MAX && v[2] == FLT_MAX && v[3] == -FLT_MAX && v[5] == 3.5f);
    EXPECT(std::signbit(v[4]) && v[4] == 0.0f);  // finite values keep their bits
    EXPECT(tona::nan::count(v, 6) == 0);
}

void test_report_once_per_epoch() {
    tona::nan::new_epoch();
    const uint64_t t0 = tona::nan::total();
    EXPECT(tona::nan::report("site a", 3, "layer 1"));
    EXPECT(!tona::nan::report("site a", 2));  // same site, same epoch: counted, not printed again
    EXPECT(tona::nan::report("site b", 1));
    EXPECT(!tona::nan::report("site c", 0));
    EXPECT(tona::nan::total() == t0 + 6);
    tona::nan::new_epoch();
    EXPECT(tona::nan::report("site a", 1));
}

void test_argmax() {
    for (uint32_t seed : {1u, 2u, 3u}) {
        std::vector<float> v = row(5000, seed);
        v[10] = v[400] = 99.0f;  // a tie: the first one
        EXPECT(tona::server::argmax(v.data(), v.size()) == (int32_t)(std::max_element(v.begin(), v.end()) - v.begin()));
    }
    std::vector<float> v = row(1000, 5);
    const int32_t best = tona::server::argmax(v.data(), v.size());
    v[0] = kNan;  // std::max_element would answer 0 here
    EXPECT(tona::server::argmax(v.data(), v.size()) == best || best == 0);
    v[best] = kNan;
    const int32_t second = tona::server::argmax(v.data(), v.size());
    EXPECT(second != best && second != 0 && !std::isnan(v[second]));
    EXPECT(tona::server::argmax(std::vector<float>(100, kNan).data(), 100) == 0);
    v = row(100, 6);
    v[40] = kInf;
    EXPECT(tona::server::argmax(v.data(), v.size()) == 40);
}

void test_finite_rows_unchanged() {
    for (float t : {0.3f, 1.0f, 2.0f})
        for (int top_k : {0, 20}) {
            const std::vector<float> v = row(2000, 9);
            const Dist a = old_make_dist(v.data(), v.size(), smp(t, top_k));
            const Dist b = tona::server::make_dist(v.data(), v.size(), smp(t, top_k));
            EXPECT(a.tok == b.tok && a.p == b.p);
        }
}

void test_bad_rows() {
    const int V = 3000;
    std::vector<std::vector<float>> rows;
    rows.emplace_back(V, kNan);  // everything NaN: no candidate before the change
    rows.emplace_back(V, kInf);
    rows.emplace_back(V, -kInf);
    rows.push_back(row(V, 1)), rows.back()[7] = kNan;
    rows.push_back(row(V, 2)), rows.back()[8] = kInf, rows.back()[9] = -kInf;
    rows.push_back(row(V, 3)), rows.back()[100] = rows.back()[200] = kInf;
    std::mt19937 rng(1);
    for (const auto& r : rows)
        for (float t : {0.7f, 1.0f})
            for (int top_k : {0, 10}) {
                const Dist d = tona::server::make_dist(r.data(), V, smp(t, top_k));
                EXPECT(valid(d, V));
                const int32_t tok = tona::server::sample(d, rng);
                EXPECT(tok >= 0 && tok < V);
            }
    // the NaN of the one-NaN row is not a candidate; the best finite token stays on top
    const Dist d = tona::server::make_dist(rows[3].data(), V, smp(1.0f));
    EXPECT(d.prob(7) == 0 || d.p[0] >= d.prob(7));
    // rows of the head without a token: the first row that has one
    const auto tok_of = [](size_t i) { return i < 10 ? -1 : (int32_t)i; };
    const Dist e = tona::server::make_dist_rows(rows[0].data(), V, tok_of, smp(1.0f));
    EXPECT(valid(e, V) && e.tok[0] == 10);
    // no row has a token at all: an empty distribution, and a draw from it does not read outside it
    const Dist none = tona::server::make_dist_rows(rows[0].data(), V, [](size_t) { return -1; }, smp(1.0f));
    EXPECT(none.tok.empty() && tona::server::sample(none, rng) == 0);
}

}  // namespace

int main() {
    return tona::test::run("nan_guard", [] {
        test_clean();
        test_report_once_per_epoch();
        test_argmax();
        test_finite_rows_unchanged();
        test_bad_rows();
    });
}
