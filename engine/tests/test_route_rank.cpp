// route_softmax and route_top (backend/route_rank.h) against the version they replaced: softmax, then partial_sort of
// all indices by p descending and index ascending. Bit-identical probabilities and the same experts in the same order,
// also with ties (equal logits, underflow to 0) and extreme values.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

#include "backend/route_rank.h"
#include "tests/test_support.h"

namespace {

// The code of ExpertSelect::rank_token before route_rank.h
void rank_plain(const float* r, int E, int k, std::vector<float>& p, std::vector<int>& idx) {
    p.assign(E, 0.0f);
    idx.assign(E, 0);
    const float mx = *std::max_element(r, r + E);
    double sum = 0;
    for (int e = 0; e < E; e++) sum += (p[e] = std::exp(r[e] - mx));
    for (int e = 0; e < E; e++) p[e] = (float)(p[e] / sum);
    std::iota(idx.begin(), idx.end(), 0);
    const auto rank = [&](int a, int c) { return p[a] > p[c] || (p[a] == p[c] && a < c); };
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), rank);
}

void expect_same(const std::vector<float>& r, int k) {
    const int E = (int)r.size();
    std::vector<float> p0;
    std::vector<int> i0;
    rank_plain(r.data(), E, k, p0, i0);
    std::vector<float> p1(E);
    std::vector<int> i1(E, -1);
    EXPECT(tona::backend::route_softmax(r.data(), E, p1.data()) == 0);
    tona::backend::route_top(p1.data(), E, k, i1.data());
    EXPECT(std::equal(p0.begin(), p0.end(), p1.begin()));  // same bits
    EXPECT(std::equal(i0.begin(), i0.begin() + k, i1.begin()));
}

void test_random() {
    std::mt19937 g(7);
    std::normal_distribution<float> n(0.0f, 3.0f);
    for (int rep = 0; rep < 2000; rep++) {
        std::vector<float> r(256);
        for (float& v : r) v = n(g);
        expect_same(r, 8);
    }
}

// Few distinct logits: many equal p, the order of the ties decides
void test_ties() {
    std::mt19937 g(11);
    for (int rep = 0; rep < 2000; rep++) {
        const int levels = 1 + (int)(g() % 6);
        std::vector<float> r(256);
        for (float& v : r) v = (float)(g() % levels) * 0.5f;
        expect_same(r, 8);
    }
}

// Logits far apart: exp underflows to 0 for most experts, so the tail of the top-k is a tie of zeros; also equal
// maxima, large magnitudes, and other E and k
void test_extremes() {
    std::mt19937 g(5);
    for (int rep = 0; rep < 500; rep++) {
        std::vector<float> r(256);
        for (float& v : r) v = (g() % 4 == 0) ? 0.0f : -1.0e4f * (float)(g() % 3) - 200.0f;
        expect_same(r, 8);
        for (float& v : r) v = (g() % 2 ? 1.0e30f : -1.0e30f);
        expect_same(r, 8);
    }
    expect_same(std::vector<float>(256, 3.5f), 8);
    for (int E : {8, 9, 64, 128}) {
        std::vector<float> r(E);
        for (float& v : r) v = (float)(g() % 5);
        for (int k : {1, 2, 8}) expect_same(r, std::min(k, E));
    }
}

// A NaN logit is -inf (probability 0), +inf the largest float: no NaN in p, the ranking stays a strict total order,
// the unusable ones are counted, and the other logits rank as if the bad ones were -inf. An all-NaN row is uniform.
void test_unusable() {
    std::mt19937 g(3);
    std::normal_distribution<float> n(0.0f, 3.0f);
    for (int rep = 0; rep < 500; rep++) {
        std::vector<float> r(256), clean;
        for (float& v : r) v = n(g);
        for (int k = 1 + (int)(g() % 5); k > 0; k--) r[g() % 256] = std::nanf("");
        int bad = 0;
        for (float v : r) bad += v != v;
        clean = r;
        for (float& v : clean) v = v != v ? -INFINITY : v;
        std::vector<float> p(256), p_ref;
        std::vector<int> idx(8), i_ref;
        EXPECT(tona::backend::route_softmax(r.data(), 256, p.data()) == bad);
        tona::backend::route_top(p.data(), 256, 8, idx.data());
        rank_plain(clean.data(), 256, 8, p_ref, i_ref);
        EXPECT(std::equal(p_ref.begin(), p_ref.end(), p.begin()));  // -inf is exact in the plain softmax
        EXPECT(std::equal(i_ref.begin(), i_ref.begin() + 8, idx.begin()));
    }
    std::vector<float> p(256);
    std::vector<int> idx(8);
    EXPECT(tona::backend::route_softmax(std::vector<float>(256, std::nanf("")).data(), 256, p.data()) == 256);
    tona::backend::route_top(p.data(), 256, 8, idx.data());
    for (int k = 0; k < 8; k++) EXPECT(idx[k] == k && p[k] == 1.0f / 256.0f);
    std::vector<float> inf(256, 1.0f);
    inf[40] = INFINITY, inf[7] = INFINITY;  // two +inf: largest float, a tie, smaller index first
    EXPECT(tona::backend::route_softmax(inf.data(), 256, p.data()) == 2);
    tona::backend::route_top(p.data(), 256, 8, idx.data());
    EXPECT(idx[0] == 7 && idx[1] == 40);
    for (float v : p) EXPECT(v == v);
}

}  // namespace

int main() {
    return tona::test::guard("route_rank", [&] {
        test_unusable();
        test_random();
        test_ties();
        test_extremes();
        return tona::test::finish("route_rank");
    });
}
