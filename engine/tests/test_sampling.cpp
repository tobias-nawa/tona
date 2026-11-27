// Sampling distributions: temperature, top_k, top_p, the window below the maximum, draws and the residual draw of
// speculative sampling (server/sampling.h).
#include <cmath>
#include <random>
#include <vector>

#include "server/sampling.h"
#include "tests/test_support.h"

using tona::server::Dist;
using tona::server::Sampling;

namespace {

double sum(const Dist& d) {
    double s = 0;
    for (double p : d.p) s += p;
    return s;
}

bool near(double a, double b, double tol = 1e-9) { return std::fabs(a - b) <= tol; }

Sampling smp(float t, int top_k = 0, float top_p = 1) {
    Sampling s;
    s.temperature = t, s.top_k = top_k, s.top_p = top_p;
    return s;
}

void test_softmax_order_and_window() {
    const float lg[5] = {0.0f, 2.0f, 1.0f, -100.0f, 2.0f};
    const Dist d = tona::server::make_dist(lg, 5, smp(1.0f));
    // The token 30 * T below the maximum drops out; equal logits keep the lower token first
    EXPECT(d.tok.size() == 4 && d.tok[0] == 1 && d.tok[1] == 4 && d.tok[2] == 2 && d.tok[3] == 0);
    EXPECT(near(sum(d), 1.0));
    const double z = 2 * std::exp(0.0) + std::exp(-1.0) + std::exp(-2.0);
    EXPECT(near(d.p[0], 1 / z) && near(d.p[2], std::exp(-1.0) / z));
    EXPECT(near(d.prob(2), d.p[2]) && d.prob(3) == 0 && d.prob(99) == 0);
}

void test_temperature() {
    const float lg[3] = {3.0f, 2.0f, 1.0f};
    const Dist hot = tona::server::make_dist(lg, 3, smp(5.0f)), cold = tona::server::make_dist(lg, 3, smp(0.1f));
    EXPECT(cold.p[0] > 0.999 && hot.p[0] < 0.5);
    EXPECT(near(sum(hot), 1.0) && near(sum(cold), 1.0));
}

void test_top_k_and_top_p() {
    const float lg[4] = {3.0f, 2.0f, 1.0f, 0.0f};
    const Dist k = tona::server::make_dist(lg, 4, smp(1.0f, 2));
    EXPECT(k.tok.size() == 2 && k.tok[0] == 0 && k.tok[1] == 1 && near(sum(k), 1.0));
    EXPECT(tona::server::make_dist(lg, 4, smp(1.0f, 10)).tok.size() == 4);  // top_k above the vocabulary
    const Dist full = tona::server::make_dist(lg, 4, smp(1.0f));
    // top_p keeps the smallest prefix reaching the mass, at least one token
    const Dist p = tona::server::make_dist(lg, 4, smp(1.0f, 0, 0.8f));
    double acc = 0;
    size_t n = 0;
    while (acc < 0.8) acc += full.p[n++];
    EXPECT(p.tok.size() == n && near(sum(p), 1.0));
    const Dist one = tona::server::make_dist(lg, 4, smp(1.0f, 0, 0.01f));
    EXPECT(one.tok.size() == 1 && near(one.p[0], 1.0));
}

void test_rows_and_argmax() {
    const float lg[4] = {9.0f, 1.0f, 5.0f, 3.0f};
    // Row 0 is unused: its high logit must not set the maximum or become a candidate
    const Dist d =
        tona::server::make_dist_rows(lg, 4, [](size_t i) { return i == 0 ? -1 : (int32_t)(100 + i); }, smp(1.0f));
    EXPECT(d.tok.size() == 3 && d.tok[0] == 102 && d.prob(100) == 0);
    EXPECT(tona::server::argmax(lg, 4) == 0);
    const float tie[3] = {1.0f, 2.0f, 2.0f};
    EXPECT(tona::server::argmax(tie, 3) == 1);
}

void test_sample_follows_the_distribution() {
    const float lg[3] = {std::log(0.6f), std::log(0.3f), std::log(0.1f)};
    const Dist d = tona::server::make_dist(lg, 3, smp(1.0f));
    std::mt19937_64 rng(1234);
    int n[3] = {0, 0, 0};
    constexpr int N = 20000;
    for (int i = 0; i < N; i++) n[tona::server::sample(d, rng)]++;
    EXPECT(near(n[0] / (double)N, 0.6, 0.02) && near(n[1] / (double)N, 0.3, 0.02) && near(n[2] / (double)N, 0.1, 0.02));
    std::mt19937_64 a(5), b(5);
    EXPECT(tona::server::sample(d, a) == tona::server::sample(d, b));  // same seed, same token
}

void test_sample_skips_a_token() {
    const float lg[3] = {2.0f, 1.0f, 0.0f};
    const Dist d = tona::server::make_dist(lg, 3, smp(1.0f));
    std::mt19937_64 rng(9);
    bool hit = false;
    for (int i = 0; i < 500; i++) hit |= tona::server::sample(d, rng, 0) == 0;
    EXPECT(!hit);
    const float one[1] = {1.0f};
    const Dist single = tona::server::make_dist(one, 1, smp(1.0f));
    EXPECT(tona::server::sample(single, rng, 0) == 0);  // nothing left to draw: the first token
}

void test_residual_draw() {
    const float lg[2] = {0.0f, 0.0f};
    const Dist p = tona::server::make_dist(lg, 2, smp(1.0f));  // 0.5 / 0.5
    Dist q;
    q.tok = {0}, q.p = {1.0};  // the draft was token 0 with certainty
    std::mt19937_64 rng(3);
    // max(0, p - q) leaves only token 1
    for (int i = 0; i < 200; i++) EXPECT(tona::server::sample_residual(p, rng, q) == 1);
    // p == q: nothing is left, so the draw falls back to p
    int zero = 0;
    for (int i = 0; i < 400; i++) zero += tona::server::sample_residual(p, rng, p) == 0;
    EXPECT(zero > 100 && zero < 300);
}

}  // namespace

int main() {
    return tona::test::run("test_sampling", [] {
        test_softmax_order_and_window();
        test_temperature();
        test_top_k_and_top_p();
        test_rows_and_argmax();
        test_sample_follows_the_distribution();
        test_sample_skips_a_token();
        test_residual_draw();
    });
}
