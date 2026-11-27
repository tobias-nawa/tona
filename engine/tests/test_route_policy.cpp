// The routing policies of backend/route_policy.h as the CPU side runs them (backend/expert_route.h): both policies
// against the float64 reference of tools/gen_route_ref.py (experts exactly, weights within float rounding), the rules
// of the sigmoid policy one by one (clean logits, bias decides the choice only, scale, floor of the sum, ties), the
// union over several tokens, and that OEA leaves the sigmoid policy alone.
#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "backend/expert_route.h"
#include "tests/route_ref_data.h"
#include "tests/test_support.h"

namespace {

using tona::backend::ExpertRoute;
using tona::backend::Gate;
using tona::backend::RoutePolicy;

constexpr int UMAX = 128;
std::string g_ref;  // tests/data/route_ref.txt

RoutePolicy policy_of(const routeref::Case& c) {
    return {c.gate == "sigmoid" ? Gate::Sigmoid : Gate::Softmax, c.scale, c.with_bias ? c.bias.data() : nullptr};
}

// One token through ExpertRoute: its experts in rank order and their weights
struct Picked {
    std::vector<int> experts;
    std::vector<float> weights;
};

Picked pick(const routeref::Case& c) {
    ExpertRoute route({c.E, c.K, 1, UMAX, policy_of(c)});
    route.rank_token(0, c.logits.data(), [](int) { return false; });
    std::vector<float> ew(UMAX, 0.0f);
    std::vector<uint32_t> um(UMAX, 0);
    route.build_union(ew.data(), um.data());
    Picked p;
    p.experts = route.uni();
    p.weights.assign(ew.begin(), ew.begin() + (long)p.experts.size());
    return p;
}

void test_reference() {
    const auto cases = routeref::load(g_ref);
    EXPECT(cases.size() >= 50);
    for (const auto& c : cases) {
        const Picked p = pick(c);
        EXPECT(p.experts == c.experts);
        EXPECT(p.weights.size() == c.weights.size());
        for (size_t i = 0; i < c.weights.size() && i < p.weights.size(); i++)
            EXPECT(std::fabs(p.weights[i] - c.weights[i]) <= 2e-6 * c.weights[i]);
    }
}

// score(r) = 1 / (1 + e^-r) within a few ulp, 0 for NaN and -inf, 1 for +inf and large logits
void test_scores() {
    using namespace tona::backend::sigmoid_route;
    for (int i = 0; i <= 162; i++) {
        const float r = -30.0f + 0.37f * (float)i;
        const double want = 1.0 / (1.0 + std::exp(-(double)r));
        EXPECT(std::fabs(score(r) - want) <= 4e-7 * std::max(want, 1e-3));
    }
    EXPECT(score(std::nanf("")) == 0.0f);
    EXPECT(score(-INFINITY) == 0.0f);
    EXPECT(score(INFINITY) == 1.0f);
    EXPECT(score(1e30f) == 1.0f);
    EXPECT(score(-1e30f) < 1e-30f);
    EXPECT(key(0.5f, std::nanf("")) == -FLT_MAX);  // a NaN bias ranks last
}

// The bias decides the choice only: a large bias on expert 0 picks it, its weight stays its score's share
void test_bias_choice() {
    routeref::Case c;
    c.E = 16, c.K = 2, c.gate = "sigmoid", c.with_bias = 1;
    c.logits.assign(16, 0.0f);
    c.logits[3] = 2.0f, c.logits[9] = 1.0f;
    c.bias.assign(16, 0.0f);
    c.bias[0] = 5.0f;
    c.scale = 1.0f;
    const Picked p = pick(c);
    EXPECT(p.experts == (std::vector<int>{0, 3}));
    const float s0 = tona::backend::sigmoid_route::score(0.0f), s3 = tona::backend::sigmoid_route::score(2.0f);
    EXPECT(std::fabs(p.weights[0] - s0 / (s0 + s3)) < 1e-6f);
    EXPECT(std::fabs(p.weights[0] + p.weights[1] - 1.0f) < 1e-6f);
}

// The weights add up to the scale (up to rounding); the floor keeps a sum of tiny scores from blowing them up
void test_scale_and_floor() {
    routeref::Case c;
    c.E = 32, c.K = 4, c.gate = "sigmoid", c.with_bias = 0, c.scale = 5.0f;
    c.logits = routeref::make_logits(5, 32, 0);
    const Picked p = pick(c);
    float sum = 0;
    for (float w : p.weights) sum += w;
    EXPECT(std::fabs(sum - 5.0f) < 1e-5f);
    c.logits.assign(32, -60.0f);  // all scores ~1e-26: the sum is under the floor
    const Picked q = pick(c);
    for (float w : q.weights) EXPECT(w >= 0.0f && w < 1e-18f);
}

// Unusable logits are counted and rank last (score 0); a row of NaN picks the experts 0 ... K - 1, never an index
// outside the experts
void test_unusable_logits() {
    std::vector<float> r(64, std::nanf("")), score(64), key(64);
    EXPECT(tona::backend::route_sigmoid(r.data(), nullptr, 64, score.data(), key.data()) == 64);
    routeref::Case c;
    c.E = 64, c.K = 8, c.gate = "sigmoid", c.with_bias = 0, c.scale = 1.0f, c.logits = r;
    const Picked p = pick(c);
    EXPECT(p.experts == (std::vector<int>{0, 1, 2, 3, 4, 5, 6, 7}));
    c.logits[40] = INFINITY;
    EXPECT(pick(c).experts[0] == 40);
}

// Several tokens: the union holds each expert once, every token's weights sum to the scale, the masks name the tokens
void test_union() {
    constexpr int NB = 3;
    const routeref::Case base = routeref::load(g_ref)[8];  // sigmoid-496
    ExpertRoute route({base.E, base.K, NB, UMAX, policy_of(base)});
    for (int b = 0; b < NB; b++) {
        const auto r = routeref::make_logits(base.seed + 7 * (uint64_t)b, base.E, 0);
        route.rank_token(b, r.data(), [](int) { return false; });
    }
    std::vector<float> ew((size_t)NB * UMAX, 0.0f);
    std::vector<uint32_t> um(UMAX, 0);
    route.build_union(ew.data(), um.data());
    const auto& uni = route.uni();
    EXPECT(std::set<int>(uni.begin(), uni.end()).size() == uni.size());
    for (int b = 0; b < NB; b++) {
        float sum = 0;
        for (size_t u = 0; u < uni.size(); u++) sum += ew[(size_t)b * UMAX + u];
        EXPECT(std::fabs(sum - base.scale) < 1e-4f);
    }
    for (size_t u = 0; u < uni.size(); u++)
        for (int b = 0; b < NB; b++) EXPECT(((um[u] >> b) & 1u) == (ew[(size_t)b * UMAX + u] != 0.0f ? 1u : 0u));
}

// OEA ranks by probability: with the sigmoid policy it does nothing, the choice stays the policy's
void test_oea_off() {
    const routeref::Case c = routeref::load(g_ref)[8];
    ExpertRoute route({c.E, c.K, 2, UMAX, policy_of(c)});
    route.begin_oea(3);
    for (int b = 0; b < 2; b++) route.rank_token(b, c.logits.data(), [](int) { return true; });
    EXPECT(std::vector<int>(route.ranked(0), route.ranked(0) + c.K) == c.experts);
    EXPECT(route.oea_swapped() == 0 && route.oea_short() == 0);
}

}  // namespace

int main(int argc, char** argv) {
    g_ref = argc > 1 ? argv[1] : "tests/data/route_ref.txt";
    return tona::test::run("test_route_policy", [] {
        test_reference();
        test_scores();
        test_bias_choice();
        test_scale_and_floor();
        test_unusable_logits();
        test_union();
        test_oea_off();
    });
}
