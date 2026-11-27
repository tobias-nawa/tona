// The shared operations of the CPU reference (cpu/ref_ops.h) on small hand-computed cases: no model, no GPU.
#include <algorithm>
#include <cmath>
#include <vector>

#include "cpu/ref_ops.h"
#include "tests/test_support.h"

namespace ref = tona::ref;

namespace {

bool near(double a, double b, double tol = 1e-5) { return std::fabs(a - b) <= tol * (1.0 + std::fabs(b)); }

// An F32 tensor in ggml order [in, out]: row o holds the weights of output o
tona::TensorInfo f32_tensor(const std::vector<float>& data, std::vector<uint64_t> dims) {
    tona::TensorInfo t;
    t.dims = std::move(dims);
    t.type = 0;
    t.bytes = data.size() * sizeof(float);
    t.data = reinterpret_cast<const uint8_t*>(data.data());
    return t;
}

void test_routing() {
    // Experts 1 and 3 tie for the top: the lower index comes first; the weights are renormalized over the chosen ones
    const ref::Routing r = ref::route_softmax_topk({1.0f, 3.0f, 2.0f, 3.0f}, 2);
    EXPECT((r.experts == std::vector<int>{1, 3}));
    EXPECT(near(r.weights[0], 0.5) && near(r.weights[1], 0.5));
    // Three experts of different logits: weights follow the softmax ratios and sum to one
    const ref::Routing s = ref::route_softmax_topk({0.0f, 2.0f, 1.0f}, 2);
    EXPECT((s.experts == std::vector<int>{1, 2}));
    EXPECT(near(s.weights[0] + s.weights[1], 1.0));
    EXPECT(near(s.weights[0] / s.weights[1], std::exp(1.0)));
}

void test_attention() {
    // One KV head, head dim 2, two cached positions; a zero query attends uniformly to the chosen cells
    const std::vector<float> K = {1, 0, 0, 1}, V = {2, 4, 6, 8};
    const float q[2] = {0, 0};
    const auto both = ref::attend_cells({1, 1, 2}, q, K, V, {0, 1});
    EXPECT(near(both[0], 4.0) && near(both[1], 6.0));
    // Only the cells in the list count: the second position is ignored
    const auto first = ref::attend_cells({1, 1, 2}, q, K, V, {0});
    EXPECT(near(first[0], 2.0) && near(first[1], 4.0));
    // A query along key 0 prefers position 0 by softmax(1 / sqrt(2)) against 0
    const float q0[2] = {1, 0};
    const auto sharp = ref::attend_cells({1, 1, 2}, q0, K, V, {0, 1});
    const double e = std::exp(1.0 / std::sqrt(2.0)), p0 = e / (e + 1.0);
    EXPECT(near(sharp[0], p0 * 2 + (1 - p0) * 6) && near(sharp[1], p0 * 4 + (1 - p0) * 8));
    // Two query heads share one KV head
    const float qq[4] = {0, 0, 0, 0};
    const auto grouped = ref::attend_cells({2, 1, 2}, qq, K, V, {0, 1});
    EXPECT(grouped.size() == 4 && near(grouped[0], grouped[2]) && near(grouped[1], grouped[3]));
}

void test_norm() {
    // {3, 4}: mean of squares 12.5, so the scale is 1 / sqrt(12.5 + eps)
    float x[2] = {3, 4};
    ref::rms_norm(x, 2, 0.0f);
    EXPECT(near(x[0], 3 / std::sqrt(12.5)) && near(x[1], 4 / std::sqrt(12.5)));
    float y[2] = {3, 4};
    const float w[2] = {2, 0.5f};
    ref::rms_norm(y, 2, 0.0f, w);
    EXPECT(near(y[0], 2 * 3 / std::sqrt(12.5)) && near(y[1], 0.5 * 4 / std::sqrt(12.5)));
}

void test_deltanet() {
    // One K head and one V head of size 2, zero state: the delta rule writes beta * v along k, the readout along q is
    // that row's k-component, then the per-head norm and the sigmoid gate
    const float q[2] = {1, 0}, k[2] = {1, 0}, v[2] = {1, 2}, z[2] = {0, 0};
    const float beta_logit = 0.0f, alpha_logit = 0.0f, dt_bias = 0.0f, a = -1.0f, norm_w[2] = {1, 1};
    const ref::DeltaNetInputs in{q, k, v, z, &beta_logit, &alpha_logit, &dt_bias, &a, norm_w, 1, 1, 2, 1e-6f};
    std::vector<float> state(4, 0.0f), o(2);
    ref::deltanet_recurrence(in, state.data(), o.data());
    // beta = sigmoid(0) = 0.5; state[vi][ki] = 0.5 * v[vi] * k[ki]
    EXPECT(near(state[0], 0.5) && near(state[1], 0.0) && near(state[2], 1.0) && near(state[3], 0.0));
    float expect[2] = {0.5f / std::sqrt(2.0f), 1.0f / std::sqrt(2.0f)};
    ref::rms_norm(expect, 2, 1e-6f, norm_w);
    EXPECT(near(o[0], expect[0] * 0.5) && near(o[1], expect[1] * 0.5));  // gate sigmoid(0) = 0.5
    // A second token decays the state by alpha = exp(softplus(0) * -1) before the update
    ref::deltanet_recurrence(in, state.data(), o.data());
    const double alpha = std::exp(std::log1p(std::exp(0.0)) * -1.0);  // exp(ln 2 * -1) = 0.5
    // Row vi: decayed s' = s * alpha, then s' + beta * (v - s') along k: 0.5 + 0.25 alpha and 1 + 0.5 alpha
    EXPECT(near(state[0], 0.5 + 0.25 * alpha) && near(state[2], 1.0 + 0.5 * alpha));
}

void test_deltanet_silu_gate() {
    // Same case with the Qwen3.5 gate: the normalized readout is multiplied by silu(z), not sigmoid(z)
    const float q[2] = {1, 0}, k[2] = {1, 0}, v[2] = {1, 2}, z[2] = {2.0f, -2.0f};
    const float beta_logit = 0.0f, alpha_logit = 0.0f, dt_bias = 0.0f, a = -1.0f, norm_w[2] = {1, 1};
    ref::DeltaNetInputs in{q, k, v, z, &beta_logit, &alpha_logit, &dt_bias, &a, norm_w, 1, 1, 2, 1e-6f};
    std::vector<float> state(4, 0.0f), plain(2), silu_out(2);
    ref::deltanet_recurrence(in, state.data(), plain.data());
    in.out_gate = ref::OutGate::Silu;
    std::fill(state.begin(), state.end(), 0.0f);
    ref::deltanet_recurrence(in, state.data(), silu_out.data());
    for (int i = 0; i < 2; i++) EXPECT(near(silu_out[i], plain[i] / ref::sigmoid(z[i]) * ref::silu(z[i])));
}

void test_swiglu() {
    // d = 2, ff = 2: gate rows (1, 0) and (0, 1), up rows (1, 1) and (2, 0), down rows (1, 1) and (0, 1)
    const std::vector<float> gate = {1, 0, 0, 1}, up = {1, 1, 2, 0}, down = {1, 1, 0, 1};
    const auto g = f32_tensor(gate, {2, 2}), u = f32_tensor(up, {2, 2}), d = f32_tensor(down, {2, 2});
    const float x[2] = {1.0f, 2.0f};
    const auto y = ref::swiglu_expert(g, u, d, x, 0);
    const double h0 = ref::silu(1.0f) * 3.0, h1 = ref::silu(2.0f) * 2.0;  // gate(x) = (1, 2), up(x) = (3, 2)
    EXPECT(y.size() == 2 && near(y[0], h0 + h1) && near(y[1], h1));
}

void test_matvec_and_dequant() {
    const std::vector<float> w = {1, 2, 3, 4, 5, 6};  // [in 3, out 2]: row 0 = (1,2,3), row 1 = (4,5,6)
    const auto t = f32_tensor(w, {3, 2});
    const float x[3] = {1, 1, 1};
    float y[2];
    ref::matvec(t, x, y);
    EXPECT(near(y[0], 6) && near(y[1], 15));
    const auto row = ref::dequant(t, 1, 3);
    EXPECT(row.size() == 3 && near(row[0], 4) && near(row[2], 6));
    EXPECT(ref::dequant_all(t).size() == 6);
    EXPECT(near(ref::sigmoid(0.0f), 0.5) && near(ref::softplus(30.0f), 30.0) && near(ref::silu(0.0f), 0.0));
}

}  // namespace

int main() {
    return tona::test::run("ref_ops", [] {
        test_routing();
        test_attention();
        test_norm();
        test_deltanet();
        test_deltanet_silu_gate();
        test_swiglu();
        test_matvec_and_dequant();
    });
}
