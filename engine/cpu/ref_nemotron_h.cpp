#include "cpu/ref_nemotron_h.h"

#include <algorithm>
#include <format>
#include <numeric>
#include <stdexcept>

#include "backend/nan_guard.h"
#include "backend/route_policy.h"
#include "cpu/ref_mamba2.h"
#include "cpu/ref_ops.h"

namespace tona::ref {

namespace {

// relu(x)^2, the activation of the experts and the shared expert (LLM_FFN_RELU_SQR)
float relu_sq(float v) { return v > 0.0f ? v * v : 0.0f; }

// down(relu(up(x))^2) for one expert of the stacked tensors, or (expert 0) for a plain 2D pair
std::vector<float> relu_sq_expert(const TensorInfo& up, const TensorInfo& down, const float* x, int64_t expert) {
    const int64_t ff = (int64_t)up.dims[1], out_dim = (int64_t)down.dims[1];
    std::vector<float> u(ff), y(out_dim);
    matvec(up, x, u.data(), expert);
    for (float& v : u) v = relu_sq(v);
    matvec(down, u.data(), y.data(), expert);
    return y;
}

// The sigmoid policy of backend/route_policy.h, the one definition the GPU routing uses too: the top n_used of
// sigmoid(logit) + bias (ties to the lower index), weights score / max(sum of the chosen scores, 2^-14) * scale
Routing route_sigmoid_bias(const std::vector<float>& logits, const float* bias, int64_t n_used, float scale) {
    namespace sr = tona::backend::sigmoid_route;
    const int64_t E = (int64_t)logits.size();
    std::vector<float> s(E), key(E);
    for (int64_t e = 0; e < E; e++) {
        s[e] = sr::score(logits[e]);
        key[e] = sr::key(s[e], bias ? bias[e] : 0.0f);
    }
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + n_used, idx.end(),
                      [&](int a, int b) { return key[a] > key[b] || (key[a] == key[b] && a < b); });
    idx.resize(n_used);
    float sum = 0.0f;
    for (int e : idx) sum += s[e];  // in rank order, as the policy states
    Routing r;
    r.experts = idx;
    for (int e : idx) r.weights.push_back(sr::weight(s[e], sum, scale));
    return r;
}

}  // namespace

NemotronHOracle::NemotronHOracle(const NemotronHMoeModel& m) : m_(m), h_(m.hp()) { reset(); }

void NemotronHOracle::reset() {
    pos_ = 0;
    const Mamba2Dims D = h_.mamba2();
    ssm_state_.assign(h_.n_layer, {});
    conv_state_.assign(h_.n_layer, {});
    kv_k_.assign(h_.n_layer, {});
    kv_v_.assign(h_.n_layer, {});
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        if (m_.layers()[l].kind != BoltBlockKind::Mamba2) continue;
        ssm_state_[l].assign((size_t)D.inner * D.state, 0.0f);
        conv_state_[l].assign((size_t)(D.conv - 1) * D.conv_dim(), 0.0f);
    }
}

// Attention over every cached position, no RoPE and no QK norm: the new K and V join the cache first
std::vector<float> NemotronHOracle::attention(uint32_t l, const std::vector<float>& x) {
    const NemotronLayer& L = m_.layers()[l];
    const int64_t hd = h_.head_dim, nh = h_.n_head, nkv = L.n_kv;
    std::vector<float> q(nh * hd), k(nkv * hd), v(nkv * hd);
    matvec(*L.attn.q, x.data(), q.data());
    matvec(*L.attn.k, x.data(), k.data());
    matvec(*L.attn.v, x.data(), v.data());
    kv_k_[l].insert(kv_k_[l].end(), k.begin(), k.end());
    kv_v_[l].insert(kv_v_[l].end(), v.begin(), v.end());
    std::vector<int64_t> cells(pos_ + 1);
    std::iota(cells.begin(), cells.end(), 0);
    const std::vector<float> o = attend_cells({nh, nkv, hd}, q.data(), kv_k_[l], kv_v_[l], cells);
    std::vector<float> y(h_.d);
    matvec(*L.attn.o, o.data(), y.data());
    return y;
}

// Latent MoE: the router reads the full-width input, the experts work on its projection into the latent space and
// their sum is projected back; the shared expert reads the full-width input. Both add up.
std::vector<float> NemotronHOracle::moe(uint32_t l, const std::vector<float>& x, std::vector<int>* chosen) {
    const NemotronLayer& L = m_.layers()[l];
    const MoeWeights& w = L.moe;
    std::vector<float> logits(L.n_expert);
    matvec(*w.router, x.data(), logits.data());
    nan::report("CPU reference: router logits", nan::clean(logits.data(), logits.size()),
                std::format("block {}, position {}", l, pos_));
    const auto bias = dequant_all(*w.router_bias);
    const Routing route = route_sigmoid_bias(logits, bias.data(), L.n_used, h_.expert_scale);

    std::vector<float> latent(h_.latent ? h_.latent : h_.d);
    if (w.latent_down)
        matvec(*w.latent_down, x.data(), latent.data());
    else
        latent = x;
    std::vector<float> mixed(latent.size(), 0.0f);
    for (size_t i = 0; i < route.experts.size(); i++) {
        const auto out = relu_sq_expert(*w.up_exps, *w.down_exps, latent.data(), route.experts[i]);
        for (size_t j = 0; j < mixed.size(); j++) mixed[j] += route.weights[i] * out[j];
    }
    std::vector<float> y(h_.d);
    if (w.latent_up)
        matvec(*w.latent_up, mixed.data(), y.data());
    else
        y = mixed;

    const auto shared = relu_sq_expert(*w.shexp_up, *w.shexp_down, x.data(), 0);
    for (size_t j = 0; j < y.size(); j++) y[j] += shared[j];
    if (chosen) *chosen = route.experts;
    return y;
}

std::vector<float> NemotronHOracle::step(int32_t token, Trace* trace) {
    if (token < 0 || token >= (int32_t)h_.vocab) throw std::runtime_error("ref::step: token outside the vocabulary");
    std::vector<float> R = dequant(*m_.tok_embd(), token, h_.d);

    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const NemotronLayer& L = m_.layers()[l];
        std::vector<float> xn(R);
        rms_norm(xn.data(), (int64_t)xn.size(), h_.rms_eps, dequant_all(*L.attn_norm).data());
        std::vector<float> y;
        std::vector<int> chosen;
        switch (L.kind) {
            case BoltBlockKind::Mamba2:
                y = mamba2_block(L.ssm, h_.mamba2(), h_.rms_eps, xn.data(),
                                 {ssm_state_[l].data(), conv_state_[l].data()});
                break;
            case BoltBlockKind::Attention:
                y = attention(l, xn);
                break;
            case BoltBlockKind::Moe:
                y = moe(l, xn, trace ? &chosen : nullptr);
                break;
        }
        for (size_t i = 0; i < R.size(); i++) R[i] += y[i];
        if (trace) {
            trace->residual.push_back(R);
            if (L.kind == BoltBlockKind::Moe) trace->experts.push_back(std::move(chosen));
        }
    }

    std::vector<float> xn(R);
    rms_norm(xn.data(), (int64_t)xn.size(), h_.rms_eps, dequant_all(*m_.output_norm()).data());
    std::vector<float> logits(h_.vocab);
    matvec(*m_.output(), xn.data(), logits.data());
    pos_++;
    return logits;
}

}  // namespace tona::ref
