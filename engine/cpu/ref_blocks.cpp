#include "cpu/ref_blocks.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#include "backend/nan_guard.h"

namespace tona::ref {

std::vector<float> deltanet_block(const DeltaNetWeights& w, const DeltaNetDims& D, float eps, OutGate out_gate,
                                  const float* x, const DeltaNetState& st) {
    const int64_t kd = (int64_t)D.k_dim(), vd = (int64_t)D.v_dim(), ch = 2 * kd + vd;
    const int64_t nk = (int64_t)D.k_heads, nv = (int64_t)D.v_heads, S = (int64_t)D.head_dim, K = (int64_t)D.conv;
    if (nk <= 0) throw std::runtime_error("ref::deltanet: no K heads");

    std::vector<float> qkv(ch), z(vd), b(nv), a(nv);
    matvec(*w.qkv, x, qkv.data());
    matvec(*w.gate, x, z.data());
    matvec(*w.beta, x, b.data());
    matvec(*w.alpha, x, a.data());

    // Causal depthwise conv over [K-1 old inputs; current], weight (k, channel) at channel*K + k
    const auto cw = dequant_all(*w.conv1d);
    float* cs = st.conv_state;
    std::vector<float> conv(ch);
    for (int64_t c = 0; c < ch; c++) {
        float acc = 0;
        for (int64_t k = 0; k < K - 1; k++) acc += cw[c * K + k] * cs[k * ch + c];
        acc += cw[c * K + K - 1] * qkv[c];
        conv[c] = silu(acc);
    }
    std::memmove(cs, cs + ch, (K - 2) * ch * sizeof(float));
    std::memcpy(cs + (K - 2) * ch, qkv.data(), ch * sizeof(float));

    float* q = conv.data();
    float* k = conv.data() + kd;
    const float* v = conv.data() + 2 * kd;
    // L2 norm per head as build_gdn_l2_norm: rms_norm(x, eps/n) * 1/sqrt(n)
    for (int64_t hh = 0; hh < nk; hh++) {
        for (float* p : {q + hh * S, k + hh * S}) {
            rms_norm(p, S, eps / (float)S);
            for (int64_t i = 0; i < S; i++) p[i] *= 1.0f / std::sqrt((float)S);
        }
    }

    const auto dt = dequant_all(*w.dt_bias), A = dequant_all(*w.a), wn = dequant_all(*w.ssm_norm);
    std::vector<float> o(vd);
    DeltaNetInputs in{q, k, v, z.data(), b.data(), a.data(), dt.data(), A.data(), wn.data(), nk, nv, S, eps};
    in.out_gate = out_gate;
    deltanet_recurrence(in, st.state, o.data());

    std::vector<float> y(D.d);
    matvec(*w.ssm_out, o.data(), y.data());
    return y;
}

AttentionProjection attention_project(const GatedAttentionWeights& w, const AttentionConfig& c, int64_t pos,
                                      const float* x) {
    const int64_t hd = (int64_t)c.dims.head_dim, nh = (int64_t)c.dims.n_head, nkv = (int64_t)c.dims.n_head_kv;
    std::vector<float> qfull((c.dims.gate ? 2 : 1) * nh * hd);
    AttentionProjection p;
    p.k.resize(nkv * hd), p.v.resize(nkv * hd);
    matvec(*w.q, x, qfull.data());
    matvec(*w.k, x, p.k.data());
    matvec(*w.v, x, p.v.data());
    const std::vector<float> wq = w.q_norm ? dequant_all(*w.q_norm) : std::vector<float>();
    const std::vector<float> wk = w.k_norm ? dequant_all(*w.k_norm) : std::vector<float>();
    p.q.assign(nh * hd, 0.0f), p.gate.assign(nh * hd, 0.0f);
    const int64_t stride = (c.dims.gate ? 2 : 1) * hd;
    for (int64_t hh = 0; hh < nh; hh++) {
        std::copy_n(&qfull[hh * stride], hd, &p.q[hh * hd]);
        if (c.dims.gate) std::copy_n(&qfull[hh * stride + hd], hd, &p.gate[hh * hd]);
        rms_norm(&p.q[hh * hd], hd, c.eps, wq.empty() ? nullptr : wq.data());
        rope(&p.q[hh * hd], c.rope_dims, pos, c.rope_base);
    }
    for (int64_t hh = 0; hh < nkv; hh++) {
        rms_norm(&p.k[hh * hd], hd, c.eps, wk.empty() ? nullptr : wk.data());
        rope(&p.k[hh * hd], c.rope_dims, pos, c.rope_base);
    }
    return p;
}

std::vector<float> attention_finish(const TensorInfo& wo, std::vector<float> o, const std::vector<float>& gate) {
    for (size_t i = 0; i < o.size(); i++) o[i] *= sigmoid(gate[i]);
    std::vector<float> y(wo.dims[1]);
    matvec(wo, o.data(), y.data());
    return y;
}

std::vector<float> moe_block(const MoeWeights& w, int64_t n_used, const float* x, std::vector<int>* chosen,
                             const std::string& where) {
    const int64_t d = (int64_t)w.router->dims[0];

    std::vector<float> logits(w.router->dims[1]);
    matvec(*w.router, x, logits.data());
    tona::nan::report("CPU reference: router logits", tona::nan::clean(logits.data(), logits.size()), where);
    const Routing route = route_softmax_topk(logits, n_used);

    std::vector<float> y(d, 0.0f);
    for (size_t i = 0; i < route.experts.size(); i++) {
        const auto out = swiglu_expert(*w.gate_exps, *w.up_exps, *w.down_exps, x, route.experts[i]);
        for (int64_t j = 0; j < d; j++) y[j] += route.weights[i] * out[j];
    }
    if (chosen) *chosen = route.experts;

    // Shared expert with its own sigmoid gate
    const auto so = swiglu_expert(*w.shexp_gate, *w.shexp_up, *w.shexp_down, x, 0);
    const auto wg = dequant_all(*w.shexp_gate_inp);
    double gd = 0;
    for (int64_t i = 0; i < d; i++) gd += (double)wg[i] * x[i];
    const float gate = sigmoid((float)gd);
    for (int64_t i = 0; i < d; i++) y[i] += so[i] * gate;
    return y;
}

}  // namespace tona::ref
