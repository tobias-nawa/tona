#include "cpu/ref_ops.h"

#include <algorithm>
#include <cmath>
#include <numeric>

#include "backend/env.h"
#include "backend/rope_yarn.h"
#include "model/quant.h"

namespace tona::ref {

std::vector<float> dequant(const TensorInfo& t, int64_t row, int64_t n) {
    std::vector<float> out(n);
    dequantize_rows(t.type, t.data, row, 1, n, out.data());
    return out;
}

std::vector<float> dequant_all(const TensorInfo& t) {
    std::vector<float> out(t.elements());
    const auto k = (int64_t)t.dims[0];
    dequantize_rows(t.type, t.data, 0, (int64_t)out.size() / k, k, out.data());
    return out;
}

// TONA_REF_Q8ACT=1: the input vector is rounded the way the GPU decode kernels round it (blocks of 32, scale
// max|x| / 127, round to nearest), to tell the oracle's precision from the kernels' activation rounding
static std::vector<float> q8_rounded(const float* x, int64_t n) {
    std::vector<float> r(x, x + n);
    for (int64_t b = 0; b + 32 <= n; b += 32) {
        float amax = 0;
        for (int i = 0; i < 32; i++) amax = std::max(amax, std::fabs(x[b + i]));
        const float d = amax / 127.0f;
        for (int i = 0; i < 32; i++) r[b + i] = d > 0 ? std::nearbyint(x[b + i] / d) * d : 0.0f;
    }
    return r;
}

void matvec(const TensorInfo& w, const float* x, float* y, int64_t expert) {
    const int64_t in = (int64_t)w.dims[0], out = (int64_t)w.dims[1];
    static const bool q8act = env::flag("TONA_REF_Q8ACT");
    std::vector<float> xr;
    if (q8act && w.type != 0 && w.type != 1 && w.type != 30 /* F32, F16, BF16: float path */)
        xr = q8_rounded(x, in), x = xr.data();
    const GgmlTypeInfo* ti = ggml_type_info(w.type);
    const int64_t row_bytes = in / ti->block_elems * ti->block_bytes;
    const uint8_t* base = w.data + expert * out * row_bytes;
#pragma omp parallel
    {
        std::vector<float> row(in);
#pragma omp for schedule(static)
        for (int64_t o = 0; o < out; o++) {
            dequantize_rows(w.type, base, o, 1, in, row.data());
            double acc = 0;
            for (int64_t i = 0; i < in; i++) acc += (double)row[i] * x[i];
            y[o] = (float)acc;
        }
    }
}

void rms_norm(float* x, int64_t n, float eps, const float* w) {
    double ss = 0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

// For plain text tokens IMRoPE with sections [11, 11, 10] is exactly this (research/06 section 5). The YaRN table
// scales the frequency per pair and cos/sin by mscale.
void rope(float* x, int64_t n_rot, int64_t pos, float base) {
    const int64_t half = n_rot / 2;
    const tona::rope::YarnTable& yarn = tona::rope::yarn_active(base);
    for (int64_t i = 0; i < half; i++) {
        const auto [c, s] = tona::rope::pair_cos_sin(yarn, n_rot, pos, base, i);
        const double x0 = x[i], x1 = x[i + half];
        x[i] = (float)(x0 * c - x1 * s);
        x[i + half] = (float)(x0 * s + x1 * c);
    }
}

void deltanet_recurrence(const DeltaNetInputs& in, float* state, float* o) {
    const int64_t S = in.head_dim, nk = in.k_heads;
    const float qscale = 1.0f / std::sqrt((float)S);
    for (int64_t j = 0; j < in.v_heads; j++) {
        const float beta = sigmoid(in.beta[j]);
        const float alpha = std::exp(softplus(in.alpha[j] + in.dt_bias[j]) * in.a[j]);
        const float* kj = in.k + (j % nk) * S;  // GGUF: V head j uses K head j mod nk
        const float* qj = in.q + (j % nk) * S;
        const float* vj = in.v + j * S;
        float* st = state + j * S * S;  // st[vi * S + ki]
        for (int64_t i = 0; i < S * S; i++) st[i] *= alpha;
        for (int64_t vi = 0; vi < S; vi++) {
            float* row = st + vi * S;
            double sk = 0;
            for (int64_t ki = 0; ki < S; ki++) sk += (double)row[ki] * kj[ki];
            const float dlt = (vj[vi] - (float)sk) * beta;
            double ov = 0;
            for (int64_t ki = 0; ki < S; ki++) {
                row[ki] += kj[ki] * dlt;
                ov += (double)row[ki] * qj[ki];
            }
            o[j * S + vi] = (float)ov * qscale;
        }
        // RMSNorm per head, weight used directly (not 1+w), then the output gate
        rms_norm(&o[j * S], S, in.eps, in.norm_w);
        for (int64_t vi = 0; vi < S; vi++) {
            const float z = in.z[j * S + vi];
            o[j * S + vi] *= in.out_gate == OutGate::Silu ? silu(z) : sigmoid(z);
        }
    }
}

std::vector<float> attend_cells(const AttendShape& s, const float* q, const std::vector<float>& Kf,
                                const std::vector<float>& Vf, const std::vector<int64_t>& cells) {
    // TONA_REF_Q8KV=1: K and V as the GPU's int8 cache holds them (blocks of 32), see TONA_REF_Q8ACT
    static const bool q8kv = env::flag("TONA_REF_Q8KV");
    const std::vector<float> Kq = q8kv ? q8_rounded(Kf.data(), (int64_t)Kf.size()) : std::vector<float>();
    const std::vector<float> Vq = q8kv ? q8_rounded(Vf.data(), (int64_t)Vf.size()) : std::vector<float>();
    const std::vector<float>& K = q8kv ? Kq : Kf;
    const std::vector<float>& V = q8kv ? Vq : Vf;
    const int64_t hd = s.head_dim, nh = s.n_head, nkv = s.n_head_kv;
    std::vector<float> o(nh * hd, 0.0f);
    const float scale = 1.0f / std::sqrt((float)hd);
    std::vector<double> sc(cells.size());
    for (int64_t hh = 0; hh < nh; hh++) {
        const int64_t kvh = hh / (nh / nkv);
        double mx = -INFINITY;
        for (size_t c = 0; c < cells.size(); c++) {
            const float* kc = &K[(cells[c] * nkv + kvh) * hd];
            double dot = 0;
            for (int64_t i = 0; i < hd; i++) dot += (double)q[hh * hd + i] * kc[i];
            sc[c] = dot * scale;
            mx = std::max(mx, sc[c]);
        }
        double sum = 0;
        for (auto& v : sc) sum += (v = std::exp(v - mx));
        for (size_t c = 0; c < cells.size(); c++) {
            const float* vc = &V[(cells[c] * nkv + kvh) * hd];
            const float p = (float)(sc[c] / sum);
            for (int64_t i = 0; i < hd; i++) o[hh * hd + i] += p * vc[i];
        }
    }
    return o;
}

Routing route_softmax_topk(const std::vector<float>& logits, int64_t n_used) {
    const int64_t E = (int64_t)logits.size();
    const float mx = *std::max_element(logits.begin(), logits.end());
    double sum = 0;
    std::vector<float> p(E);
    for (int64_t e = 0; e < E; e++) sum += (p[e] = std::exp(logits[e] - mx));
    for (auto& v : p) v = (float)(v / sum);
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::partial_sort(idx.begin(), idx.begin() + n_used, idx.end(),
                      [&](int a, int b) { return p[a] > p[b] || (p[a] == p[b] && a < b); });
    idx.resize(n_used);
    double wsum = 0;
    for (int e : idx) wsum += p[e];
    Routing r;
    r.experts = idx;
    for (int e : idx) r.weights.push_back((float)(p[e] / wsum));
    return r;
}

std::vector<float> swiglu_expert(const TensorInfo& gate, const TensorInfo& up, const TensorInfo& down, const float* x,
                                 int64_t expert) {
    const int64_t ff = (int64_t)gate.dims[1], d = (int64_t)down.dims[1];
    std::vector<float> g(ff), u(ff), out(d);
    matvec(gate, x, g.data(), expert);
    matvec(up, x, u.data(), expert);
    for (int64_t i = 0; i < ff; i++) g[i] = silu(g[i]) * u[i];
    matvec(down, g.data(), out.data(), expert);
    return out;
}

}  // namespace tona::ref
