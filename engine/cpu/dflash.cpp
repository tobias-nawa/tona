#include "cpu/dflash.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <stdexcept>
#include <string>

#include "backend/nan_guard.h"
#include "model/quant.h"

namespace tona::ref {

namespace {

float silu(float x) { return x / (1.0f + std::exp(-x)); }

std::vector<float> dequant_all(const TensorInfo& t) {
    std::vector<float> out(t.elements());
    dequantize_row(t.type, t.data, out.data(), (int64_t)out.size());
    return out;
}

void rms_norm(float* x, int64_t n, float eps, const float* w) {
    double ss = 0;
    for (int64_t i = 0; i < n; i++) ss += (double)x[i] * x[i];
    const float scale = 1.0f / std::sqrt((float)(ss / n) + eps);
    for (int64_t i = 0; i < n; i++) x[i] = x[i] * scale * (w ? w[i] : 1.0f);
}

// NeoX RoPE over all n dimensions (M-RoPE with sections [n/2, 0, 0, 0] is exactly this)
void rope(float* x, int64_t n, int64_t pos, float base) {
    const int64_t half = n / 2;
    for (int64_t i = 0; i < half; i++) {
        const double theta = (double)pos * std::pow((double)base, -2.0 * i / n);
        const double c = std::cos(theta), s = std::sin(theta);
        const double x0 = x[i], x1 = x[i + half];
        x[i] = (float)(x0 * c - x1 * s);
        x[i + half] = (float)(x0 * s + x1 * c);
    }
}

// Y[b] = W X[b] for nb vectors; every row is unpacked once
void matmul(const TensorInfo& w, const float* X, float* Y, int nb) {
    const int64_t in = (int64_t)w.dims[0], out = (int64_t)w.dims[1];
    const GgmlTypeInfo* ti = ggml_type_info(w.type);
    const int64_t row_bytes = in / ti->block_elems * ti->block_bytes;
#pragma omp parallel
    {
        std::vector<float> row(in);
#pragma omp for schedule(static)
        for (int64_t o = 0; o < out; o++) {
            dequantize_row(w.type, w.data + o * row_bytes, row.data(), in);
            for (int b = 0; b < nb; b++) {
                double acc = 0;
                const float* x = X + (int64_t)b * in;
                for (int64_t i = 0; i < in; i++) acc += (double)row[i] * x[i];
                Y[(int64_t)b * out + o] = (float)acc;
            }
        }
    }
}

// A NaN logit breaks the order of the selector's list, which must hold top_k_ valid entries: the row without NaN (a
// copy in buf, only if there is NaN), the NaN reported
const float* nan_free_row(const float* lg, int64_t V, std::vector<float>& buf, int64_t position) {
    const size_t nan = tona::nan::count(lg, (size_t)V);
    if (nan == 0) return lg;
    buf.assign(lg, lg + V);
    tona::nan::clean(buf.data(), (size_t)V);
    tona::nan::report("DFlash selector logits", nan, std::format("position {}", position));
    return buf.data();
}

}  // namespace

DFlash::DFlash(const Gguf& g, const Qwen4ExpModel& target) : target_(target) {
    if (g.get_string("general.architecture") != "dflash") throw std::runtime_error("DFlash: not a dflash file");
    d_ = (int64_t)g.get_uint("dflash.embedding_length");
    n_head_ = (int64_t)g.get_uint("dflash.attention.head_count");
    n_kv_ = (int64_t)g.get_uint("dflash.attention.head_count_kv");
    hd_ = (int64_t)g.get_uint("dflash.attention.key_length");
    ff_ = (int64_t)g.get_uint("dflash.feed_forward_length");
    n_swa_ = g.has("dflash.attention.sliding_window") ? (int64_t)g.get_uint("dflash.attention.sliding_window") : 0;
    block_kernel_ = (int64_t)g.get_uint("dflash.conv_kernel_size");
    group_ = (int64_t)g.get_uint("dflash.conv_group_size");
    rank_ = (int64_t)g.get_uint("dflash.selector_rank");
    top_k_ = (int64_t)g.get_uint("dflash.selector_top_k");
    eps_ = (float)g.get_float("dflash.attention.layer_norm_rms_epsilon");
    rope_base_ = (float)g.get_float("dflash.rope.freq_base");
    mask_token_ = (int32_t)g.get_uint("tokenizer.ggml.mask_token_id");
    target_layers_ = g.get_int_array("dflash.target_layers");
    if (d_ != (int64_t)target.hp().d || block_kernel_ != 2)
        throw std::runtime_error("DFlash: shapes do not match the target model");
    if (g.has("dflash.attention.causal") && std::get<bool>(g.kv().at("dflash.attention.causal").v))
        throw std::runtime_error("DFlash: causal drafters are not supported");
    auto t = [&](const std::string& name) {
        const TensorInfo* p = g.find(name);
        if (!p) throw std::runtime_error("DFlash: tensor missing: " + name);
        return p;
    };
    fc_ = t("fc.weight");
    sel_hidden_ = t("selector_hidden.weight");
    sel_prev_ = t("selector_predecessor.weight");
    sel_next_ = t("selector_successor.weight");
    enc_norm_ = dequant_all(*t("enc.output_norm.weight"));
    out_norm_ = dequant_all(*t("output_norm.weight"));
    const int64_t n_layer = (int64_t)g.get_uint("dflash.block_count");
    layers_.resize(n_layer);
    for (int64_t l = 0; l < n_layer; l++) {
        Layer& L = layers_[l];
        const std::string p = "blk." + std::to_string(l) + ".";
        L.attn_norm = t(p + "attn_norm.weight"), L.q = t(p + "attn_q.weight"), L.k = t(p + "attn_k.weight");
        L.v = t(p + "attn_v.weight"), L.o = t(p + "attn_output.weight");
        L.q_norm = t(p + "attn_q_norm.weight"), L.k_norm = t(p + "attn_k_norm.weight");
        L.ffn_norm = t(p + "ffn_norm.weight"), L.gate = t(p + "ffn_gate.weight"), L.up = t(p + "ffn_up.weight");
        L.down = t(p + "ffn_down.weight");
        L.attn_conv_base = t(p + "attn_conv_base"), L.attn_conv_proj = t(p + "attn_conv_proj.weight");
        L.ffn_conv_base = t(p + "ffn_conv_base"), L.ffn_conv_proj = t(p + "ffn_conv_proj.weight");
        L.attn_norm_w = dequant_all(*L.attn_norm), L.q_norm_w = dequant_all(*L.q_norm);
        L.k_norm_w = dequant_all(*L.k_norm), L.ffn_norm_w = dequant_all(*L.ffn_norm);
        L.attn_base = dequant_all(*L.attn_conv_base), L.ffn_base = dequant_all(*L.ffn_conv_base);
    }
}

void DFlash::reset() {
    for (auto& L : layers_) L.K.clear(), L.V.clear();
}

// k: n_kv heads of hd each; norm per head, RoPE at pos
void DFlash::kv_rope_norm(Layer& L, float* k, int64_t pos) const {
    for (int64_t h = 0; h < n_kv_; h++) {
        rms_norm(k + h * hd_, hd_, eps_, L.k_norm_w.data());
        rope(k + h * hd_, hd_, pos, rope_base_);
    }
}

void DFlash::inject(const float* feat, int64_t pos) {
    std::vector<float> g(d_);
    matmul(*fc_, feat, g.data(), 1);
    rms_norm(g.data(), d_, eps_, enc_norm_.data());
    for (auto& L : layers_) {
        std::vector<float> k(n_kv_ * hd_), v(n_kv_ * hd_);
        matmul(*L.k, g.data(), k.data(), 1);
        matmul(*L.v, g.data(), v.data(), 1);
        kv_rope_norm(L, k.data(), pos);
        if ((int64_t)L.K.size() <= pos) L.K.resize(pos + 1), L.V.resize(pos + 1);
        L.K[pos] = std::move(k), L.V[pos] = std::move(v);
    }
}

// x[t][c] <- sum over tap of (dyn[t][g + G*(tap + 2*side)] + base[c + d*(tap + 2*side)]) * x[t - tap][c], g = c / group
void DFlash::conv(std::vector<float>& x, const std::vector<float>& dyn, const std::vector<float>& base, int side,
                  int n) const {
    const int64_t G = d_ / group_, P = 2 * block_kernel_ * G;
    std::vector<float> out(x.size(), 0.0f);
    for (int t = 0; t < n; t++)
        for (int64_t tap = 0; tap < std::min<int64_t>(block_kernel_, n); tap++) {
            if (t - tap < 0) continue;
            for (int64_t c = 0; c < d_; c++) {
                const float w = dyn[t * P + c / group_ + G * (tap + block_kernel_ * side)] +
                                base[c + d_ * (tap + block_kernel_ * side)];
                out[t * d_ + c] += w * x[(t - tap) * d_ + c];
            }
        }
    x = std::move(out);
}

// Scores and value rows of one (token, head): the context [max(0, p - n_swa + 1), pos), then the whole block
// (not causal)
void DFlash::gather_scores(const AttnIn& in, int t, int64_t hh, std::vector<double>& s,
                           std::vector<const float*>& vs) const {
    const int64_t kvd = n_kv_ * hd_, qd = n_head_ * hd_;
    const float scale = 1.0f / std::sqrt((float)hd_);
    const int64_t kvh = hh / (n_head_ / n_kv_), p = in.pos + t;
    const int64_t c0 = n_swa_ > 0 ? std::max<int64_t>(0, p - n_swa_ + 1) : 0;
    const float* qh = in.q.data() + t * qd + hh * hd_;
    auto add = [&](const float* kc, const float* vc) {
        double dot = 0;
        for (int64_t i = 0; i < hd_; i++) dot += (double)qh[i] * kc[i];
        s.push_back(dot * scale), vs.push_back(vc);
    };
    for (int64_t c = c0; c < in.pos; c++) add(in.L.K[c].data() + kvh * hd_, in.L.V[c].data() + kvh * hd_);
    for (int u = 0; u < in.n; u++) {
        if (n_swa_ > 0 && p - (in.pos + u) >= n_swa_) continue;
        add(in.k.data() + u * kvd + kvh * hd_, in.v.data() + u * kvd + kvh * hd_);
    }
}

std::vector<float> DFlash::attend(const Layer& L, const std::vector<float>& q, const std::vector<float>& k,
                                  const std::vector<float>& v, int64_t pos, int n) const {
    const int64_t qd = n_head_ * hd_;
    const AttnIn in{L, q, k, v, pos, n};
    std::vector<float> o(n * qd);
#pragma omp parallel for collapse(2) schedule(static)
    for (int t = 0; t < n; t++)
        for (int64_t hh = 0; hh < n_head_; hh++) {
            std::vector<double> s;
            std::vector<const float*> vs;
            gather_scores(in, t, hh, s, vs);
            const double mx = *std::max_element(s.begin(), s.end());
            double sum = 0;
            for (auto& e : s) sum += (e = std::exp(e - mx));
            float* oh = o.data() + t * qd + hh * hd_;
            for (int64_t i = 0; i < hd_; i++) {
                double acc = 0;
                for (size_t c = 0; c < s.size(); c++) acc += s[c] * vs[c][i];
                oh[i] = (float)(acc / sum);
            }
        }
    return o;
}

// Attention half of a layer: x += conv_out(attn(conv_in(norm(x))))
void DFlash::layer_attention(Layer& L, std::vector<float>& x, int64_t pos, int n) const {
    const int64_t qd = n_head_ * hd_, kvd = n_kv_ * hd_, P = 2 * block_kernel_ * (d_ / group_);
    std::vector<float> h = x;
    for (int t = 0; t < n; t++) rms_norm(h.data() + t * d_, d_, eps_, L.attn_norm_w.data());
    std::vector<float> dyn(n * P);
    matmul(*L.attn_conv_proj, h.data(), dyn.data(), n);
    conv(h, dyn, L.attn_base, 0, n);
    std::vector<float> q(n * qd), k(n * kvd), v(n * kvd);
    matmul(*L.q, h.data(), q.data(), n);
    matmul(*L.k, h.data(), k.data(), n);
    matmul(*L.v, h.data(), v.data(), n);
    for (int t = 0; t < n; t++) {
        for (int64_t hh = 0; hh < n_head_; hh++) {
            rms_norm(q.data() + t * qd + hh * hd_, hd_, eps_, L.q_norm_w.data());
            rope(q.data() + t * qd + hh * hd_, hd_, pos + t, rope_base_);
        }
        kv_rope_norm(L, k.data() + t * kvd, pos + t);
    }
    const std::vector<float> o = attend(L, q, k, v, pos, n);
    std::vector<float> a(n * d_);
    matmul(*L.o, o.data(), a.data(), n);
    conv(a, dyn, L.attn_base, 1, n);
    for (int64_t i = 0; i < n * d_; i++) x[i] += a[i];  // ffn_inp
}

// Feed-forward half of a layer: x += conv_out(down(silu(gate) * up)) of the conv_in-filtered norm(x)
void DFlash::layer_ffn(const Layer& L, std::vector<float>& x, int n) const {
    const int64_t P = 2 * block_kernel_ * (d_ / group_);
    std::vector<float> f = x;
    for (int t = 0; t < n; t++) rms_norm(f.data() + t * d_, d_, eps_, L.ffn_norm_w.data());
    std::vector<float> dynf(n * P);
    matmul(*L.ffn_conv_proj, f.data(), dynf.data(), n);
    conv(f, dynf, L.ffn_base, 0, n);
    std::vector<float> gt(n * ff_), up(n * ff_), y(n * d_);
    matmul(*L.gate, f.data(), gt.data(), n);
    matmul(*L.up, f.data(), up.data(), n);
    for (int64_t i = 0; i < n * ff_; i++) gt[i] = silu(gt[i]) * up[i];
    matmul(*L.down, gt.data(), y.data(), n);
    conv(y, dynf, L.ffn_base, 1, n);
    for (int64_t i = 0; i < n * d_; i++) x[i] += y[i];
}

std::vector<int32_t> DFlash::draft(int32_t last, int64_t pos, int n) {
    for (const auto& L : layers_)
        if ((int64_t)L.K.size() < pos) throw std::runtime_error("DFlash: context not fed up to the position");
    std::vector<float> x(n * d_);
    for (int t = 0; t < n; t++) embed(t == 0 ? last : mask_token_, x.data() + t * d_);
    for (auto& L : layers_) {
        layer_attention(L, x, pos, n);
        layer_ffn(L, x, n);
    }
    for (int t = 0; t < n; t++) rms_norm(x.data() + t * d_, d_, eps_, out_norm_.data());
    // The head only for the mask positions 1 ... n - 1
    const int64_t V = (int64_t)target_.output()->dims[1];
    std::vector<float> logits((n - 1) * V), gate((n - 1) * rank_);
    matmul(*target_.output(), x.data() + d_, logits.data(), n - 1);
    matmul(*sel_hidden_, x.data() + d_, gate.data(), n - 1);
    return select(last, logits.data(), gate.data(), n);
}

void DFlash::embed(int32_t token, float* out) const {
    const auto& emb = *target_.tok_embd();
    const GgmlTypeInfo* ti = ggml_type_info(emb.type);
    dequantize_row(emb.type, emb.data + token * (d_ / ti->block_elems * ti->block_bytes), out, d_);
}

std::vector<int32_t> DFlash::select(int32_t last, const float* logits, const float* gate, int n,
                                    std::vector<float>* probs) const {
    if (probs) probs->clear();
    const int64_t V = (int64_t)target_.output()->dims[1];
    auto row = [&](const TensorInfo& t, int64_t r) {
        std::vector<float> out(rank_);
        const GgmlTypeInfo* ti = ggml_type_info(t.type);
        dequantize_row(t.type, t.data + r * (rank_ / ti->block_elems * ti->block_bytes), out.data(), rank_);
        return out;
    };
    // Selector: top-k per position, score(p -> s) = logit(s) + sum over r of next[s]_r * prev[p]_r * gate_r; walk
    // greedily
    std::vector<int32_t> out, prev_cand{last};
    int prev_k = 0;
    for (int i = 1; i < n; i++) {
        const float* lg = logits + (i - 1) * V;
        std::vector<float> clean_row;
        lg = nan_free_row(lg, V, clean_row, i - 1);
        // Top-k in one pass (like partial_sort with ties going to the smaller index, only without sorting all V):
        // indices ascend, so on a tie the earlier insertion wins; the list is sorted descending
        std::vector<int32_t> cand;
        cand.reserve(top_k_ + 1);
        for (int64_t c = 0; c < V; c++) {
            if ((int64_t)cand.size() == top_k_ && !(lg[c] > lg[cand.back()])) continue;
            auto at = std::upper_bound(cand.begin(), cand.end(), (int32_t)c,
                                       [&](int32_t a, int32_t b) { return lg[a] > lg[b]; });
            cand.insert(at, (int32_t)c);
            if ((int64_t)cand.size() > top_k_) cand.pop_back();
        }
        const auto pr = row(*sel_prev_, prev_cand[prev_k]);
        const float* gi = gate + (i - 1) * rank_;
        int best = 0;
        double best_s = -INFINITY;
        std::vector<double> sc(top_k_);
        for (int64_t kk = 0; kk < top_k_; kk++) {
            const auto nx = row(*sel_next_, cand[kk]);
            double s = lg[cand[kk]];
            for (int64_t r = 0; r < rank_; r++) s += (double)nx[r] * pr[r] * gi[r];
            sc[kk] = s;
            if (s > best_s) best_s = s, best = (int)kk;
        }
        if (probs) {
            double z = 0;
            for (double v : sc) z += std::exp(v - best_s);
            probs->push_back((float)(1.0 / z));
        }
        out.push_back(cand[best]);
        prev_cand = cand, prev_k = best;
    }
    return out;
}

}  // namespace tona::ref
