#include "cpu/ref_qwen4exp.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

#include "backend/nan_guard.h"
#include "cpu/ref_blocks.h"
#include "cpu/ref_helper.h"
#include "cpu/ref_ops.h"
#include "model/ple.h"
#include "model/quant.h"

namespace tona::ref {

Qwen4ExpOracle::Qwen4ExpOracle(const Qwen4ExpModel& m) : m_(m), h_(m.hp()) { reset(); }

void Qwen4ExpOracle::reset() {
    pos_ = 0;
    tokens_.clear();
    const int64_t hc_dim = (int64_t)h_.n_hc * h_.d;
    ple_hist_.assign((size_t)(h_.ple_conv - 1) * h_.ple_ngram, std::vector<float>(hc_dim, 0.0f));
    const int64_t conv_ch = 2 * h_.dn_k_dim() + h_.dn_v_dim();
    dn_state_.assign(h_.n_layer, {});
    dn_conv_.assign(h_.n_layer, {});
    kv_k_.assign(h_.n_layer, {});
    kv_v_.assign(h_.n_layer, {});
    idx_raw_.assign(h_.n_layer, {});
    idx_pool_.assign(h_.n_layer, {});
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        if (h_.is_qsa(l)) continue;
        dn_state_[l].assign((size_t)h_.dn_v_heads * h_.dn_head_dim * h_.dn_head_dim, 0.0f);
        dn_conv_[l].assign((size_t)(h_.dn_conv - 1) * conv_ch, 0.0f);
    }
}

void Qwen4ExpOracle::hc_mix(const std::vector<float>& R, const GatedResidual& gr, std::vector<float>& mixed,
                            std::vector<float>* inject) const {
    const int64_t d = h_.d, hc = h_.n_hc;
    const auto w = dequant_all(*gr.norm);
    std::vector<float> xn(R);
    for (int64_t c = 0; c < hc; c++) rms_norm(&xn[c * d], d, h_.rms_eps, &w[c * d]);

    std::vector<float> lo(h_.hc_rank), gate(hc * d);
    matvec(*gr.down, xn.data(), lo.data());
    for (auto& v : lo) v = silu(v / (float)hc);
    matvec(*gr.up, lo.data(), gate.data());

    mixed.assign(d, 0.0f);
    for (int64_t c = 0; c < hc; c++)
        for (int64_t i = 0; i < d; i++) mixed[i] += xn[c * d + i] * sigmoid(gate[c * d + i]);
    for (auto& v : mixed) v /= (float)hc;

    if (inject) {
        inject->assign(hc, 0.0f);
        matvec(*gr.inject, xn.data(), inject->data());
    }
}

void Qwen4ExpOracle::hc_combine(std::vector<float>& R, const std::vector<float>& y,
                                const std::vector<float>& inject) const {
    const int64_t d = h_.d, hc = h_.n_hc;
    for (int64_t c = 0; c < hc; c++) {
        const float s = 2.0f * sigmoid(inject[c] / (float)hc);  // 2*sigmoid centers the write gate on 1
        for (int64_t i = 0; i < d; i++) R[c * d + i] += y[i] * s;
    }
}

// Rows of the n-gram table for the hashed context of the newest token (model/ple.h), concatenated over the heads
std::vector<float> Qwen4ExpOracle::ple_lookup() const {
    std::vector<int64_t> rows(h_.ple_heads);
    tona::ple::rows_at(h_, tokens_.data(), (int64_t)tokens_.size() - 1,
                       rows.data());  // tokens_ already holds the token
    std::vector<float> e(h_.d);
    for (uint32_t head = 0; head < h_.ple_heads; head++) {
        const auto r = dequant(*m_.ple_embd(), rows[head], h_.ple_dim);
        std::copy(r.begin(), r.end(), e.begin() + (int64_t)head * h_.ple_dim);
    }
    return e;
}

// val scaled per stream by a sigmoid gate of the key/query agreement
std::vector<float> Qwen4ExpOracle::ple_gated(const std::vector<float>& R, const Layer& L,
                                             const std::vector<float>& e) const {
    const int64_t d = h_.d, hc = h_.n_hc, hc_dim = hc * d;
    std::vector<float> key(hc_dim), val(d);
    matvec(*L.ple_key, e.data(), key.data());
    matvec(*L.ple_value, e.data(), val.data());
    const auto w_key = dequant_all(*L.ple_norm_key), w_query = dequant_all(*L.ple_norm_query);

    std::vector<float> query(R), gated(hc_dim);
    for (int64_t c = 0; c < hc; c++) {
        rms_norm(&key[c * d], d, h_.rms_eps, &w_key[c * d]);
        rms_norm(&query[c * d], d, h_.rms_eps, &w_query[c * d]);
        double dot = 0;
        for (int64_t i = 0; i < d; i++) dot += (double)key[c * d + i] * query[c * d + i];
        const float s = (float)dot / std::sqrt((float)d);
        const float mag = std::sqrt(std::max(std::fabs(s), 1e-6f));
        const float sgn = (float)((s > 0) - (s < 0));  // ggml_sgn: 0 for 0
        const float gate = sigmoid(sgn * mag);
        for (int64_t i = 0; i < d; i++) gated[c * d + i] = val[i] * gate;
    }
    return gated;
}

void Qwen4ExpOracle::ple(std::vector<float>& R) {
    const Layer& L = m_.layers()[h_.ple_layer];
    const int64_t d = h_.d, hc = h_.n_hc, hc_dim = hc * d, n = h_.ple_ngram;

    const std::vector<float> e = ple_lookup();
    const std::vector<float> gated = ple_gated(R, L, e);
    const auto w_conv = dequant_all(*L.ple_norm_conv), conv = dequant_all(*L.ple_conv1d);

    std::vector<float> normed(gated);
    for (int64_t c = 0; c < hc; c++) rms_norm(&normed[c * d], d, h_.rms_eps, &w_conv[c * d]);

    // Causal depthwise conv with dilation n: tap k reads (K-1-k)*n positions back; weight (k, channel) is at
    // channel*K + k. ple_hist_ holds the last (K-1)*n inputs, oldest first.
    const int64_t K = h_.ple_conv, H = (int64_t)ple_hist_.size();
    for (int64_t ch = 0; ch < hc_dim; ch++) {
        float acc = 0;
        for (int64_t k = 0; k < K; k++) {
            const int64_t back = (K - 1 - k) * n;
            const float x = back == 0 ? normed[ch] : ple_hist_[H - back][ch];
            acc += conv[ch * K + k] * x;
        }
        R[ch] += gated[ch] + silu(acc);
    }
    ple_hist_.erase(ple_hist_.begin());
    ple_hist_.push_back(std::move(normed));
}

std::vector<float> Qwen4ExpOracle::deltanet(uint32_t l, const std::vector<float>& x) {
    const Layer& L = m_.layers()[l];
    const DeltaNetWeights w{L.qkv, L.gate, L.beta, L.alpha, L.conv1d, L.dt_bias, L.a, L.ssm_norm, L.ssm_out};
    const DeltaNetDims D{h_.d, h_.dn_k_heads, h_.dn_v_heads, h_.dn_head_dim, h_.dn_conv};
    return deltanet_block(w, D, h_.rms_eps, OutGate::Sigmoid, x.data(), {dn_state_[l].data(), dn_conv_[l].data()});
}

// Indexer: the raw key goes into the cache; a token that completes a block triggers pooling of that block
void Qwen4ExpOracle::index_update(uint32_t l, const std::vector<float>& x) {
    const Layer& L = m_.layers()[l];
    const int64_t idim = h_.idx_dim, blk = h_.idx_block, pos = pos_;
    std::vector<float> kraw(idim);
    matvec(*L.idx_k, x.data(), kraw.data());
    idx_raw_[l].insert(idx_raw_[l].end(), kraw.begin(), kraw.end());
    if ((pos + 1) % blk != 0) return;
    const int64_t b0 = pos + 1 - blk;
    std::vector<float> pooled(idim, 0.0f);
    for (int64_t m = 0; m < blk; m++)
        for (int64_t i = 0; i < idim; i++) pooled[i] += idx_raw_[l][(b0 + m) * idim + i];
    for (auto& v : pooled) v *= 1.0f / (float)blk;
    const auto kn = dequant_all(*L.idx_k_norm);
    rms_norm(pooled.data(), idim, h_.rms_eps, kn.data());
    rope(pooled.data(), h_.rope_dims, b0, h_.rope_base);  // position of the block's first token
    idx_pool_[l].insert(idx_pool_[l].end(), pooled.begin(), pooled.end());
}

// Scores the blocks: sum over heads of ReLU(q_h . k_b) / sqrt(idim); the cells are the top-k blocks plus the
// block in progress (tail)
std::vector<int64_t> Qwen4ExpOracle::index_cells(uint32_t l, const std::vector<float>& x) {
    const Layer& L = m_.layers()[l];
    const int64_t idim = h_.idx_dim, blk = h_.idx_block, pos = pos_;
    std::vector<float> qi(h_.idx_heads * idim);
    matvec(*L.idx_q, x.data(), qi.data());
    const auto qn = dequant_all(*L.idx_q_norm);
    for (uint32_t hh = 0; hh < h_.idx_heads; hh++) {
        rms_norm(&qi[hh * idim], idim, h_.rms_eps, qn.data());
        rope(&qi[hh * idim], h_.rope_dims, pos, h_.rope_base);
    }

    const int64_t n_pool = (int64_t)idx_pool_[l].size() / idim;
    std::vector<std::pair<float, int64_t>> scored(n_pool);
    for (int64_t b = 0; b < n_pool; b++) {
        double s = 0;
        for (uint32_t hh = 0; hh < h_.idx_heads; hh++) {
            double dot = 0;
            for (int64_t i = 0; i < idim; i++) dot += (double)qi[hh * idim + i] * idx_pool_[l][b * idim + i];
            s += std::max(dot, 0.0);
        }
        scored[b] = {(float)(s / std::sqrt((double)idim)), b};
    }
    // NaN scores break the order of the sort below
    size_t nan = 0;
    for (auto& sc : scored) nan += sc.first != sc.first, sc.first = tona::nan::clean_one(sc.first);
    tona::nan::report("CPU reference: block scores", nan, std::format("layer {}, position {}", l, pos));
    const int64_t n_top = std::min<int64_t>(n_pool, h_.idx_top_k / blk);
    std::partial_sort(scored.begin(), scored.begin() + n_top, scored.end(), [](const auto& a, const auto& b) {
        return a.first > b.first || (a.first == b.first && a.second < b.second);
    });
    std::vector<int64_t> cells;
    for (int64_t t = 0; t < n_top; t++)
        for (int64_t m = 0; m < blk; m++) cells.push_back(scored[t].second * blk + m);
    for (int64_t p = n_pool * blk; p <= pos; p++) cells.push_back(p);
    return cells;
}

// Q with its gate (per head [q 256 | gate 256]), K and V; K and V go into the cache
void Qwen4ExpOracle::qsa_project(uint32_t l, const std::vector<float>& x, std::vector<float>& q,
                                 std::vector<float>& gate) {
    const Layer& L = m_.layers()[l];
    GatedAttentionWeights w;
    w.q = L.q, w.k = L.k, w.v = L.v, w.o = L.o, w.q_norm = L.q_norm, w.k_norm = L.k_norm;
    const AttentionConfig cfg{
        {h_.d, h_.n_head, h_.n_head_kv, h_.head_dim, true, true, 0, 0}, h_.rms_eps, h_.rope_dims, h_.rope_base};
    AttentionProjection p = attention_project(w, cfg, pos_, x.data());
    q = std::move(p.q), gate = std::move(p.gate);
    kv_k_[l].insert(kv_k_[l].end(), p.k.begin(), p.k.end());
    kv_v_[l].insert(kv_v_[l].end(), p.v.begin(), p.v.end());
}

std::vector<float> Qwen4ExpOracle::qsa(uint32_t l, const std::vector<float>& x) {
    const Layer& L = m_.layers()[l];
    index_update(l, x);
    const std::vector<int64_t> cells = index_cells(l, x);
    std::vector<float> q, gate;
    qsa_project(l, x, q, gate);
    std::vector<float> o = attend_cells({(int64_t)h_.n_head, (int64_t)h_.n_head_kv, (int64_t)h_.head_dim}, q.data(),
                                        kv_k_[l], kv_v_[l], cells);
    return attention_finish(*L.o, std::move(o), gate);
}

std::vector<float> Qwen4ExpOracle::moe(uint32_t l, const std::vector<float>& x, Trace* trace) {
    const Layer& L = m_.layers()[l];
    // The model's own routing, through the block shared with the other architectures
    if (draft_.n_used == 0 && !draft_.helper && !(trace && trace->want_moe)) {
        MoeWeights w;
        w.router = L.router, w.gate_exps = L.gate_exps, w.up_exps = L.up_exps, w.down_exps = L.down_exps;
        w.shexp_gate_inp = L.shexp_gate_inp, w.shexp_gate = L.shexp_gate, w.shexp_up = L.shexp_up;
        w.shexp_down = L.shexp_down;
        std::vector<int> chosen;
        auto y = moe_block(w, h_.n_expert_used, x.data(), trace ? &chosen : nullptr,
                           std::format("layer {}, position {}", l, pos_));
        if (trace) trace->experts.push_back(std::move(chosen));
        return y;
    }
    return moe_split(l, x, trace);
}

// The MoE block in its two parts, for the draft pass (fewer experts, none, or a trained helper in their place) and for
// the data a helper is distilled from; with the model's own routing it computes what moe_block does
std::vector<float> Qwen4ExpOracle::moe_split(uint32_t l, const std::vector<float>& x, Trace* trace) {
    const Layer& L = m_.layers()[l];
    const int64_t d = h_.d;

    std::vector<float> routed(d, 0.0f);
    std::vector<int> chosen;
    const Helper* hp = draft_.helper.get();
    if (hp) {
        routed = helper_apply(*hp, l, x);
    } else if (draft_.n_used >= 0) {
        std::vector<float> logits(h_.n_expert);
        matvec(*L.router, x.data(), logits.data());
        tona::nan::report("CPU reference: router logits", tona::nan::clean(logits.data(), logits.size()),
                          std::format("layer {}, position {}", l, pos_));
        const Routing route = route_softmax_topk(logits, draft_.n_used > 0 ? draft_.n_used : h_.n_expert_used);
        for (size_t i = 0; i < route.experts.size(); i++) {
            const auto out = swiglu_expert(*L.gate_exps, *L.up_exps, *L.down_exps, x.data(), route.experts[i]);
            for (int64_t j = 0; j < d; j++) routed[j] += route.weights[i] * out[j];
        }
        chosen = route.experts;
    }

    // Shared expert with its own sigmoid gate; a helper for the whole block replaces it too
    std::vector<float> shared(d, 0.0f);
    if (!(hp && hp->whole)) {
        shared = swiglu_expert(*L.shexp_gate, *L.shexp_up, *L.shexp_down, x.data(), 0);
        const auto wg = dequant_all(*L.shexp_gate_inp);
        double gd = 0;
        for (int64_t i = 0; i < d; i++) gd += (double)wg[i] * x[i];
        const float gate = sigmoid((float)gd);
        for (auto& v : shared) v *= gate;
    }

    std::vector<float> y(d);
    for (int64_t i = 0; i < d; i++) y[i] = routed[i] + shared[i];
    if (trace) {
        trace->experts.push_back(std::move(chosen));
        if (trace->want_moe)
            trace->moe_x.push_back(x), trace->moe_routed.push_back(std::move(routed)),
                trace->moe_shared.push_back(std::move(shared));
    }
    return y;
}

std::vector<float> Qwen4ExpOracle::step(int32_t token, Trace* trace) {
    if (token < 0 || token >= (int32_t)h_.vocab) throw std::runtime_error("ref::step: token outside the vocabulary");
    const int64_t d = h_.d, hc = h_.n_hc;
    tokens_.push_back(token);

    const auto emb = dequant(*m_.tok_embd(), token, d);
    std::vector<float> R(hc * d);
    for (int64_t c = 0; c < hc; c++) std::copy(emb.begin(), emb.end(), R.begin() + c * d);

    std::vector<float> mixed, inject;
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const Layer& L = m_.layers()[l];
        // PLE also runs when its layer is skipped: its history must see every token
        if (l == h_.ple_layer) ple(R);
        if (!draft_.keep.empty() && !draft_.keep[l]) {
            if (trace) trace->residual.push_back(R), trace->experts.emplace_back();
            continue;
        }

        hc_mix(R, L.gr_attn, mixed, &inject);
        const auto y = h_.is_qsa(l) ? qsa(l, mixed) : deltanet(l, mixed);
        hc_combine(R, y, inject);

        hc_mix(R, L.gr_ffn, mixed, &inject);
        const auto f = moe(l, mixed, trace);
        hc_combine(R, f, inject);
        if (trace) trace->residual.push_back(R);
    }

    hc_mix(R, m_.gr_head(), mixed, nullptr);
    std::vector<float> logits(h_.vocab);
    matvec(*m_.output(), mixed.data(), logits.data());
    pos_++;
    return logits;
}

namespace {

// A full copy: a snapshot is restored more than once and from a state that has grown past it, so cutting the caches
// back to a recorded size would lose the data of the positions in between
struct Qwen4Snapshot final : Snapshot {
    int64_t pos = 0;
    std::vector<int32_t> tokens;
    std::vector<std::vector<float>> ple_hist, dn_state, dn_conv, kv_k, kv_v, idx_raw, idx_pool;
};

}  // namespace

void Qwen4ExpOracle::set_draft(const DraftPass& d) {
    if (!d.keep.empty() && d.keep.size() != h_.n_layer)
        throw std::runtime_error("ref::set_draft: keep needs one entry per layer");
    if (d.n_used < -1 || d.n_used > (int)h_.n_expert_used)
        throw std::runtime_error("ref::set_draft: n_used out of range");
    draft_ = d;
}

std::unique_ptr<Snapshot> Qwen4ExpOracle::snapshot() const {
    auto s = std::make_unique<Qwen4Snapshot>();
    s->pos = pos_, s->tokens = tokens_;
    s->ple_hist = ple_hist_, s->dn_state = dn_state_, s->dn_conv = dn_conv_;
    s->kv_k = kv_k_, s->kv_v = kv_v_, s->idx_raw = idx_raw_, s->idx_pool = idx_pool_;
    return s;
}

void Qwen4ExpOracle::restore(const Snapshot& snap) {
    const auto& s = static_cast<const Qwen4Snapshot&>(snap);
    pos_ = s.pos;
    tokens_ = s.tokens;
    ple_hist_ = s.ple_hist, dn_state_ = s.dn_state, dn_conv_ = s.dn_conv;
    kv_k_ = s.kv_k, kv_v_ = s.kv_v, idx_raw_ = s.idx_raw, idx_pool_ = s.idx_pool;
}

}  // namespace tona::ref
