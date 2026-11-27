#include "cpu/ref_qwen35moe.h"

#include <format>
#include <numeric>
#include <stdexcept>

#include "cpu/ref_blocks.h"
#include "cpu/ref_ops.h"

namespace tona::ref {

Qwen35MoeOracle::Qwen35MoeOracle(const Qwen35MoeModel& m) : m_(m), h_(m.hp()) { reset(); }

void Qwen35MoeOracle::reset() {
    pos_ = 0;
    const DeltaNetDims D = h_.deltanet();
    const size_t channels = 2 * D.k_dim() + D.v_dim();
    dn_state_.assign(h_.n_layer, {});
    dn_conv_.assign(h_.n_layer, {});
    kv_k_.assign(h_.n_layer, {});
    kv_v_.assign(h_.n_layer, {});
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        if (m_.layers()[l].kind != Qwen35LayerKind::DeltaNet) continue;
        dn_state_[l].assign((size_t)h_.dn_v_heads * h_.dn_head_dim * h_.dn_head_dim, 0.0f);
        dn_conv_[l].assign((size_t)(h_.dn_conv - 1) * channels, 0.0f);
    }
}

// rmsnorm(R) * w as a new vector
std::vector<float> Qwen35MoeOracle::normed(const std::vector<float>& R, const TensorInfo& w) const {
    std::vector<float> xn(R);
    rms_norm(xn.data(), (int64_t)xn.size(), h_.rms_eps, dequant_all(w).data());
    return xn;
}

// Full attention over every cached position (no indexer): the new K and V join the cache first
std::vector<float> Qwen35MoeOracle::attention(uint32_t l, const std::vector<float>& x) {
    const GatedAttentionWeights& w = m_.layers()[l].attn;
    const AttentionConfig cfg{h_.attention(), h_.rms_eps, h_.rope_dims, h_.rope_base};
    AttentionProjection p = attention_project(w, cfg, pos_, x.data());
    kv_k_[l].insert(kv_k_[l].end(), p.k.begin(), p.k.end());
    kv_v_[l].insert(kv_v_[l].end(), p.v.begin(), p.v.end());
    std::vector<int64_t> cells(pos_ + 1);
    std::iota(cells.begin(), cells.end(), 0);
    std::vector<float> o = attend_cells({(int64_t)h_.n_head, (int64_t)h_.n_head_kv, (int64_t)h_.head_dim}, p.q.data(),
                                        kv_k_[l], kv_v_[l], cells);
    return attention_finish(*w.o, std::move(o), p.gate);
}

std::vector<float> Qwen35MoeOracle::mixer(uint32_t l, const std::vector<float>& x) {
    const Qwen35Layer& L = m_.layers()[l];
    if (L.kind == Qwen35LayerKind::GatedAttn) return attention(l, x);
    // Qwen3.5 gates the DeltaNet readout with silu(z); the Coder with sigmoid(z)
    return deltanet_block(L.dn, h_.deltanet(), h_.rms_eps, OutGate::Silu, x.data(),
                          {dn_state_[l].data(), dn_conv_[l].data()});
}

std::vector<float> Qwen35MoeOracle::step(int32_t token, Trace* trace) {
    if (token < 0 || token >= (int32_t)h_.vocab) throw std::runtime_error("ref::step: token outside the vocabulary");
    std::vector<float> R = dequant(*m_.tok_embd(), token, h_.d);

    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const Qwen35Layer& L = m_.layers()[l];
        const std::vector<float> y = mixer(l, normed(R, *L.attn_norm));
        for (size_t i = 0; i < R.size(); i++) R[i] += y[i];
        if (trace) trace->residual_mid.push_back(R);

        std::vector<int> chosen;
        const std::vector<float> f =
            moe_block(L.moe, h_.n_expert_used, normed(R, *L.post_attn_norm).data(), trace ? &chosen : nullptr,
                      std::format("layer {}, position {}", l, pos_));
        for (size_t i = 0; i < R.size(); i++) R[i] += f[i];

        if (trace) {
            trace->residual.push_back(R);
            trace->experts.push_back(std::move(chosen));
        }
    }

    std::vector<float> logits(h_.vocab);
    matvec(*m_.output(), normed(R, *m_.output_norm()).data(), logits.data());
    pos_++;
    return logits;
}

}  // namespace tona::ref
