// Model description for qwen35moe (Ornith 1.5 35B-A3B, the Qwen3.5/3.6 MoE family) read from a GGUF: hyperparameters,
// the tensors of each layer with checked shapes, the embedded MTP layer and the weight plan. Layer pattern: gated
// DeltaNet with a full-context gated attention layer at every full_attention_interval-th position, each followed by a
// routed MoE with a gated shared expert; one residual stream, a norm before the mixer and one before the MoE.
// Reference: llama.cpp src/models/qwen35moe.cpp (pinned b11822), docs/architecture/model-architectures.md.
#pragma once

#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "model/binder.h"
#include "model/blocks.h"
#include "model/model_base.h"

namespace tona {

struct Qwen35Hparams {
    uint32_t n_layer = 0;  // trunk layers (40); the file has n_mtp more blocks after them
    uint32_t n_mtp = 0;    // nextn_predict_layers (1)
    uint32_t d = 0;        // 2048
    uint32_t vocab = 0;    // 248,320
    float rms_eps = 0;
    uint32_t full_attn_interval = 0;  // 4: layers 3, 7, .. are attention

    // Gated DeltaNet; in the GGUF V head j uses K head j mod dn_k_heads, as for the Coder
    uint32_t dn_k_heads = 0, dn_v_heads = 0, dn_head_dim = 0, dn_conv = 0;  // 16, 32, 128, 4

    // Gated full-context attention, QK norm, partial RoPE
    uint32_t n_head = 0, n_head_kv = 0, head_dim = 0;  // 16, 2, 256
    uint32_t rope_dims = 0;                            // 64 of 256
    std::array<uint32_t, 4> rope_sections{};           // [11, 11, 10, 0]: text carries one position in all of them
    float rope_base = 0;
    uint32_t native_context = 0;

    // MoE: softmax router over all experts, top n_expert_used, weights renormalized; shared expert with a scalar gate
    uint32_t n_expert = 0, n_expert_used = 0, expert_ff = 0, shexp_ff = 0;

    bool is_attn(uint32_t l) const { return (l + 1) % full_attn_interval == 0; }
    DeltaNetDims deltanet() const { return {d, dn_k_heads, dn_v_heads, dn_head_dim, dn_conv}; }
    AttentionDims attention() const { return {d, n_head, n_head_kv, head_dim, true, true, 0, 0}; }
    MoeDims moe() const { return {d, n_expert, expert_ff, shexp_ff, true, true, 0, false}; }
};

enum class Qwen35LayerKind { DeltaNet, GatedAttn };

struct Qwen35Layer {
    Qwen35LayerKind kind = Qwen35LayerKind::DeltaNet;
    T attn_norm = nullptr;       // before the mixer
    T post_attn_norm = nullptr;  // before the MoE (named post_attention_norm in the file)
    DeltaNetWeights dn;          // kind DeltaNet
    GatedAttentionWeights attn;  // kind GatedAttn
    MoeWeights moe;
};

// The MTP layer inside the model file (blk.<n_layer>): a gated attention layer with MoE, preceded by eh_proj over
// [enorm(embedding) | hnorm(hidden state)]; the LM head and the embedding are the trunk's
struct Qwen35Mtp {
    Qwen35Layer layer;
    T eh_proj = nullptr, enorm = nullptr, hnorm = nullptr, shared_head_norm = nullptr;
};

class Qwen35MoeModel final : public ModelBase {
public:
    static constexpr ArchId ARCH = ArchId::Qwen35Moe;

    // Checks the architecture, the hyperparameters and the shape of every tensor; throws on a mismatch or a tensor
    // the description does not know.
    explicit Qwen35MoeModel(const std::string& path);

    const Qwen35Hparams& hp() const { return hp_; }
    const std::vector<Qwen35Layer>& layers() const { return layers_; }
    const Qwen35Mtp& mtp() const { return mtp_; }
    T tok_embd() const { return tok_embd_; }
    T output_norm() const { return output_norm_; }
    T output() const { return output_; }

    WeightPlan plan() const override;
    MoeShape moe_shape() const override;
    const MoeWeights& moe_weights(uint32_t layer) const override { return layers_[layer].moe; }
    T embedding_table() const override { return tok_embd_; }
    T lm_head() const override { return output_; }
    // Same tags as the Coder's template, but no reasoning-effort lines (tests/chat_vs_jinja.py against the GGUF)
    ChatFamily chat_family() const override { return ChatFamily::Qwen35; }
    MtpSource mtp_source() const override { return hp_.n_mtp ? MtpSource::EmbeddedInModel : MtpSource::None; }
    void fingerprint(Hasher& h) const override;
    std::string describe() const override;

private:
    Qwen35Hparams hp_;
    std::vector<Qwen35Layer> layers_;
    Qwen35Mtp mtp_;
    T tok_embd_ = nullptr, output_norm_ = nullptr, output_ = nullptr;
};

}  // namespace tona
