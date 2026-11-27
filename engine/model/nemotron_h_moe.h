// Model description for nemotron_h_moe (Nemotron 3.5 Bolt 74B-A9B, the Nemotron-H family with a latent MoE) read from a
// GGUF: hyperparameters, the kind and the tensors of each block with checked shapes, and the weight plan. A block is
// one of three kinds, each with its own attn_norm in front: Mamba-2, attention (no gate, no QK norm, no RoPE) or a
// latent MoE (squared-ReLU experts without a gate tensor in a 1024-wide latent space, sigmoid router with a selection
// bias, a shared expert on the full width). One residual stream: x' = x + block(rmsnorm(x)). The blocks are uneven:
// the experts of a layer differ in count (240 to 496), width and top-k, so everything per layer lives in the layer.
// Reference: llama.cpp src/models/nemotron-h.cpp (pinned b11822), docs/architecture/model-architectures.md.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "model/binder.h"
#include "model/blocks.h"
#include "model/model_base.h"

namespace tona {

struct NemotronHparams {
    uint32_t n_layer = 0;  // blocks (88)
    uint32_t d = 0;        // 4096
    uint32_t vocab = 0;    // 131,072
    float rms_eps = 0;
    uint32_t native_context = 0;

    // Attention: n_head query heads of head_dim; the KV heads differ per layer (0: not an attention block)
    uint32_t n_head = 0, head_dim = 0;  // 32, 128

    // Mamba-2
    uint32_t ssm_inner = 0, ssm_heads = 0, ssm_state = 0, ssm_groups = 0, ssm_conv = 0;  // 8192, 128, 128, 8, 4
    uint32_t ssm_head_dim() const { return ssm_inner / ssm_heads; }                      // 64

    // Latent MoE: sigmoid router, top-k by score + bias, weights = scores normalized to sum 1, times expert_scale
    uint32_t latent = 0;        // 1024; 0: the experts work on d
    uint32_t shexp_ff = 0;      // 5376
    uint32_t n_expert_max = 0;  // 496: the largest expert count of any layer
    float expert_scale = 1.0f;  // 5.0
    // Per layer; 0 where the layer has no such part
    std::vector<uint32_t> n_head_kv, ff, expert_ff, expert_used;

    Mamba2Dims mamba2() const { return {d, ssm_inner, ssm_heads, ssm_state, ssm_groups, ssm_conv}; }
};

enum class BoltBlockKind { Mamba2, Attention, Moe };

struct NemotronLayer {
    BoltBlockKind kind = BoltBlockKind::Mamba2;
    T attn_norm = nullptr;                             // before the block, whatever its kind
    Mamba2Weights ssm;                                 // kind Mamba2
    GatedAttentionWeights attn;                        // kind Attention (the gate and QK norm members stay null)
    MoeWeights moe;                                    // kind Moe
    uint32_t n_kv = 0;                                 // Attention: KV heads
    uint32_t n_expert = 0, n_used = 0, expert_ff = 0;  // Moe
};

class NemotronHMoeModel final : public ModelBase {
public:
    static constexpr ArchId ARCH = ArchId::NemotronHMoe;

    // Checks the architecture, the hyperparameters and the shape of every tensor; throws on a mismatch or a tensor the
    // description does not know.
    explicit NemotronHMoeModel(const std::string& path);

    const NemotronHparams& hp() const { return hp_; }
    const std::vector<NemotronLayer>& layers() const { return layers_; }
    uint32_t count(BoltBlockKind k) const;
    // Layer index of the i-th MoE block (the index moe_weights and the MoeShape work with)
    uint32_t moe_layer(uint32_t i) const { return moe_index_[i]; }
    uint32_t n_blocks() const override { return hp_.n_layer; }
    uint32_t moe_block(uint32_t i) const override { return moe_index_[i]; }
    int moe_ordinal(uint32_t block) const override {
        return layers_[block].kind == BoltBlockKind::Moe ? ordinal_[block] : -1;
    }
    bool has_kv(uint32_t block) const override { return layers_[block].kind == BoltBlockKind::Attention; }
    T tok_embd() const { return tok_embd_; }
    T output_norm() const { return output_norm_; }
    T output() const { return output_; }

    WeightPlan plan() const override;
    // The MoE blocks only, in order: n_layer is their number (40), n_expert and expert_ff the largest of any layer. The
    // per-layer counts are NemotronLayer::n_expert / expert_ff (the per-layer geometry of ticket R6/B5 reads them).
    // Maxima over the MoE blocks (expert count, width, top-k) for sizing buffers; the geometry of a block is in
    // moe_weights(layer) and layers()[block] (B5/B6 read it per layer, never from this shape)
    MoeShape moe_shape() const override;
    const MoeWeights& moe_weights(uint32_t i) const override { return layers_[moe_index_[i]].moe; }
    uint32_t expert_used(uint32_t i) const override { return layers_[moe_index_[i]].n_used; }
    bool predicts_next_router() const override { return false; }  // Mamba-2 and attention blocks lie between MoE blocks
    T embedding_table() const override { return tok_embd_; }
    T lm_head() const override { return output_; }
    ChatFamily chat_family() const override { return ChatFamily::Nemotron; }
    MtpSource mtp_source() const override { return MtpSource::None; }
    void fingerprint(Hasher& h) const override;
    std::string describe() const override;

private:
    NemotronHparams hp_;
    std::vector<NemotronLayer> layers_;
    std::vector<uint32_t> moe_index_;
    std::vector<int> ordinal_;  // per block: its place among the MoE blocks
    T tok_embd_ = nullptr, output_norm_ = nullptr, output_ = nullptr;
};

}  // namespace tona
