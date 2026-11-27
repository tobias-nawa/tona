// Model description for qwen4exp (Qwen3.8-Flash-Next) read from a GGUF: hyperparameters, the tensors of each layer
// with checked shapes, and the weight plan (what is read per token, what lives where).
// Structure and formulas: research/06-qwen38-architecture.md.
#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <string>
#include <vector>

#include "model/binder.h"
#include "model/gguf.h"
#include "model/model_base.h"

namespace tona {

struct Hparams {
    uint32_t n_layer = 0;  // 48
    uint32_t d = 0;        // 2560
    uint32_t vocab = 0;    // 248,320
    float rms_eps = 0;
    uint32_t n_hc = 0;     // residual streams (4)
    uint32_t hc_rank = 0;  // 320
    uint32_t full_attn_interval = 0;

    // Gated DeltaNet
    uint32_t dn_k_heads = 0;   // 16
    uint32_t dn_v_heads = 0;   // 48; in the GGUF, V head j uses K head j mod dn_k_heads
    uint32_t dn_head_dim = 0;  // 128
    uint32_t dn_conv = 0;      // 4

    // QSA
    uint32_t n_head = 0, n_head_kv = 0, head_dim = 0;  // 24, 2, 256
    uint32_t rope_dims = 0;                            // 64
    float rope_base = 0;
    uint32_t idx_heads = 0, idx_dim = 0, idx_top_k = 0, idx_block = 0;  // 4, 128, 2048 tokens, 4
    uint32_t idx_top_k_gguf = 0;  // idx_top_k as the GGUF has it; idx_top_k itself follows TONA_IDX_TOPK

    // MoE
    uint32_t n_expert = 0, n_expert_used = 0, expert_ff = 0, shexp_ff = 0;

    // n-gram embedding (PLE)
    uint32_t ple_layer = 0, ple_dim = 0, ple_heads = 0, ple_ngram = 0, ple_conv = 0;
    uint32_t ple_eos = 0, ple_image_token = 0;
    std::array<int64_t, 3> ple_mult{};
    std::vector<uint64_t> ple_head_vocab, ple_head_offset;

    bool is_qsa(uint32_t l) const { return (l + 1) % full_attn_interval == 0; }
    uint32_t dn_k_dim() const { return dn_k_heads * dn_head_dim; }  // 2048
    uint32_t dn_v_dim() const { return dn_v_heads * dn_head_dim; }  // 6144
};

// Gated residual module: read (norm, down, up) and write gate (inject; absent at the head)
struct GatedResidual {
    T norm = nullptr, down = nullptr, up = nullptr, inject = nullptr;
};

struct Layer {
    GatedResidual gr_attn, gr_ffn;

    // DeltaNet (not QSA)
    T qkv = nullptr, gate = nullptr, beta = nullptr, alpha = nullptr, conv1d = nullptr;
    T dt_bias = nullptr, a = nullptr, ssm_norm = nullptr, ssm_out = nullptr;

    // QSA
    T q = nullptr, k = nullptr, v = nullptr, o = nullptr, q_norm = nullptr, k_norm = nullptr;
    T idx_q = nullptr, idx_k = nullptr, idx_q_norm = nullptr, idx_k_norm = nullptr;

    // MoE
    T router = nullptr, gate_exps = nullptr, up_exps = nullptr, down_exps = nullptr;
    T shexp_gate_inp = nullptr, shexp_gate = nullptr, shexp_up = nullptr, shexp_down = nullptr;

    // PLE (only layer hp.ple_layer)
    T ple_key = nullptr, ple_value = nullptr, ple_conv1d = nullptr;
    T ple_norm_key = nullptr, ple_norm_query = nullptr, ple_norm_conv = nullptr;
};

// MTP head from its own GGUF (ggml-org mtp-...-Q8_0.gguf): one QSA block with its own MoE (512 experts, top 10),
// preceded by eh_proj over [norm(embedding) | norm(h per stream)] and followed by its own head mixer; LM head and
// embedding as in the target model (research/06-qwen38-architecture.md section 8)
struct MtpHead {
    Gguf gguf;
    Layer L;              // QSA and MoE like a layer of the target
    T eh_proj = nullptr;  // [2d → d]: x = [enorm(e) | hnorm(h_c)] per stream c
    T enorm = nullptr, hnorm = nullptr;
    GatedResidual gr_head;  // hc_head_norm/down/up, without inject
    uint32_t n_expert = 0, n_used = 0;
    float rope_base = 0;

    MtpHead(const std::string& path, const Hparams& h);
};

// Indexer cells of a QSA layer at position pos (after the raw key of pos is in): n_top blocks by score, then the
// block in progress
struct QsaCells {
    int64_t blk = 0, n_pool = 0, top_blocks = 0, tail0 = 0, n_tail = 0;
    int64_t count() const { return top_blocks * blk + n_tail; }
};

inline QsaCells qsa_cells(int64_t pos, int64_t blk, int64_t top_k) {
    QsaCells c;
    c.blk = blk;
    c.n_pool = (pos + 1) / blk;
    c.top_blocks = std::min<int64_t>(c.n_pool, top_k / blk);
    c.tail0 = c.n_pool * blk;
    c.n_tail = pos + 1 - c.tail0;
    return c;
}

class Qwen4ExpModel final : public ModelBase, private NgramInput {
public:
    static constexpr ArchId ARCH = ArchId::Qwen4Exp;

    // Checks the architecture, the hyperparameters and the shape of every expected tensor; throws on a mismatch.
    explicit Qwen4ExpModel(const std::string& path);

    const Hparams& hp() const { return hp_; }
    const std::vector<Layer>& layers() const { return layers_; }
    T tok_embd() const { return tok_embd_; }
    T output() const { return output_; }
    const GatedResidual& gr_head() const { return gr_head_; }
    T ple_embd() const { return ple_embd_; }

    WeightPlan plan() const override;
    MoeShape moe_shape() const override;
    const MoeWeights& moe_weights(uint32_t layer) const override { return moe_[layer]; }
    const NgramInput* ngram_input() const override { return this; }
    T embedding_table() const override { return tok_embd_; }
    T lm_head() const override { return output_; }
    ChatFamily chat_family() const override { return ChatFamily::Qwen3Coder; }
    MtpSource mtp_source() const override { return MtpSource::SeparateGguf; }
    DrafterSupport drafters() const override { return {true, true, true}; }
    void fingerprint(Hasher& h) const override;
    std::string describe() const override;

private:
    T table() const override { return ple_embd_; }
    void rows_at(const int32_t* seq, int64_t i, int64_t* out) const override;

    Hparams hp_;
    std::vector<Layer> layers_;
    std::vector<MoeWeights> moe_;  // the MoE tensors of each layer in the shared block's form
    T tok_embd_ = nullptr, output_ = nullptr, ple_embd_ = nullptr;
    GatedResidual gr_head_;
};

}  // namespace tona
