// Weight structs and shape-checked binders of the blocks that more than one architecture uses (docs/architecture/
// model-architectures.md section 3.1). A binder reads the tensors of one block under a name prefix ("blk.3.") and
// checks every shape; an architecture module calls them and keeps the pointers where its own layer struct wants them.
#pragma once

#include <cstdint>
#include <string>

#include "model/binder.h"

namespace tona {

// Gated DeltaNet (Coder, Ornith)
struct DeltaNetDims {
    uint64_t d, k_heads, v_heads, head_dim, conv;
    uint64_t k_dim() const { return k_heads * head_dim; }
    uint64_t v_dim() const { return v_heads * head_dim; }
};
struct DeltaNetWeights {
    T qkv = nullptr, gate = nullptr, beta = nullptr, alpha = nullptr, conv1d = nullptr;
    T dt_bias = nullptr, a = nullptr, ssm_norm = nullptr, ssm_out = nullptr;
};
DeltaNetWeights bind_deltanet_weights(Binder& b, const std::string& prefix, const DeltaNetDims& D);

// Attention over several KV heads with an optional output gate, QK norm and block indexer (QSA)
struct AttentionDims {
    uint64_t d, n_head, n_head_kv, head_dim;
    bool gate = false;                    // attn_q holds [q | gate] per head
    bool qk_norm = false;                 // attn_q_norm and attn_k_norm
    uint64_t idx_heads = 0, idx_dim = 0;  // idx_heads 0: no indexer
};
struct GatedAttentionWeights {
    T q = nullptr, k = nullptr, v = nullptr, o = nullptr, q_norm = nullptr, k_norm = nullptr;
    T idx_q = nullptr, idx_k = nullptr, idx_q_norm = nullptr, idx_k_norm = nullptr;
};
GatedAttentionWeights bind_attention_weights(Binder& b, const std::string& prefix, const AttentionDims& D);

// Routed experts with an optional shared expert, gate tensor, latent projection pair and shared-expert gate
struct MoeDims {
    uint64_t d, n_expert, ff, shared_ff;  // shared_ff 0: no shared expert
    bool gated = true;                    // experts have gate and up (SwiGLU); false: up and down only
    bool shared_gate = false;             // ffn_gate_inp_shexp: a scalar gate per token on the shared expert
    uint64_t latent = 0;                  // 0: experts work on d; else the width of the latent space
    bool router_bias = false;             // exp_probs_b.bias
};
struct MoeWeights {
    T router = nullptr, router_bias = nullptr;
    T gate_exps = nullptr, up_exps = nullptr, down_exps = nullptr;
    T shexp_gate_inp = nullptr, shexp_gate = nullptr, shexp_up = nullptr, shexp_down = nullptr;
    T latent_down = nullptr, latent_up = nullptr;
};
MoeWeights bind_moe_weights(Binder& b, const std::string& prefix, const MoeDims& D);

// Mamba-2 block (Nemotron-H); the per-tensor F32 scales exist only for NVFP4 projections
struct Mamba2Dims {
    uint64_t d, inner, heads, state, groups, conv;
    uint64_t conv_dim() const { return inner + 2 * groups * state; }
    uint64_t in_proj_dim() const { return 2 * inner + 2 * groups * state + heads; }
};
struct Mamba2Weights {
    T in_proj = nullptr, in_scale = nullptr, out_proj = nullptr, out_scale = nullptr;
    T conv1d = nullptr, conv1d_bias = nullptr, dt_bias = nullptr, a = nullptr, dskip = nullptr, norm = nullptr;
};
Mamba2Weights bind_mamba2_weights(Binder& b, const std::string& prefix, const Mamba2Dims& D);

}  // namespace tona
