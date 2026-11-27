// CPU reference of the DFlash2 drafter (dflash-Qwen3.8-Flash-Next, ggml-org): oracle for the GPU and for measuring
// the acceptance rate. Structure follows llama.cpp d9ee007 src/models/dflash.cpp and common/speculative.cpp.
//
// Context: for each accepted token the inputs of the target layers (mean over the hyper-connection streams)
// -> fc -> RMSNorm; from that, K and V of every drafter layer in its own cache (inject). Draft: one block
// [last token, mask x (n - 1)] at pos ... pos + n - 1 through 5 layers with non-causal attention over cache and
// block, the target model's head, then the selector: top 16 per position and transition scores, walked greedily
// (draft).
#pragma once

#include <cstdint>
#include <vector>

#include "model/gguf.h"
#include "model/qwen4exp.h"

namespace tona::ref {

class DFlash {
public:
    DFlash(const Gguf& draft, const Qwen4ExpModel& target);

    const std::vector<int64_t>& target_layers() const { return target_layers_; }
    // feat: inputs of the target layers target_layers() for the token at position pos, [n_layers * d]
    void inject(const float* feat, int64_t pos);
    // n - 1 draft tokens after last (position pos); the context must be fed up to pos - 1
    std::vector<int32_t> draft(int32_t last, int64_t pos, int n);
    void reset();

    // Building blocks, also for the drafter on the GPU: embedding (target model), selector over the logits
    // [n - 1][vocab] and gates [n - 1][rank] of the mask positions
    int32_t mask_token() const { return mask_token_; }
    void embed(int32_t token, float* out) const;
    // probs (optional): softmax of the scores at the chosen candidate per position, estimate of acceptance
    std::vector<int32_t> select(int32_t last, const float* logits, const float* gate, int n,
                                std::vector<float>* probs = nullptr) const;

private:
    struct Layer {
        const TensorInfo *attn_norm, *q, *k, *v, *o, *q_norm, *k_norm, *ffn_norm, *gate, *up, *down;
        const TensorInfo *attn_conv_base, *attn_conv_proj, *ffn_conv_base, *ffn_conv_proj;
        std::vector<float> attn_norm_w, q_norm_w, k_norm_w, ffn_norm_w, attn_base, ffn_base;
        std::vector<std::vector<float>> K, V;  // by position, [n_kv * hd]
    };

    void kv_rope_norm(Layer& L, float* k, int64_t pos) const;
    struct AttnIn {
        const Layer& L;
        const std::vector<float>&q, &k, &v;
        int64_t pos;
        int n;
    };
    void gather_scores(const AttnIn& in, int t, int64_t hh, std::vector<double>& s,
                       std::vector<const float*>& vs) const;
    std::vector<float> attend(const Layer& L, const std::vector<float>& q, const std::vector<float>& k,
                              const std::vector<float>& v, int64_t pos, int n) const;
    void layer_attention(Layer& L, std::vector<float>& x, int64_t pos, int n) const;
    void layer_ffn(const Layer& L, std::vector<float>& x, int n) const;
    // Convolution over the block with kernel 2 (side 0: input, 1: output), weight = dynamic + base
    void conv(std::vector<float>& x, const std::vector<float>& dyn, const std::vector<float>& base, int side,
              int n) const;

    const Qwen4ExpModel& target_;
    int64_t d_, n_head_, n_kv_, hd_, ff_, n_swa_, block_kernel_, group_, rank_, top_k_;
    int32_t mask_token_;
    float eps_, rope_base_;
    std::vector<int64_t> target_layers_;
    std::vector<Layer> layers_;
    const TensorInfo *fc_, *sel_hidden_, *sel_prev_, *sel_next_;
    std::vector<float> enc_norm_, out_norm_;
};

}  // namespace tona::ref
