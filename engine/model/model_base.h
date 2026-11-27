// Architecture-neutral view of a loaded model: what code that serves every architecture may ask (docs/architecture/
// model-architectures.md section 3.1). Cold path only (load, status, cache key); the per-token code works with the
// concrete model, resolved once with model_as<M>(). Nothing here is a superset of the architectures' hyperparameters.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>

#include "model/arch.h"
#include "model/blocks.h"
#include "model/gguf.h"
#include "model/hasher.h"

namespace tona {

// Plain facts every architecture has
struct CommonDims {
    uint32_t vocab = 0;
    uint32_t d_model = 0;
    uint32_t n_layer = 0;
    uint32_t native_context = 0;  // 0 if the GGUF does not say
    float rope_base = 0;
    float rms_eps = 0;
};

// Bytes by use; "per token" means read when decoding one token
struct WeightPlan {
    uint64_t dense = 0;          // everything every token reads (without the embedding row)
    uint64_t dense_hc = 0;       // of which gated residual
    uint64_t dense_lm_head = 0;  // of which LM head
    uint64_t experts = 0;        // all routed experts
    uint64_t expert_bytes = 0;   // one expert of one layer (gate + up + down), mean over the layers
    uint64_t experts_per_token = 0;
    uint64_t embedding = 0;  // token_embd, one row per token
    uint64_t ple_table = 0;  // n-gram table, ple_heads rows per token
    uint64_t ple_row_bytes = 0;
    uint64_t mtp = 0;  // an MTP layer inside the model file (all its tensors); counted in none of the fields above

    uint64_t per_token() const { return dense + experts_per_token; }
};

// What a backend with routed experts needs to know of any model: the sizes it plans caches and buffers from
struct MoeShape {
    uint32_t n_layer = 0;  // trunk layers, each with routed experts
    uint32_t d = 0, vocab = 0;
    uint32_t n_expert = 0, n_expert_used = 0, expert_ff = 0, shexp_ff = 0;
    float rope_base = 0;
    // Residual streams (1: one plain stream) and the host-side input row: the embedding, plus for n-gram models the
    // ple_heads rows of ple_dim of each token (ple_ngram 0: none)
    uint32_t n_hc = 1;
    uint32_t ple_ngram = 0, ple_heads = 0, ple_dim = 0;
    // Layers with an int8 KV cache: every qsa_interval-th one, with an indexer pool of idx_dim per idx_block positions
    // (idx_dim 0: full attention without a pool)
    uint32_t qsa_interval = 0, idx_block = 0, idx_dim = 0;
    bool is_qsa(uint32_t l) const { return qsa_interval && (l + 1) % qsa_interval == 0; }
    // Floats of the input row of one token as the host writes it
    uint32_t in_row() const { return ple_ngram ? 2 * d : d; }
};

// A model whose input row carries n-gram embeddings next to the token embedding (the Coder's PLE): the table, and for
// the token at seq[i] the table rows of its hashed n-grams (MoeShape::ple_heads of them). Pure integer code.
class NgramInput {
public:
    virtual ~NgramInput() = default;
    virtual T table() const = 0;
    virtual void rows_at(const int32_t* seq, int64_t i, int64_t* out) const = 0;
};

class ModelBase {
public:
    virtual ~ModelBase() = default;
    ModelBase(const ModelBase&) = delete;
    ModelBase& operator=(const ModelBase&) = delete;

    ArchId arch() const { return arch_; }
    const Gguf& gguf() const { return gguf_; }
    const CommonDims& dims() const { return dims_; }

    virtual WeightPlan plan() const = 0;
    virtual MoeShape moe_shape() const = 0;
    // The tensors of the routed experts and the shared expert of trunk layer l (< moe_shape().n_layer)
    virtual const MoeWeights& moe_weights(uint32_t layer) const = 0;
    // Experts per token of trunk layer l; the same in every layer unless the model says otherwise (Bolt keeps an array)
    virtual uint32_t expert_used(uint32_t) const { return moe_shape().n_expert_used; }
    // The router of MoE layer l + 1 on the input of layer l predicts its experts well enough to prefetch them (the next
    // layer follows right behind; not so when other blocks lie between two MoE layers)
    virtual bool predicts_next_router() const { return true; }
    // The trunk as the prefill walks it: blocks (every block has a mixer or a MoE; for a model whose every layer has
    // routed experts, the same as the layers), the block of the i-th MoE (the expert caches and moe_weights are
    // indexed by that ordinal), the ordinal of a block (-1: no MoE) and whether a block holds an attention KV cache
    virtual uint32_t n_blocks() const { return moe_shape().n_layer; }
    virtual uint32_t moe_block(uint32_t ordinal) const { return ordinal; }
    virtual int moe_ordinal(uint32_t block) const { return (int)block; }
    virtual bool has_kv(uint32_t block) const { return moe_shape().is_qsa(block); }
    // Token embedding table and LM head; the host reads rows of the first, the backend uploads the second
    // Null for a model without n-gram embeddings (MoeShape::ple_ngram == 0)
    virtual const NgramInput* ngram_input() const { return nullptr; }
    virtual T embedding_table() const = 0;
    virtual T lm_head() const = 0;
    virtual ChatFamily chat_family() const = 0;
    virtual MtpSource mtp_source() const = 0;
    // The drafter files this model accepts; by default none (it runs one token per pass)
    virtual DrafterSupport drafters() const { return {}; }
    // Adds what besides the tensor directory changes the tokens or the states of this model (prompt-cache key)
    virtual void fingerprint(Hasher& h) const = 0;
    // One line for /status
    virtual std::string describe() const = 0;

protected:
    ModelBase(ArchId arch, const std::string& path) : arch_(arch), gguf_(path) {}
    // Called by the module's constructor once it has read its hyperparameters
    void set_dims(const CommonDims& d) { dims_ = d; }

private:
    ArchId arch_;
    Gguf gguf_;
    CommonDims dims_;
};

// The concrete model behind a base: one check of arch() instead of RTTI. Each module declares `static constexpr ArchId
// ARCH`; a mismatch throws, so a caller that assumed the Coder fails with a message instead of reading wrong data.
template <class M>
const M& model_as(const ModelBase& m) {
    if (m.arch() != M::ARCH)
        throw std::runtime_error("this needs a " + std::string(arch_name(M::ARCH)) + " model, not " +
                                 std::string(arch_name(m.arch())));
    return static_cast<const M&>(m);
}

}  // namespace tona
