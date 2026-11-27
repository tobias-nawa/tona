// Ornith's prefill sequence (arch/qwen35moe/prefill_layer.cpp), chosen by prefill_sequence() in prefill_run.cpp, and
// the phase buffers only it uses.
#pragma once

#include <any>
#include <vector>

#include "gpu/prefill_run.h"

namespace tona::gpu {

// Buffers of the full-attention phase: attn_q output ([q, gate] per head), q and gate after the preparation, K and V
// of the chunk, the attention output as f16
struct FattnBufs {
    float *qfull, *qn, *g, *tk, *tv;
    _Float16* o16;
};
FattnBufs carve_fattn(Carve& p, const PfDims& D, int C);

// Buffers of the MTP entries of the prompt (one part of S tokens)
struct Qwen35MtpBufs {
    float *e, *Rm, *tk, *tv;
    _Float16* x16;
};

// The MTP layer's weights for the MTP entries of the prompt: eh_proj, K and V, the norms and the RoPE base
struct Qwen35PfMtp {
    PfW eh, k, v;
    const float *enorm = nullptr, *hnorm = nullptr, *attn_norm = nullptr, *k_norm = nullptr;
    float rope_base = 0;
};

// Ornith's prefill records besides the shared descriptors: the gains of the norms before the mixer and before the MoE
// of every layer and of the output norm (one plain residual stream). Recorded while the program is built.
struct Qwen35PfRes {
    std::vector<const float*> norm_attn, norm_ffn;
    const float* head_norm = nullptr;
    Qwen35PfMtp mtp;
    static Qwen35PfRes& reset(std::any& store, size_t n_layer);
    // Const view of the record for code that only reads it; the MTP build is the one writer after reset
    static Qwen35PfRes& mut(std::any& store) { return std::any_cast<Qwen35PfRes&>(store); }
    static const Qwen35PfRes& of(const std::any& store);
};

// Tag of PrefillRun::Step<Qwen35MoePrefillTag>
struct Qwen35MoePrefillTag {};

const PrefillSequence& qwen35moe_prefill_sequence();

}  // namespace tona::gpu
