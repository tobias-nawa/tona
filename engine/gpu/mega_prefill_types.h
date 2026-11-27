// Weights and state the prefill (gpu/prefill.hip) uses, recorded while the decode program is built: the same copies
// in VRAM serve the matvecs and the GEMMs.
#pragma once

#include <cstdint>

#include "gpu/mega_cache.h"

namespace tona::gpu {

struct PfW {
    uint32_t type = 0;
    int64_t in = 0, rows = 0;
    const void* w = nullptr;
};

// Gated DeltaNet (Coder, Ornith): projections, causal conv, delta rule state and the pending tokens of the last decode
// pass
struct PfDeltaNet {
    PfW qkv, gate, beta, alpha, ssm_out;
    const float *conv_w = nullptr, *dt_bias = nullptr, *A = nullptr, *ssm_norm = nullptr;
    float *state = nullptr, *ring = nullptr, *pend_d = nullptr, *pend_a = nullptr, *pend_k = nullptr;
};

// Mamba-2 (Bolt): projections, the tensor scales of NVFP4 (1 for other types), conv, scan parameters, the state and the
// inputs of the last decode pass (the replay: decay per head, x * dt per row, B per group, two halves by parity)
struct PfMamba2 {
    PfW in_proj, out_proj;
    float in_scale = 1, out_scale = 1;
    const float *conv_w = nullptr, *conv_b = nullptr, *dt_bias = nullptr, *A = nullptr, *D = nullptr, *norm = nullptr;
    float *ssm = nullptr, *ring = nullptr, *pdec = nullptr, *pxdt = nullptr, *pb = nullptr;
};

// The indexer of a quantized self-attention layer (the Coder); absent in full attention
struct PfIndexer {
    PfW idx_q, idx_k;
    const float *idx_q_norm = nullptr, *idx_k_norm = nullptr;
    float *raw_ring = nullptr, *pool = nullptr;
};

// Gated attention over an int8 KV cache (Coder: with the indexer, Ornith: full context)
struct PfGatedAttn {
    PfW q, k, v, o;
    const float *q_norm = nullptr, *k_norm = nullptr;
    uint8_t* kv = nullptr;
    uint8_t* kv_cold = nullptr;  // split by position (kv_place.h): the cells from KvSplit::hot on, else null
    PfIndexer idx;
};

// Routed MoE with a gated shared expert
struct PfMoe {
    PfW router, sh_gate, sh_up, sh_down, sh_inp;
    // Latent MoE (Bolt): the experts are ungated, work on lat_down(x) and their sum goes through lat_up; the router
    // chooses by sigmoid with a selection bias (device copy) and the scale. lat_down.w null: not a latent layer.
    PfW lat_down, lat_up;
    const float* router_bias = nullptr;
    float route_scale = 1.0f;
};

// One layer: its mixer (DeltaNet, Mamba-2 or gated attention; the others stay empty) and the MoE. What the residual
// needs (gated residuals, norm gains) is the architecture's own record (MegaPrefillData::pf_arch_).
struct PfLayer {
    PfDeltaNet dn;
    PfMamba2 mamba;
    PfGatedAttn attn;
    PfMoe moe;
};

// Slot ranges of whole layers that the prefill borrows, with their saved cache
struct PfLoan {
    uint32_t layer;
    LayerCache saved;
};

}  // namespace tona::gpu
