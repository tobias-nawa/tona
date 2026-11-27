// Layer blocks of the CPU reference that more than one architecture uses, composed from ref_ops.h: the Coder's oracle
// and Ornith's call the same code, so a reading of the maths is made (and checked) once. Weights come as the structs
// of model/blocks.h. Same arithmetic order as the Coder's oracle had inline: results are bit-identical.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "cpu/ref_ops.h"
#include "model/blocks.h"

namespace tona::ref {

// Recurrent state of one DeltaNet layer; the caller owns the storage
struct DeltaNetState {
    float* state;       // [v_heads][head_dim (value)][head_dim (key)]
    float* conv_state;  // [conv - 1][channels], oldest row first
};

// Gated DeltaNet layer for one token: projections, causal depthwise conv with SiLU, L2 norm of q and k, recurrence,
// output projection. Returns [d].
std::vector<float> deltanet_block(const DeltaNetWeights& w, const DeltaNetDims& D, float eps, OutGate out_gate,
                                  const float* x, const DeltaNetState& st);

struct AttentionConfig {
    AttentionDims dims;
    float eps;
    int64_t rope_dims;
    float rope_base;
};

// Q (QK-normed, rotated), its gate, and K (normed, rotated) and V of the new position
struct AttentionProjection {
    std::vector<float> q, gate;  // [n_head x head_dim]
    std::vector<float> k, v;     // [n_head_kv x head_dim]
};
// attn_q holds per head [q | gate]; needs the gate and QK-norm flags of dims
AttentionProjection attention_project(const GatedAttentionWeights& w, const AttentionConfig& c, int64_t pos,
                                      const float* x);
// o (attention readout [n_head x head_dim]) times sigmoid(gate), then the output projection. Returns [d].
std::vector<float> attention_finish(const TensorInfo& wo, std::vector<float> o, const std::vector<float>& gate);

// Routed experts (softmax router, top n_used, renormalized weights) plus the shared expert with its scalar sigmoid
// gate. chosen (optional) receives the picked experts; where names the layer and position for NaN reports.
std::vector<float> moe_block(const MoeWeights& w, int64_t n_used, const float* x, std::vector<int>* chosen,
                             const std::string& where);

}  // namespace tona::ref
