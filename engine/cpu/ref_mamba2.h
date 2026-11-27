// Mamba-2 for the CPU reference (Nemotron-H): the state-space step and the block around it, composed from ref_ops.h.
// One token at a time, which is the SSD recurrence in its sequential form (Dao and Gu, arXiv 2405.21060, section 6);
// the chunked matrix form of the prefill kernels computes the same numbers. Structure and order of operations follow
// llama.cpp b11822 src/models/mamba-base.cpp (build_mamba2_layer) and ggml-cpu/ops.cpp (ssm_scan).
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/ref_ops.h"
#include "model/blocks.h"

namespace tona::ref {

// Inputs of one state-space step. Head h belongs to group h / (heads / groups); B and C are per group.
struct SsmStepInputs {
    const float* x;      // [heads x head_dim] after conv and SiLU
    const float *B, *C;  // [groups x state] after conv and SiLU
    const float* dt;     // [heads]: dt projection plus dt bias, before the softplus
    const float* a;      // [heads]: ssm_a, the (negative) decay rate
    const float* d;      // [heads]: ssm_d, the skip connection
    int64_t heads, head_dim, state, groups;
};
// state [heads][head_dim][state] is updated in place:
//   dt' = softplus(dt), s = s * exp(dt' * a) + B * (x * dt'), y = (s . C) + x * d.   y: [heads x head_dim]
void ssm_step(const SsmStepInputs& in, float* state, float* y);

// Recurrent state of one Mamba-2 layer; the caller owns the storage
struct Mamba2State {
    float* ssm;   // [heads][head_dim][state]
    float* conv;  // [conv - 1][conv_dim], oldest row first
};

// One token through the block: in_proj (times its tensor scale), causal depthwise conv with bias and SiLU, the step,
// the output gate silu(z), the grouped RMS norm, out_proj (times its tensor scale). x is the normed block input
// [d]; returns [d].
std::vector<float> mamba2_block(const Mamba2Weights& w, const Mamba2Dims& D, float eps, const float* x,
                                const Mamba2State& st);

}  // namespace tona::ref
