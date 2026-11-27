// CPU reference of the forward pass: one token at a time, weights dequantized row by row, math in F32 (dot products in
// double). Slow and simple; the oracle for the GPU kernels. One Oracle per architecture (ref_<arch>.cpp), each
// composing the operations of ref_ops.h: the operations are shared, the layer sequence is the architecture's own.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#include "model/model_base.h"

namespace tona::ref {

// Intermediate values of a step, for layer-by-layer comparisons
struct Trace {
    std::vector<std::vector<float>> residual;      // after each layer, the whole residual (Coder: [n_hc x d])
    std::vector<std::vector<float>> residual_mid;  // Ornith only: after the mixer of each layer, before its MoE block
    std::vector<std::vector<int>> experts;         // chosen experts per layer
    // Per layer, the MoE block's input and its two parts (routed experts, shared expert with its gate); filled only if
    // want_moe is set: the data a draft-side helper is distilled from
    bool want_moe = false;
    std::vector<std::vector<float>> moe_x, moe_routed, moe_shared;
};

// Cheaper pass of the same model for self-speculative drafting: only the layers with keep[l] set run (the others
// leave the residual untouched and keep their state), and each kept layer routes to n_used experts (0 = the model's
// own number; -1: no routed experts, the shared expert only). An empty keep runs every layer.
// A trained dense FFN per layer that stands in for the routed experts (whole: for the whole MoE block, shared expert
// included) in the draft pass; SwiGLU, weights [h][d], [h][d], [d][h] in F32 (the format of cpu/ref_helper.h)
struct Helper {
    int64_t h = 0;
    bool whole = false;
    std::vector<std::vector<float>> gate, up, down;  // per layer
};

struct DraftPass {
    std::vector<uint8_t> keep;
    int n_used = 0;
    std::shared_ptr<const Helper> helper;  // set: the routed experts are replaced by it (n_used is ignored)
};

// The state of a sequence at one position, to rewind to after a draft
struct Snapshot {
    virtual ~Snapshot() = default;
};

class Oracle {
public:
    virtual ~Oracle() = default;

    // One token at the next position; returns the logits [vocab]
    virtual std::vector<float> step(int32_t token, Trace* trace = nullptr) = 0;
    virtual int64_t pos() const = 0;
    virtual void reset() = 0;
    // Positions the oracle can hold (the model's native context; 0 if the GGUF does not say)
    virtual int64_t max_ctx() const = 0;

    // Draft pass for the steps that follow (an empty DraftPass restores the full model); oracles without a layer
    // structure to skip refuse
    virtual void set_draft(const DraftPass&) { throw std::runtime_error("this CPU reference has no draft pass"); }
    virtual std::unique_ptr<Snapshot> snapshot() const {
        throw std::runtime_error("this CPU reference has no snapshot");
    }
    virtual void restore(const Snapshot&) { throw std::runtime_error("this CPU reference has no snapshot"); }
};

// The oracle of the model's architecture; throws if there is none yet. The model must outlive it.
std::unique_ptr<Oracle> make_oracle(const ModelBase& m);

}  // namespace tona::ref
