// CPU reference of Ornith (qwen35moe), see ref.h for the interface. Structure follows llama.cpp b11822
// src/models/qwen35moe.cpp: one residual stream, RMSNorm before the mixer and before the MoE, gated DeltaNet and
// gated full-context attention, softmax-routed MoE with a gated shared expert, final norm and LM head.
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/ref.h"
#include "model/qwen35moe.h"

namespace tona::ref {

class Qwen35MoeOracle final : public Oracle {
public:
    explicit Qwen35MoeOracle(const Qwen35MoeModel& m);

    std::vector<float> step(int32_t token, Trace* trace = nullptr) override;
    int64_t pos() const override { return pos_; }
    void reset() override;
    int64_t max_ctx() const override { return m_.dims().native_context; }

private:
    std::vector<float> mixer(uint32_t l, const std::vector<float>& x);
    std::vector<float> attention(uint32_t l, const std::vector<float>& x);
    std::vector<float> normed(const std::vector<float>& R, const TensorInfo& w) const;

    const Qwen35MoeModel& m_;
    const Qwen35Hparams& h_;
    int64_t pos_ = 0;

    // State per sequence
    std::vector<std::vector<float>> dn_state_;     // per DeltaNet layer [V heads][v][k]
    std::vector<std::vector<float>> dn_conv_;      // per DeltaNet layer [conv - 1][channels], oldest first
    std::vector<std::vector<float>> kv_k_, kv_v_;  // per attention layer [pos][KV heads x head dim]
};

}  // namespace tona::ref
