// CPU reference of Nemotron 3.5 Bolt (nemotron_h_moe), see ref.h for the interface. Structure follows llama.cpp b11822
// src/models/nemotron-h.cpp: one residual stream, an RMSNorm in front of every block, then a Mamba-2, an attention or
// a latent-MoE block per layer, final norm and LM head. Trace: `residual` has one entry per block, `experts` one per
// MoE block only (88 and 40 for the real file).
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/ref.h"
#include "model/nemotron_h_moe.h"

namespace tona::ref {

class NemotronHOracle final : public Oracle {
public:
    explicit NemotronHOracle(const NemotronHMoeModel& m);

    std::vector<float> step(int32_t token, Trace* trace = nullptr) override;
    int64_t pos() const override { return pos_; }
    void reset() override;
    int64_t max_ctx() const override { return m_.dims().native_context; }

private:
    std::vector<float> attention(uint32_t l, const std::vector<float>& x);
    std::vector<float> moe(uint32_t l, const std::vector<float>& x, std::vector<int>* chosen);

    const NemotronHMoeModel& m_;
    const NemotronHparams& h_;
    int64_t pos_ = 0;

    // State per sequence
    std::vector<std::vector<float>> ssm_state_;    // per Mamba-2 layer [heads][head_dim][state]
    std::vector<std::vector<float>> conv_state_;   // per Mamba-2 layer [conv - 1][conv_dim], oldest first
    std::vector<std::vector<float>> kv_k_, kv_v_;  // per attention layer [pos][KV heads x head_dim]
};

}  // namespace tona::ref
