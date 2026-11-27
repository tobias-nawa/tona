// CPU reference of the Coder (qwen4exp), see ref.h for the interface. Structure follows llama.cpp b11822
// src/models/qwen4exp.cpp and research/06.
#pragma once

#include <cstdint>
#include <vector>

#include "cpu/ref.h"
#include "model/qwen4exp.h"

namespace tona::ref {

// Four gated residual streams, DeltaNet and QSA layers, MoE with shared expert, PLE table. This is the code that was
// the whole of ref.cpp before the oracles were split by architecture; results are unchanged.
class Qwen4ExpOracle final : public Oracle {
public:
    explicit Qwen4ExpOracle(const Qwen4ExpModel& m);

    std::vector<float> step(int32_t token, Trace* trace = nullptr) override;
    int64_t pos() const override { return pos_; }
    void reset() override;
    int64_t max_ctx() const override { return m_.dims().native_context; }
    void set_draft(const DraftPass& d) override;
    std::unique_ptr<Snapshot> snapshot() const override;
    void restore(const Snapshot& s) override;

private:
    void ple(std::vector<float>& R);
    std::vector<float> ple_lookup() const;
    std::vector<float> ple_gated(const std::vector<float>& R, const Layer& L, const std::vector<float>& e) const;
    void hc_mix(const std::vector<float>& R, const GatedResidual& gr, std::vector<float>& mixed,
                std::vector<float>* inject) const;
    void hc_combine(std::vector<float>& R, const std::vector<float>& y, const std::vector<float>& inject) const;
    std::vector<float> deltanet(uint32_t l, const std::vector<float>& x);
    std::vector<float> qsa(uint32_t l, const std::vector<float>& x);
    void index_update(uint32_t l, const std::vector<float>& x);
    std::vector<int64_t> index_cells(uint32_t l, const std::vector<float>& x);
    void qsa_project(uint32_t l, const std::vector<float>& x, std::vector<float>& q, std::vector<float>& gate);
    std::vector<float> moe(uint32_t l, const std::vector<float>& x, Trace* trace);
    std::vector<float> moe_split(uint32_t l, const std::vector<float>& x, Trace* trace);

    const Qwen4ExpModel& m_;
    const Hparams& h_;
    int64_t pos_ = 0;
    DraftPass draft_;  // empty: the full model

    // State per sequence
    std::vector<int32_t> tokens_;                         // for the n-grams
    std::vector<std::vector<float>> ple_hist_;            // normalized PLE inputs, for the dilated conv
    std::vector<std::vector<float>> dn_state_;            // per DeltaNet layer [V heads][v][k]
    std::vector<std::vector<float>> dn_conv_;             // per DeltaNet layer [3][channels], oldest first
    std::vector<std::vector<float>> kv_k_, kv_v_;         // per QSA layer [pos][KV heads x head dim]
    std::vector<std::vector<float>> idx_raw_, idx_pool_;  // per QSA layer [pos][idx_dim], [block][idx_dim]
};

}  // namespace tona::ref
