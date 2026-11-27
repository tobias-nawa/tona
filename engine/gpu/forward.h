// Forward pass on the GPU, version 1: one kernel per step, activations in F32, decode token by token.
// Dense weights fixed in VRAM, experts in an LRU cache in VRAM (misses reloaded from the mapped file), embedding and
// n-gram rows read on the CPU. Same computation as cpu/ref.cpp.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <list>
#include <unordered_map>
#include <vector>

#include "gpu/matvec.h"
#include "model/qwen4exp.h"

namespace tona::gpu {

struct ForwardStats {
    int64_t expert_hits = 0, expert_misses = 0;
    double upload_bytes = 0;
};

class Forward {
public:
    // expert_slots: experts per layer in the VRAM cache; max_ctx: length of the KV cache
    Forward(const Qwen4ExpModel& m, int expert_slots, int64_t max_ctx);
    ~Forward();
    Forward(const Forward&) = delete;
    Forward& operator=(const Forward&) = delete;

    // One token at the next position; returns the logits [vocab]
    const std::vector<float>& step(int32_t token);
    int64_t pos() const { return pos_; }
    const ForwardStats& stats() const { return stats_; }
    size_t vram_bytes() const { return vram_; }

private:
    struct GR {
        const void *down, *up, *inject;
        const float* norm;
    };
    struct Expert {
        const void *gate, *up, *down;
    };
    struct LayerDev {
        GR gr_attn, gr_ffn;
        // DeltaNet
        const void *qkv = nullptr, *gate = nullptr, *beta = nullptr, *alpha = nullptr, *ssm_out = nullptr;
        const float *conv = nullptr, *dt = nullptr, *A = nullptr, *ssm_norm = nullptr;
        float *dn_state = nullptr, *dn_conv = nullptr;
        // QSA
        const void *q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr, *idx_q = nullptr, *idx_k = nullptr;
        const float *q_norm = nullptr, *k_norm = nullptr, *idx_q_norm = nullptr, *idx_k_norm = nullptr;
        float *K = nullptr, *V = nullptr, *idx_raw = nullptr, *idx_pool = nullptr;
        // MoE
        const void *router = nullptr, *sh_gate = nullptr, *sh_up = nullptr, *sh_down = nullptr, *sh_inp = nullptr;
        // Expert cache
        uint8_t* slots = nullptr;
        size_t slot_bytes = 0, off_up = 0, off_down = 0;
        std::list<int> lru;                                                       // front = most recently used
        std::unordered_map<int, std::pair<int, std::list<int>::iterator>> where;  // expert -> (slot, place in lru)
        std::vector<int> free_slots;
    };
    // The experts the router picked, with their normalized weights
    struct Routing {
        std::vector<int> idx;
        std::vector<float> weight;
    };
    // An expert of the current layer on its way through the cache
    struct Pending {
        int e;
        float weight;
        Expert ex;
    };

    const void* upload(const TensorInfo& t);
    const float* upload_f32(const TensorInfo& t);  // small, as F32
    float* alloc(size_t floats);
    GR upload_gr(const GatedResidual& g);
    void init_layer(uint32_t l);
    void init_qsa(LayerDev& D, const Layer& S);
    void init_deltanet(LayerDev& D, const Layer& S);
    void init_ple(const Layer& S);
    void init_expert_cache(LayerDev& D, const Layer& S);
    void init_buffers();
    // Matvec; with xq (x already as int8) through the int8 path if the format has it and it is on
    void mv(const TensorInfo& t, const void* w, const float* x, float* y, const Q8Vec* xq = nullptr);
    // Round x to int8 if the int8 path is on; then returns &out, otherwise nullptr
    const Q8Vec* quant(const float* x, int64_t n, const Q8Vec& out) const;
    void gr_mix(const GR& g, const TensorInfo& down, const TensorInfo& up, const TensorInfo* inject, float* mixed,
                float* inject_out);
    void deltanet(uint32_t l, float* y);
    void qsa(uint32_t l, float* y);
    int select_cells(int64_t n_pool, int64_t n_top);
    void moe(uint32_t l, float* y);
    Routing route(uint32_t l, const Q8Vec* xq);
    void run_expert(uint32_t l, const Pending& p, const Q8Vec* xq, float* y);
    void run_experts_cached(uint32_t l, const Routing& r, const Q8Vec* xq, float* y);
    void load_misses(uint32_t l, const std::vector<Pending>& miss);
    void shared_expert(uint32_t l, const Q8Vec* xq, float* y);
    void ple();
    Expert expert(uint32_t l, int e);
    // Claim a slot for a missing expert (without copying); false if it is already there
    bool reserve(uint32_t l, int e, Expert& out);

    const Qwen4ExpModel& m_;
    const Hparams& h_;
    int64_t pos_ = 0, max_ctx_;
    int slots_;
    std::vector<LayerDev> L_;
    GR head_{};
    const void* output_ = nullptr;
    const void *ple_key_ = nullptr, *ple_val_ = nullptr;
    const float *ple_wk_ = nullptr, *ple_wq_ = nullptr, *ple_wc_ = nullptr, *ple_conv_ = nullptr;
    float* ple_hist_ = nullptr;

    // Work buffers
    float *R_, *xn_, *lo_, *gate_, *mixed_, *inject_, *y_, *e_, *t1_, *t2_, *t3_, *t4_, *t5_, *t6_, *logits_;
    int32_t* cells_;
    // The int8 path is only on with TONA_GPU_INT8: with one kernel per step the extra quantize kernels cost more than
    // the faster matvecs save (26.8 against 27.6 tok/s decode without reloading,
    // research/measurements/2026-10-23-matvec-int8.md)
    bool use_q8_ = false;
    Q8Vec qmix_{}, qtmp_{};
    std::vector<float> logits_host_, router_host_, scores_host_;
    std::vector<int32_t> tokens_;
    // Reloading: pinned staging buffer, own copy stream
    uint8_t* staging_ = nullptr;
    size_t staging_bytes_ = 0;
    hipStream_t copy_stream_ = nullptr;
    hipEvent_t copied_ = nullptr;
    std::vector<void*> allocs_;
    size_t vram_ = 0;
    ForwardStats stats_;
};

}  // namespace tona::gpu
