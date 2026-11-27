// The latent MoE block of Bolt (nemotron_h_moe) as a building block of the architecture's program: per-layer geometry,
// the buffers shared by all MoE blocks, and the ops of one block in the decode program (router, latent down, shared
// expert, experts, latent up). The architecture's program (B6) calls it once per MoE block with the block's input
// (norm output, float and int8) and output (y, added to the residual by the next residual step); the prefill side is
// moe_prefill.cpp. Host and cold path only.
#pragma once

#include <cstdint>
#include <vector>

#include "gpu/mega_build.h"
#include "model/nemotron_h_moe.h"

namespace tona::gpu {

// What one MoE block's ops depend on, read from the block's tensors and arrays, never from the model's maxima
struct LatentGeom {
    int d, lat;      // model width, latent width
    int experts, k;  // experts of the block, experts per token
    int ff, sff;     // expert width in the latent space, shared expert width (full width)
    int act_n;       // UMAX * ff + sff: values per token of the activations of this block
};

// Maxima over the blocks, for sizing the buffers all blocks share
struct LatentMax {
    int lat = 0, ff = 0, sff = 0, k = 0, act_n = 0;
};

LatentGeom latent_geom(const NemotronHMoeModel& m, uint32_t block);
LatentMax latent_max(const NemotronHMoeModel& m);

// Throws for a model the ops are not written for (latent width not a multiple of 128, k * BMAX above the places of
// the union, experts above the router limit)
void latent_check_shapes(const NemotronHMoeModel& m);

class LatentMoe {
public:
    // The block's input and output buffers, owned by the architecture's program
    struct Io {
        float* xn;
        const Q8Vec* qxn;
        float* y;
    };

    LatentMoe(ProgramBuilder& pb, const NemotronHMoeModel& m);
    // Per-token buffers of the blocks; call once, before the first build
    void alloc_buffers();
    // Appends the ops of the i-th MoE block (the cache, the expert layout and the prefill descriptors of the core are
    // indexed by this ordinal, not by the block's place among the 88) and records its prefill descriptor
    void build(uint32_t i, const Io& io);

private:
    struct Bufs {
        float *router_out = nullptr, *xl = nullptr, *eu = nullptr, *act = nullptr, *ed = nullptr, *sh_d = nullptr,
              *yl = nullptr, *y_up = nullptr;
        Q8Vec qxl{}, q_act{}, q_yl{};
    };
    const void* router_weights(const TensorInfo& t, uint32_t i);
    void route_policy(uint32_t i, const NemotronLayer& L);
    void router_and_latent(uint32_t i, const LatentGeom& G, const MoeWeights& S, const Io& io);
    void shared_up(uint32_t i, const MoeWeights& S, const LatentGeom& G, const Io& io);
    void experts(uint32_t i, const LatentGeom& G, const MoeWeights& S, const Io& io);
    void record_prefill(uint32_t i, const NemotronLayer& L);

    ProgramBuilder& pb_;
    Mega& g_;
    const NemotronHMoeModel& m_;
    LatentMax mx_;
    Bufs b_;
    std::vector<const void*> router_dev_;
};

}  // namespace tona::gpu
