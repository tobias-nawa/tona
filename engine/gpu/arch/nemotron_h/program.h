// Bolt (nemotron_h_moe) as a program: one plain residual stream, 88 blocks of three kinds (Mamba-2, full-context
// attention without gate, QK norm and RoPE, latent MoE), each behind its own norm. The toolbox (ops, slots, upload) is
// ProgramBuilder; this class owns the Mamba-2 and attention blocks and the flow of the trunk, the latent MoE block is
// LatentMoe (moe.h). One instance lives for one Mega::build(). No MTP head and no drafter in the file. The prompt runs
// through the layered prefill (prefill_*.cpp in this directory); the descriptors it reads (PfLayer, one per block) are
// filled here while the decode program is built.
#pragma once

#include <vector>

#include "gpu/arch/nemotron_h/moe.h"
#include "gpu/arch_program.h"
#include "gpu/mega_build.h"

namespace tona::gpu {

struct NemotronPfRes;

// The shapes the kernels of Bolt are written for (throws on others)
void nemotron_h_check_shapes(const NemotronHMoeModel& m, const backend::ExpertLayout& experts);

class NemotronHProgram final : public ArchProgram {
public:
    explicit NemotronHProgram(ProgramBuilder& pb);
    void alloc_buffers() override;
    void build_target() override;
    void build_mtp() override;
    void build_drafter() override;

private:
    struct Dims {
        int d, inner, heads, state, groups, conv, conv_dim, in_dim;
        int nh, nkv, hd, ost;
        float eps;
    };
    // Per-token buffers shared by all blocks
    struct Bufs {
        float *R = nullptr, *xn = nullptr, *y = nullptr;
        Q8Vec qxn{};
        float *t_zx = nullptr, *t_conv = nullptr, *t_y = nullptr, *t_o = nullptr;
        Q8Vec q_o{};
        float *t_q = nullptr, *t_qn = nullptr, *t_k = nullptr, *t_v = nullptr, *part = nullptr;
    };

    static Dims dims_of(const NemotronHparams& h, const NemotronHMoeModel& m);

    // --- buffers (buffers.cpp)
    void alloc_mixer_buffers();
    void setup_kv();

    // --- target program (program.cpp, blocks.cpp)
    void block(uint32_t b);
    void mamba(const NemotronLayer& L, PfMamba2& P);
    void attention(const NemotronLayer& L, PfGatedAttn& A);
    void tap_block(uint32_t b);
    // R' = R + y, xn = norm(R') * w for the next consumer; returns the uploaded gain (the prefill reads it)
    const float* residual_norm(const float* y, const TensorInfo& w);
    void head();

    size_t kv_main_ = 0;  // attention blocks (KvSplit buffers)
    ProgramBuilder& pb_;
    Mega& g_;
    const NemotronHMoeModel& m_;
    const NemotronHparams& h_;
    const Dims dm_;
    Bufs b_;
    LatentMoe moe_;
    uint32_t moe_ord_ = 0;  // the next MoE block's ordinal among the MoE blocks
    bool tap_;              // TONA_BOLT_TAP
    float* tap_zero_ = nullptr;
    NemotronPfRes* res_ = nullptr;  // the norm gains recorded for the prefill (prefill.h)
};

}  // namespace tona::gpu
