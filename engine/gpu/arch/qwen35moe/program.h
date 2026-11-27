// Ornith (qwen35moe) as a program: one plain residual stream, gated DeltaNet and full-context gated attention layers,
// MoE with a shared expert, no PLE and no hyper-connections. The toolbox (ops, slots, upload) is ProgramBuilder; this
// class owns what only this architecture has. One instance lives for one Mega::build(). No drafter. The prompt
// runs through the layered prefill (prefill_*.cpp in this directory); the layer descriptors it reads (PfLayer) are
// filled here while the decode program is built.
#pragma once

#include <vector>

#include "gpu/arch_program.h"
#include "gpu/mega_build.h"

namespace tona::gpu {

struct Qwen35PfRes;

// The shapes the kernels of Ornith are written for (throws on others)
void qwen35moe_check_shapes(const Qwen35MoeModel& m, const backend::ExpertLayout& experts);

class Qwen35MoeProgram final : public ArchProgram {
public:
    explicit Qwen35MoeProgram(ProgramBuilder& pb);
    void alloc_buffers() override;
    void build_target() override;
    void build_mtp() override;
    void build_drafter() override;

private:
    struct Dims {
        int d, ff, sff, ne;
        int kd, vd, nk, nv, nh, nkv, hd, ch, ost;
        float eps, base;
    };
    // Per-token buffers shared by all layers
    struct Bufs {
        float *R = nullptr, *xn = nullptr, *y = nullptr;
        Q8Vec qxn{};
        float *t_qkv = nullptr, *t_z = nullptr, *t_b = nullptr, *t_a = nullptr, *t_conv = nullptr, *t_o = nullptr,
              *t_raw = nullptr;
        Q8Vec q_o{};
        float *t_qfull = nullptr, *t_qn = nullptr, *t_g = nullptr, *t_k = nullptr, *t_v = nullptr, *part = nullptr;
        int act_n = 0;
        float *eg = nullptr, *eu = nullptr, *act = nullptr;
        Q8Vec q_act{};
        float *ed = nullptr, *sh_d = nullptr, *sh_inp = nullptr, *router_out = nullptr, *pred_out = nullptr;
    };

    static Dims dims_of(const Qwen35Hparams& h);

    // --- buffers (buffers.cpp)
    void alloc_streams();
    void alloc_mixer_buffers();
    void setup_kv();
    // Cells of KV buffer idx (the attention layers, then the MTP layer) that live in VRAM
    int64_t kv_cells_in_vram(size_t idx) const {
        return g_.kv_split_.planned() ? g_.kv_split_.cells_in_vram(idx, kv_main_, g_.max_ctx_)
                                      : (g_.kv_host_ ? 0 : g_.max_ctx_);
    }
    void alloc_moe_buffers();

    // --- target program (program.cpp, layers.cpp, moe.cpp)
    void layer(uint32_t l);
    void layer_deltanet(const Qwen35Layer& S, PfLayer& F);
    void layer_attention(const Qwen35Layer& S, PfLayer& F);
    // R' = R + y, xn = norm(R') * w for the next consumer
    // Returns the uploaded gain: the prefill reads it from the layer descriptors
    const float* residual_norm(const float* y, const TensorInfo& w);
    void head();
    // TONA_TAP_OUT: a copy step of the residual into the next tap buffer (gpu/mega_tap.h); nothing without the variable
    void tap();
    const void* router_weights(uint32_t l);
    void layer_router(uint32_t l, const Qwen35Layer& S, PfLayer& F);
    void shared(uint32_t l);
    void shared_matvecs(const MoeWeights& S, PfLayer& F);
    void experts(uint32_t l);
    void expert_gate_up(uint32_t l, const MoeWeights& S, PfLayer& F);
    void expert_down(const MoeWeights& S, const LayerCache& C, PfLayer& F);

    // --- MTP head (mtp.cpp)
    struct MtpState {
        float *mx = nullptr, *e = nullptr, *R = nullptr, *router = nullptr;
        Q8Vec mq{};
        float* saved_R = nullptr;
        uint8_t* experts = nullptr;
        int64_t up = 0, down = 0, bytes = 0;  // offsets and size of one expert: [gate | up | down]
    };
    void mtp_enter();
    void mtp_input(const Qwen35Mtp& M);
    void mtp_layer(const Qwen35Mtp& M);
    void mtp_experts_upload(const MoeWeights& S);
    void mtp_moe(const MoeWeights& S);
    void mtp_truncated_head(const char* vf);
    void mtp_head_and_pick();
    void mtp_leave();

    ProgramBuilder& pb_;
    Mega& g_;
    const Qwen35MoeModel& m_;
    const Qwen35Hparams& h_;
    const Dims dm_;
    Bufs b_;
    Qwen35PfRes* res_ = nullptr;  // the norm gains recorded for the prefill (prefill.h)
    MtpState mt_;
    std::vector<const void*> router_dev_;
    size_t kv_allocated_ = 0;  // KV buffers allocated so far (attention layers, then the MTP layer)
    size_t kv_main_ = 0;       // attention layers of the trunk
};

}  // namespace tona::gpu
