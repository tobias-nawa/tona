// The Coder (qwen4exp) as a program: 4 gated residual streams, DeltaNet and QSA layers, MoE with a shared expert, PLE,
// the MTP head and the DFlash2 drafter. The toolbox (ops, slots, upload) is ProgramBuilder; this class owns everything
// that only this architecture has. One instance lives for one Mega::build().
#pragma once

#include <string>
#include <vector>

#include "gpu/arch/qwen4exp/prefill.h"
#include "gpu/arch_program.h"
#include "gpu/mega_build.h"

namespace tona::gpu {

struct Qwen4ExpPfRes;

// The shapes the kernels of the Coder are written for (throws on others)
void qwen4exp_check_shapes(const Qwen4ExpModel& m, const backend::ExpertLayout& experts);

class Qwen4ExpProgram final : public ArchProgram {
public:
    explicit Qwen4ExpProgram(ProgramBuilder& pb);
    void alloc_buffers() override;
    void build_target() override;
    void build_mtp() override;
    void build_drafter() override;

private:
    // Shapes derived from the hyperparameters
    struct Dims {
        int hc, d, ff, sff, ne;
        int64_t hd_all;
        int kd, vd, nk, nv, nh, nkv, hd, idim, blk, ch, ost;
        float eps, base;
    };
    // Per-token buffers shared by the programs (they run one after the other)
    struct Bufs {
        float* R[2] = {};
        float *xn = nullptr, *lo = nullptr, *gate = nullptr, *mixed = nullptr;
        Q8Vec qmix{}, qxn{}, qlo{};
        float *inject_a = nullptr, *inject_f = nullptr, *y = nullptr;
        float *t_qkv = nullptr, *t_z = nullptr, *t_b = nullptr, *t_a = nullptr, *t_conv = nullptr, *t_o = nullptr,
              *t_raw = nullptr;
        Q8Vec q_o{};
        float *t_qfull = nullptr, *t_qn = nullptr, *t_g = nullptr, *t_idxq = nullptr, *t_k = nullptr, *t_v = nullptr,
              *t_rk = nullptr;
        float *scores = nullptr, *attn_part = nullptr;
        int32_t* cells = nullptr;
        int act_n = 0;
        float *eg = nullptr, *eu = nullptr, *act = nullptr;
        Q8Vec q_act{};
        float *ed = nullptr, *sh_d = nullptr, *sh_inp = nullptr;
        float *ple_k = nullptr, *ple_v = nullptr, *ple_g = nullptr, *ple_n = nullptr, *ple_hist = nullptr;
        float *router_out = nullptr, *pred_out = nullptr;
    };
    // Shapes of the drafter model (DFlash2)
    struct DraftDims {
        int nh, nkv, hd, ff, layers, group, swa, ring, p4, qd, kvd;
        float eps, base;
        bool attn_old;
    };
    // Per-layer weights and caches of the drafter, filled while the inject program is built
    struct DraftLayers {
        std::vector<float*> Kc, Vc, kin, vin;
        std::vector<const void*> wk, wv;
        std::vector<const float*> knorm;
    };
    // Drafter program: buffers of the block
    struct DraftBufs {
        float *x = nullptr, *hn = nullptr, *dyn = nullptr, *h2 = nullptr, *qb = nullptr, *kb = nullptr, *vb = nullptr,
              *ob = nullptr, *ab = nullptr, *fg = nullptr, *fu = nullptr, *fa = nullptr;
        Q8Vec qhn{}, qh2{}, qob{}, qfa{};
    };

    static Dims dims_of(const Hparams& h);

    // --- buffers (buffers.cpp)
    void alloc_streams();
    void alloc_step_buffers();
    void setup_kv();
    void alloc_attention_buffers();
    void alloc_moe_buffers();

    // --- gated residual and attention steps (residual.cpp)
    void gr(const GatedResidual& S, float* inject_out, PfGR* rec);
    void combine(const float* yy, const float* inj, const GatedResidual* next);
    void feat(uint32_t l);
    void emit_attention(uint8_t* kv, float* pool);

    // --- target program (layer.cpp, moe.cpp)
    void layer(uint32_t l);
    void layer_qsa(const Layer& S, PfLayer& F);
    void layer_deltanet(const Layer& S, PfLayer& F);
    void layer_router(uint32_t l, const Layer& S, PfLayer& F);
    void between_layers(uint32_t l);
    void ple_block(uint32_t l);
    void head_block();
    const void* router_weights(uint32_t l);
    void shared_matvecs(const Layer& S, PfLayer& F, bool ple);
    void shared(uint32_t l, bool ple);
    void experts(uint32_t l, bool ple);
    void expert_gate_up(uint32_t l, const Layer& S, PfLayer& F, bool ple, int rep);
    void expert_down(const Layer& S, const LayerCache& C, PfLayer& F, int rep);

    // --- MTP head (mtp.cpp, mtp_dyn.cpp)
    void mtp_enter();
    void mtp_input(const Layer& S, const MtpHead& M, PfGR* pgr);
    void mtp_attention(const Layer& S, const MtpHead& M);
    void mtp_moe(const Layer& S);
    void mtp_head();
    void mtp_truncated_head(const char* vf);
    void mtp_pick_ops();
    void mtp_upload_map();
    int mtp_dyn_rank() const;
    bool mtp_dyn_setup();
    void mtp_dyn_ops(int64_t rb);
    void mtp_leave();

    // --- drafter (draft.cpp)
    DraftDims draft_dims(const Gguf& G);
    void draft_inject(const Gguf& G, const DraftDims& D, DraftLayers& L);
    void draft_block_buffers(const DraftDims& D);
    void draft_layer(const Gguf& G, const DraftDims& D, const DraftLayers& L, int l);
    void draft_attention_half(const Gguf& G, const DraftDims& D, const DraftLayers& L, int l, const float* abase);
    void draft_attention_op(const DraftDims& D, const DraftLayers& L, int l);
    void draft_ffn_half(const Gguf& G, const DraftDims& D, int l, const float* fbase);
    void draft_head(const Gguf& G);
    void draft_finish();

    ProgramBuilder& pb_;
    Mega& g_;
    const Hparams& h_;
    const Qwen4ExpModel& m_;
    const Dims dm_;
    Bufs b_;
    int cur_ = 0;
    Qwen4ExpPfRes* res_ = nullptr;  // the gated residuals recorded for the prefill (prefill.h)
    const bool hc8_;                // hyper-connection down/up as Q8_0 instead of BF16 (1.28 GB -> 0.68 GB per token)
    std::vector<bool> idx_full_;    // per QSA layer: runs its own indexer (TONA_IDX_SHARE)
    bool idx_shared_ = false;       // the QSA layer being built reuses the selection of the last full one
    bool attn_old_ = false, attn_wave_ = false;
    std::vector<const void*> router_dev_;
    DraftBufs db_;
    float* dattn_part_ = nullptr;
    // MTP program: the buffers of its own and the target's residual streams while it is being built
    float* mtp_mx_ = nullptr;
    float* mtp_router_ = nullptr;
    Q8Vec mtp_mq_{};
    float* saved_R_[2] = {};
    int saved_cur_ = 0;
};

}  // namespace tona::gpu
