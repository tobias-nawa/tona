// Ornith as an ArchProgram: construction, shape checks and the flow of the target program (embedding, layers, head).
#include "gpu/arch/qwen35moe/program.h"

#include <cstdio>
#include <stdexcept>
#include <string>

#include "backend/env.h"
#include "gpu/arch/qwen35moe/prefill.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_tap.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void qwen35moe_check_shapes(const Qwen35MoeModel& m, const backend::ExpertLayout& experts) {
    const Qwen35Hparams& h = m.hp();
    if (h.dn_head_dim != DS || h.head_dim != 256 || h.rope_dims != 64 || h.d % 128 != 0 ||
        h.n_expert_used * BMAX > UMAX || experts.max_experts() > (uint32_t)MEGA_MAX_EXPERTS ||
        h.n_head / std::max<uint32_t>(h.n_head_kv, 1) > 16 || h.dn_conv < 2)
        throw std::runtime_error("gpu::Mega: shapes of the model do not match the fixed kernel assumptions");
}

Qwen35MoeProgram::Qwen35MoeProgram(ProgramBuilder& pb)
    : pb_(pb), g_(pb.mega()), m_(pb.ornith()), h_(pb.ornith().hp()), dm_(dims_of(pb.ornith().hp())) {}

Qwen35MoeProgram::Dims Qwen35MoeProgram::dims_of(const Qwen35Hparams& h) {
    Dims s{};
    s.d = (int)h.d, s.ff = (int)h.expert_ff, s.sff = (int)h.shexp_ff, s.ne = (int)h.n_expert_used;
    s.kd = (int)(h.dn_k_heads * h.dn_head_dim), s.vd = (int)(h.dn_v_heads * h.dn_head_dim);
    s.nk = (int)h.dn_k_heads, s.nv = (int)h.dn_v_heads;
    s.nh = (int)h.n_head, s.nkv = (int)h.n_head_kv, s.hd = (int)h.head_dim;
    s.ch = 2 * s.kd + s.vd;
    s.ost = std::max(s.vd, s.nh * s.hd);  // output of attention or DeltaNet per token
    s.eps = h.rms_eps, s.base = h.rope_base;
    return s;
}

void Qwen35MoeProgram::build_drafter() { throw std::runtime_error("gpu::Mega: Ornith has no drafter"); }

// The embedding row of every token becomes the residual stream; its norm feeds the first layer
void Qwen35MoeProgram::build_target() {
    res_ = &Qwen35PfRes::reset(g_.pf_arch_, h_.n_layer);
    const float* first = g_.upload_f32(*m_.layers()[0].attn_norm);
    res_->norm_attn[0] = first;
    pb_.op(OP_RES_NORM, {g_.e_, nullptr, (void*)first, b_.R, b_.xn, b_.qxn.q, b_.qxn.meta}, {dm_.d}, {dm_.eps});
    for (uint32_t l = 0; l < h_.n_layer; l++) layer(l);
    head();
}

const float* Qwen35MoeProgram::residual_norm(const float* y, const TensorInfo& w) {
    // In place: every block has its sums before the first write (barrier inside the step)
    const float* gain = g_.upload_f32(w);
    pb_.op(OP_RES_NORM, {b_.R, (void*)y, (void*)gain, b_.R, b_.xn, b_.qxn.q, b_.qxn.meta}, {dm_.d}, {dm_.eps});
    return gain;
}

// Mixer (DeltaNet or attention), residual and norm, router and experts, residual and the norm of the next consumer
void Qwen35MoeProgram::layer(uint32_t l) {
    const Qwen35Layer& S = m_.layers()[l];
    PfLayer& F = g_.pf_[l];
    if (S.kind == Qwen35LayerKind::GatedAttn)
        layer_attention(S, F);
    else
        layer_deltanet(S, F);
    res_->norm_ffn[l] = residual_norm(b_.y, *S.post_attn_norm);
    tap();
    layer_router(l, S, F);
    experts(l);
    const bool last = l + 1 == h_.n_layer;
    const float* next = residual_norm(b_.y, last ? *m_.output_norm() : *m_.layers()[l + 1].attn_norm);
    (last ? res_->head_norm : res_->norm_attn[l + 1]) = next;
    tap();
}

void Qwen35MoeProgram::tap() {
    static const char* const out = env::text("TONA_TAP_OUT");
    if (!out) return;
    if (!g_.tap_) {
        g_.tap_ = new MegaTap;
        g_.tap_->d = dm_.d;
        g_.tap_->fp = std::fopen(out, "wb");
        if (!g_.tap_->fp) throw std::runtime_error(std::string("TONA_TAP_OUT: cannot open ") + out);
    }
    float* buf = g_.alloc_b(dm_.d);
    g_.tap_->bufs.push_back(buf);
    // OP_RES_NORM without y and without a gain is a plain copy of R
    pb_.op(OP_RES_NORM, {b_.R, nullptr, nullptr, buf, nullptr, nullptr, nullptr}, {dm_.d}, {dm_.eps});
}

void Qwen35MoeProgram::head() {
    pb_.mv_begin();
    g_.head_w_ = g_.upload(*m_.output());
    pb_.addw(*m_.output(), g_.head_w_, b_.xn, &b_.qxn, g_.logits_);
    pb_.mv_end("head.out");
    g_.pf_out_ = pb_.pw(*m_.output(), g_.head_w_);
}

}  // namespace tona::gpu
