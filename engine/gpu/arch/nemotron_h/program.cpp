// Bolt as an ArchProgram: construction, shape checks and the flow of the target program (embedding, blocks, head).
#include "gpu/arch/nemotron_h/program.h"

#include <algorithm>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/arch/nemotron_h/prefill.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void nemotron_h_check_shapes(const NemotronHMoeModel& m, const backend::ExpertLayout& experts) {
    const NemotronHparams& h = m.hp();
    latent_check_shapes(m);
    uint32_t kv = 0;
    for (const NemotronLayer& L : m.layers())
        if (L.kind == BoltBlockKind::Attention) kv = kv ? kv : L.n_kv;
    bool ok = h.head_dim == 128 && h.d % 128 == 0 && kv > 0 && h.n_head / kv <= 16 &&
              experts.max_experts() <= (uint32_t)MEGA_MAX_EXPERTS_WIDE && h.ssm_conv >= 2 && h.ssm_inner % 32 == 0;
    // One KV row geometry for all attention blocks (Mega::kv_row_)
    for (const NemotronLayer& L : m.layers()) ok = ok && (L.kind != BoltBlockKind::Attention || L.n_kv == kv);
    if (!ok) throw std::runtime_error("gpu::Mega: shapes of the model do not match the fixed kernel assumptions");
}

NemotronHProgram::NemotronHProgram(ProgramBuilder& pb)
    : pb_(pb),
      g_(pb.mega()),
      m_(model_as<NemotronHMoeModel>(pb.base())),
      h_(m_.hp()),
      dm_(dims_of(h_, m_)),
      moe_(pb, m_),
      tap_(env::flag("TONA_BOLT_TAP")) {}

NemotronHProgram::Dims NemotronHProgram::dims_of(const NemotronHparams& h, const NemotronHMoeModel& m) {
    Dims s{};
    const Mamba2Dims D = h.mamba2();
    s.d = (int)h.d, s.inner = (int)h.ssm_inner, s.heads = (int)h.ssm_heads, s.state = (int)h.ssm_state;
    s.groups = (int)h.ssm_groups, s.conv = (int)h.ssm_conv, s.conv_dim = (int)D.conv_dim();
    s.in_dim = (int)D.in_proj_dim();
    s.nh = (int)h.n_head, s.hd = (int)h.head_dim;
    for (const NemotronLayer& L : m.layers())
        if (L.kind == BoltBlockKind::Attention) s.nkv = (int)L.n_kv;
    s.ost = std::max(s.inner, s.nh * s.hd);  // output of a Mamba-2 or attention block per token
    s.eps = h.rms_eps;
    return s;
}

void NemotronHProgram::build_mtp() { throw std::runtime_error("gpu::Mega: Bolt has no MTP head"); }
void NemotronHProgram::build_drafter() { throw std::runtime_error("gpu::Mega: Bolt has no drafter"); }

// The embedding row of every token becomes the residual stream; its norm feeds the first block
void NemotronHProgram::build_target() {
    res_ = &NemotronPfRes::reset(g_.pf_arch_, h_.n_layer);
    const float* first = g_.upload_f32(*m_.layers()[0].attn_norm);
    res_->norm[0] = first;
    pb_.op(OP_RES_NORM, {g_.e_, nullptr, (void*)first, b_.R, b_.xn, b_.qxn.q, b_.qxn.meta}, {dm_.d}, {dm_.eps});
    for (uint32_t b = 0; b < h_.n_layer; b++) block(b);
    head();
}

const float* NemotronHProgram::residual_norm(const float* y, const TensorInfo& w) {
    // In place: every block has its sums before the first write (barrier inside the step)
    const float* gain = g_.upload_f32(w);
    pb_.op(OP_RES_NORM, {b_.R, (void*)y, (void*)gain, b_.R, b_.xn, b_.qxn.q, b_.qxn.meta}, {dm_.d}, {dm_.eps});
    return gain;
}

// One block, then the residual add and the norm of the next block's input. The latent MoE is the i-th of its kind: the
// expert caches and the host's selection are indexed by that ordinal.
void NemotronHProgram::block(uint32_t b) {
    const NemotronLayer& L = m_.layers()[b];
    PfLayer& F = g_.pf_[b];
    switch (L.kind) {
        case BoltBlockKind::Mamba2:
            mamba(L, F.mamba);
            break;
        case BoltBlockKind::Attention:
            attention(L, F.attn);
            break;
        case BoltBlockKind::Moe:
            res_->ordinal[b] = (int)moe_ord_;
            moe_.build(moe_ord_++, {b_.xn, &b_.qxn, b_.y});
            break;
    }
    const bool last = b + 1 == h_.n_layer;
    const float* next = residual_norm(b_.y, last ? *m_.output_norm() : *m_.layers()[b + 1].attn_norm);
    (last ? res_->head_norm : res_->norm[b + 1]) = next;
    if (tap_) tap_block(b);
}

void NemotronHProgram::head() {
    pb_.mv_begin();
    g_.head_w_ = g_.upload(*m_.output());
    pb_.addw(*m_.output(), g_.head_w_, b_.xn, &b_.qxn, g_.logits_);
    pb_.mv_end("head.out");
    g_.pf_out_ = pb_.pw(*m_.output(), g_.head_w_);
}

// TONA_BOLT_TAP=1: the residual stream after every block is copied to a buffer of its own (y = R + 0), so that a test
// can compare each block against the oracle's trace (Mega::taps)
void NemotronHProgram::tap_block(uint32_t b) {
    float* dst = g_.alloc_b(dm_.d);
    pb_.op(OP_MOE_ADD, {b_.R, tap_zero_, dst}, {dm_.d});
    g_.taps_.emplace_back("block" + std::to_string(b), dst);
}

}  // namespace tona::gpu
