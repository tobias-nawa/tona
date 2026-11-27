// Buffers of Bolt's blocks: the residual stream, the Mamba-2 and attention steps, the latent MoE, logits and the KV
// geometry.
#include <algorithm>

#include "backend/env.h"
#include "gpu/arch/nemotron_h/kernel_set.h"
#include "gpu/arch/nemotron_h/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void NemotronHProgram::alloc_buffers() {
    b_.R = g_.alloc_b(dm_.d);
    b_.xn = g_.alloc_b(dm_.d);
    b_.qxn = g_.alloc_q8_b(dm_.d);
    b_.y = g_.alloc_b(dm_.d);
    alloc_mixer_buffers();
    setup_kv();
    moe_.alloc_buffers();
    g_.e_ = g_.alloc_b(dm_.d);
    g_.logits_ = g_.alloc_b(h_.vocab);
    g_.logits_host_.resize((size_t)MEGA_BMAX * h_.vocab);
    g_.logits1_.resize(h_.vocab);
    if (tap_) {
        tap_zero_ = g_.alloc_b(dm_.d);
        HIPCHECK(hipMemset(tap_zero_, 0, (size_t)dm_.d * MEGA_BMAX * sizeof(float)));
    }
}

void NemotronHProgram::alloc_mixer_buffers() {
    const Dims& s = dm_;
    b_.t_zx = g_.alloc_b(s.in_dim);
    b_.t_conv = g_.alloc_b(s.conv_dim);
    b_.t_y = g_.alloc_b(s.inner);
    b_.t_o = g_.alloc_b(s.ost);
    b_.q_o = g_.alloc_q8_b(s.ost);
    b_.t_q = g_.alloc_b(s.nh * s.hd);
    b_.t_qn = g_.alloc_b(s.nh * s.hd);
    b_.t_k = g_.alloc_b(s.nkv * s.hd);
    b_.t_v = g_.alloc_b(s.nkv * s.hd);
    // Partial results of the split attention: sized for the largest pass whatever the context (mega_op_fattn.h)
    const size_t floats = (size_t)MEGA_BMAX * s.nh * kBoltFattnSplits * (s.hd + 2);
    b_.part = g_.alloc(floats);
    g_.rows_[reinterpret_cast<uintptr_t>(b_.part)] = {
        floats * sizeof(float), (int64_t)s.nh * kBoltFattnSplits * (s.hd + 2) * (int64_t)sizeof(float)};
}

// KV cache in int8 (K and V with a scale per 32 values) as Ornith's, split by position between VRAM and pinned RAM
// within a budget (kv_place.h); TONA_KV_VRAM_GB sets the budget, TONA_KV_HOST=0/1 forces all of it into one place. The
// row is the one of fattn_prep_kv. Every attention block has the same number of KV heads (nemotron_h_check_shapes).
void NemotronHProgram::setup_kv() {
    const Dims& s = dm_;
    g_.kv_row_ = (int64_t)s.nkv * 2 * s.hd + (int64_t)s.nkv * 2 * (s.hd / 32) * (int64_t)sizeof(float);
    kv_main_ = (size_t)m_.count(BoltBlockKind::Attention);
    g_.kv_bytes_ = kv_main_ * g_.max_ctx_ * g_.kv_row_;
    const int64_t force = env::integer("TONA_KV_HOST", -1);
    if (force >= 0) {
        g_.kv_host_ = force == 1;
        return;
    }
    g_.kv_split_ = plan_kv_split(kv_main_, g_.mtp_on_ ? 1 : 0, g_.max_ctx_, g_.kv_row_);
    g_.kv_host_ = g_.kv_split_.host_frac > 0.0;
}

}  // namespace tona::gpu
