// Buffers of Ornith's blocks: the residual stream, DeltaNet and attention steps, MoE, logits and the KV geometry.
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "backend/env.h"
#include "gpu/arch/qwen35moe/kernel_set.h"
#include "gpu/arch/qwen35moe/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void Qwen35MoeProgram::alloc_buffers() {
    alloc_streams();
    alloc_mixer_buffers();
    setup_kv();
    alloc_moe_buffers();
    router_dev_.assign(h_.n_layer, nullptr);
}

void Qwen35MoeProgram::alloc_streams() {
    b_.R = g_.alloc_b(dm_.d);
    b_.xn = g_.alloc_b(dm_.d);
    b_.qxn = g_.alloc_q8_b(dm_.d);
    b_.y = g_.alloc_b(dm_.d);
}

void Qwen35MoeProgram::alloc_mixer_buffers() {
    const Dims& s = dm_;
    b_.t_qkv = g_.alloc_b(s.ch);
    b_.t_z = g_.alloc_b(s.vd);
    b_.t_b = g_.alloc_b(s.nv);
    b_.t_a = g_.alloc_b(s.nv);
    b_.t_conv = g_.alloc_b(s.ch);
    b_.t_o = g_.alloc_b(s.ost);
    b_.t_raw = g_.alloc_b(s.vd);
    b_.q_o = g_.alloc_q8_b(s.ost);
    b_.t_qfull = g_.alloc_b(2 * s.nh * s.hd);
    b_.t_qn = g_.alloc_b(s.nh * s.hd);
    b_.t_g = g_.alloc_b(s.nh * s.hd);
    b_.t_k = g_.alloc_b(s.nkv * s.hd);
    b_.t_v = g_.alloc_b(s.nkv * s.hd);
    // Partial results of the split attention: sized for the largest pass whatever the context (mega_op_fattn.h)
    const size_t floats = (size_t)MEGA_BMAX * s.nh * kFattnSplits * (s.hd + 2);
    b_.part = g_.alloc(floats);
    g_.rows_[reinterpret_cast<uintptr_t>(b_.part)] = {
        floats * sizeof(float), (int64_t)s.nh * kFattnSplits * (s.hd + 2) * (int64_t)sizeof(float)};
}

// KV cache in int8 (K and V with a scale per 32 values), as the Coder's, but split by position: the first cells of
// every attention layer in VRAM within a budget, the rest in pinned RAM (kv_place.h). The decode attention reads RAM at
// 18 to 24 GB/s, so 64K of context cost 36 ms per pass from RAM; with the first cells of all layers in VRAM a context
// up to them is read from VRAM at any max_ctx (research/measurements/2026-11-20-kv-hot-head.md). The budget is a share
// of the free VRAM (KvSplit::default_budget: 128,000 cells, at most 13.5 %); TONA_KV_VRAM_GB sets it, TONA_KV_HOST=0/1
// forces all of it into one place. The MTP layer's cache stays whole, first in the budget. The row is the one of
// fattn_prep_kv.
void Qwen35MoeProgram::setup_kv() {
    const Dims& s = dm_;
    g_.kv_row_ = (int64_t)s.nkv * 2 * s.hd + (int64_t)s.nkv * 2 * (s.hd / 32) * (int64_t)sizeof(float);
    int n_attn = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) n_attn += h_.is_attn(l);
    kv_main_ = (size_t)n_attn;
    g_.kv_bytes_ = (size_t)n_attn * g_.max_ctx_ * g_.kv_row_;
    const int64_t force = env::integer("TONA_KV_HOST", -1);
    if (force >= 0) {
        g_.kv_host_ = force == 1;
        return;
    }
    g_.kv_split_ = plan_kv_split(kv_main_, g_.mtp_on_ ? 1 : 0, g_.max_ctx_, g_.kv_row_);
    g_.kv_host_ = g_.kv_split_.host_frac > 0.0;
}

// MoE: UMAX places per token for the union, the shared expert behind them; logits
void Qwen35MoeProgram::alloc_moe_buffers() {
    const Dims& s = dm_;
    b_.act_n = UMAX * s.ff + s.sff;
    b_.eg = g_.alloc_b(b_.act_n);
    b_.eu = g_.alloc_b(b_.act_n);
    b_.act = g_.alloc_b(b_.act_n);
    b_.q_act = g_.alloc_q8_b(b_.act_n);
    b_.ed = g_.alloc_b((size_t)UMAX * s.d);
    b_.sh_d = g_.alloc_b(s.d);
    b_.sh_inp = g_.alloc_b(32);
    g_.e_ = g_.alloc_b(s.d);
    g_.logits_ = g_.alloc_b(h_.vocab);
    g_.logits_host_.resize((size_t)MEGA_BMAX * h_.vocab);
    g_.logits1_.resize(h_.vocab);
    b_.router_out = g_.alloc_b(MEGA_MAX_EXPERTS);
    b_.pred_out = g_.alloc_b(MEGA_MAX_EXPERTS);
}

}  // namespace tona::gpu
