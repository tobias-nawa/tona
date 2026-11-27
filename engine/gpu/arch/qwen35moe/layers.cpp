// Ornith's two mixers: gated DeltaNet (silu readout gate) and full-context gated attention.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "gpu/arch/qwen35moe/kernel_set.h"
#include "gpu/arch/qwen35moe/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void Qwen35MoeProgram::layer_deltanet(const Qwen35Layer& S, PfLayer& F) {
    const Dims& s = dm_;
    const DeltaNetWeights& W = S.dn;
    float* state = g_.alloc_state((size_t)s.nv * DS * DS);
    float* ring = g_.alloc_state((size_t)(h_.dn_conv - 1 + MEGA_BMAX) * s.ch);
    float* pend_d = g_.alloc_state((size_t)2 * MEGA_BMAX * s.vd);
    float* pend_a = g_.alloc_state((size_t)2 * MEGA_BMAX * s.nv);
    float* pend_k = g_.alloc_state((size_t)2 * MEGA_BMAX * s.kd);
    F.dn.state = state, F.dn.ring = ring, F.dn.pend_d = pend_d, F.dn.pend_a = pend_a, F.dn.pend_k = pend_k;
    const void *w_qkv = g_.upload(*W.qkv), *w_gate = g_.upload(*W.gate), *w_beta = g_.upload(*W.beta),
               *w_alpha = g_.upload(*W.alpha);
    F.dn.qkv = pb_.pw(*W.qkv, w_qkv), F.dn.gate = pb_.pw(*W.gate, w_gate), F.dn.beta = pb_.pw(*W.beta, w_beta);
    F.dn.alpha = pb_.pw(*W.alpha, w_alpha);
    pb_.mv_begin();
    pb_.addw(*W.qkv, w_qkv, b_.xn, &b_.qxn, b_.t_qkv);
    pb_.addw(*W.gate, w_gate, b_.xn, &b_.qxn, b_.t_z);
    pb_.addw(*W.beta, w_beta, b_.xn, &b_.qxn, b_.t_b);
    pb_.addw(*W.alpha, w_alpha, b_.xn, &b_.qxn, b_.t_a);
    pb_.mv_end("dn.in");
    const float *conv_w = g_.upload_f32(*W.conv1d), *dt_bias = g_.upload_f32(*W.dt_bias), *A = g_.upload_f32(*W.a);
    const float* ssm_norm = g_.upload_f32(*W.ssm_norm);
    F.dn.conv_w = conv_w, F.dn.dt_bias = dt_bias, F.dn.A = A, F.dn.ssm_norm = ssm_norm;
    pb_.op(OP_DN_CONV, {b_.t_qkv, ring, (void*)conv_w, b_.t_conv}, {s.kd, s.vd, (int)h_.dn_conv, DS, s.nk}, {s.eps});
    pb_.op(OP_DN_RECUR, {b_.t_conv, b_.t_b, b_.t_a, (void*)dt_bias, (void*)A, pend_d, pend_a, state, b_.t_raw, pend_k},
           {s.nk, s.nv, DS, s.kd}, {s.eps});
    pb_.op(OP_DN_NORM, {b_.t_raw, (void*)ssm_norm, b_.t_z, b_.t_o, b_.q_o.q, b_.q_o.meta}, {s.nv, s.ost}, {s.eps});
    const void* w_out = g_.upload(*W.ssm_out);
    F.dn.ssm_out = pb_.pw(*W.ssm_out, w_out);
    pb_.mv_begin();
    pb_.addw(*W.ssm_out, w_out, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("dn.out");
}

// attn_q holds [q, gate] per head; K and V go into the int8 cache of the layer, which the steps attend over in full
void Qwen35MoeProgram::layer_attention(const Qwen35Layer& S, PfLayer& F) {
    const Dims& s = dm_;
    const GatedAttentionWeights& W = S.attn;
    const KvBuf kvb = pb_.kv_buffer(kv_cells_in_vram(kv_allocated_++));
    uint8_t* kv = kvb.kv;
    F.attn.kv = kv, F.attn.kv_cold = kvb.cold;
    const void *w_q = g_.upload(*W.q), *w_k = g_.upload(*W.k), *w_v = g_.upload(*W.v);
    F.attn.q = pb_.pw(*W.q, w_q), F.attn.k = pb_.pw(*W.k, w_k), F.attn.v = pb_.pw(*W.v, w_v);
    pb_.mv_begin();
    pb_.addw(*W.q, w_q, b_.xn, &b_.qxn, b_.t_qfull);
    pb_.addw(*W.k, w_k, b_.xn, &b_.qxn, b_.t_k);
    pb_.addw(*W.v, w_v, b_.xn, &b_.qxn, b_.t_v);
    pb_.mv_end("attn.in");
    const float *q_norm = g_.upload_f32(*W.q_norm), *k_norm = g_.upload_f32(*W.k_norm);
    F.attn.q_norm = q_norm, F.attn.k_norm = k_norm;
    pb_.op(OP_FATTN_PREP,
           {b_.t_qfull, b_.t_qn, b_.t_g, (void*)q_norm, b_.t_k, (void*)k_norm, nullptr, nullptr, nullptr, nullptr,
            nullptr, kv, b_.t_v, kvb.cold},
           {s.nh, s.nkv, 0, 0, 0, 0, (int)g_.kv_row_, (int)std::min<int64_t>(kvb.hot, INT32_MAX)}, {s.eps, s.base});
    pb_.op(OP_FATTN_SPLIT, {b_.t_qn, kv, b_.part, kvb.cold},
           {s.nh, s.nkv, 0, 0, (int)g_.kv_row_, (int)std::min<int64_t>(kvb.hot, INT32_MAX)},
           {1.0f / std::sqrt((float)s.hd)});
    pb_.op(OP_FATTN_MERGE, {nullptr, nullptr, b_.part, nullptr, b_.t_g, b_.t_o, b_.q_o.q, b_.q_o.meta},
           {s.nh, 0, 0, s.ost});
    const void* w_o = g_.upload(*W.o);
    F.attn.o = pb_.pw(*W.o, w_o);
    pb_.mv_begin();
    pb_.addw(*W.o, w_o, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("attn.out");
}

}  // namespace tona::gpu
