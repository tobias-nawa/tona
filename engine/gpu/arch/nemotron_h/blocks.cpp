// Bolt's two mixers: Mamba-2 (conv, selective scan with the replay commit, gated group norm between two matvec phases)
// and full-context attention without gate, QK norm or RoPE. The latent MoE block is moe.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "gpu/arch/nemotron_h/kernel_set.h"
#include "gpu/arch/nemotron_h/program.h"
#include "gpu/arch/nemotron_h/state.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

// The per-tensor scale of an NVFP4 projection (F32, one value); 1 for the other types. The ops apply it themselves (in
// the conv, the scan and the norm), the matvec does not: it must not be applied a second time anywhere.
float tensor_scale(const TensorInfo* t) {
    if (!t) return 1.0f;
    if (t->type != 0 || t->elements() != 1 || !t->data)
        throw std::runtime_error("gpu::Mega: " + t->name + " is not a single F32 value");
    return *reinterpret_cast<const float*>(t->data);
}

}  // namespace

// in-projection [z | x B C | dt], conv, scan, gate and group norm, out-projection into y
void NemotronHProgram::mamba(const NemotronLayer& L, PfMamba2& P) {
    const Dims& s = dm_;
    const Mamba2Weights& W = L.ssm;
    P = NemotronHState::alloc_layer(g_, h_.mamba2());
    const void *w_in = g_.upload(*W.in_proj), *w_out = g_.upload(*W.out_proj);
    P.in_proj = pb_.pw(*W.in_proj, w_in), P.out_proj = pb_.pw(*W.out_proj, w_out);
    P.in_scale = tensor_scale(W.in_scale), P.out_scale = tensor_scale(W.out_scale);
    P.conv_w = g_.upload_f32(*W.conv1d), P.conv_b = g_.upload_f32(*W.conv1d_bias);
    P.dt_bias = g_.upload_f32(*W.dt_bias), P.A = g_.upload_f32(*W.a), P.D = g_.upload_f32(*W.dskip);
    P.norm = g_.upload_f32(*W.norm);
    pb_.mv_begin();
    pb_.addw(*W.in_proj, w_in, b_.xn, &b_.qxn, b_.t_zx);
    pb_.mv_end("ssm.in");
    pb_.op(OP_MAMBA_CONV, {b_.t_zx, P.ring, (void*)P.conv_w, (void*)P.conv_b, b_.t_conv},
           {s.conv_dim, s.conv, s.in_dim, s.inner}, {P.in_scale});
    pb_.op(OP_MAMBA_SCAN,
           {b_.t_zx, b_.t_conv, (void*)P.dt_bias, (void*)P.A, (void*)P.D, P.ssm, P.pdec, P.pxdt, P.pb, b_.t_y},
           {s.heads, s.inner / s.heads, 0, s.groups, s.in_dim, s.in_dim - s.heads, s.conv_dim, s.inner}, {P.in_scale});
    pb_.op(OP_MAMBA_NORM, {b_.t_y, b_.t_zx, (void*)P.norm, b_.q_o.q, b_.q_o.meta, b_.t_o},
           {s.inner, s.groups, s.in_dim}, {s.eps, P.in_scale, P.out_scale});
    pb_.mv_begin();
    pb_.addw(*W.out_proj, w_out, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("ssm.out");
}

// attn_q holds q only (no gate); K and V go into the int8 cache of the block, which the steps attend over in full. No
// norm of q or k and no RoPE: the preparation step only moves the values.
void NemotronHProgram::attention(const NemotronLayer& L, PfGatedAttn& A) {
    const Dims& s = dm_;
    const GatedAttentionWeights& W = L.attn;
    const KvBuf kvb = pb_.kv_buffer(g_.kv_split_.planned() ? g_.kv_split_.cells_in_vram(0, kv_main_, g_.max_ctx_)
                                                           : (g_.kv_host_ ? 0 : g_.max_ctx_));
    uint8_t* kv = kvb.kv;
    A.kv = kv, A.kv_cold = kvb.cold;
    const void *w_q = g_.upload(*W.q), *w_k = g_.upload(*W.k), *w_v = g_.upload(*W.v), *w_o = g_.upload(*W.o);
    A.q = pb_.pw(*W.q, w_q), A.k = pb_.pw(*W.k, w_k), A.v = pb_.pw(*W.v, w_v), A.o = pb_.pw(*W.o, w_o);
    pb_.mv_begin();
    pb_.addw(*W.q, w_q, b_.xn, &b_.qxn, b_.t_q);
    pb_.addw(*W.k, w_k, b_.xn, &b_.qxn, b_.t_k);
    pb_.addw(*W.v, w_v, b_.xn, &b_.qxn, b_.t_v);
    pb_.mv_end("attn.in");
    pb_.op(OP_FATTN_PREP,
           {b_.t_q, b_.t_qn, nullptr, nullptr, b_.t_k, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, kv, b_.t_v,
            kvb.cold},
           {s.nh, s.nkv, 0, 0, 0, 0, (int)g_.kv_row_, (int)std::min<int64_t>(kvb.hot, INT32_MAX)}, {s.eps, 0.0f});
    pb_.op(OP_FATTN_SPLIT, {b_.t_qn, kv, b_.part, kvb.cold},
           {s.nh, s.nkv, 0, 0, (int)g_.kv_row_, (int)std::min<int64_t>(kvb.hot, INT32_MAX)},
           {1.0f / std::sqrt((float)s.hd)});
    pb_.op(OP_FATTN_MERGE, {nullptr, nullptr, b_.part, nullptr, nullptr, b_.t_o, b_.q_o.q, b_.q_o.meta},
           {s.nh, 0, 0, s.ost});
    pb_.mv_begin();
    pb_.addw(*W.o, w_o, b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("attn.out");
}

}  // namespace tona::gpu
