// Buffers of the Coder's blocks: residual streams, DeltaNet and QSA steps, MoE and PLE, logits.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "backend/env.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

Qwen4ExpProgram::Dims Qwen4ExpProgram::dims_of(const Hparams& h) {
    Dims s{};
    s.hc = (int)h.n_hc, s.d = (int)h.d, s.ff = (int)h.expert_ff, s.sff = (int)h.shexp_ff, s.ne = (int)h.n_expert_used;
    s.hd_all = (int64_t)s.hc * s.d;
    s.kd = (int)h.dn_k_dim(), s.vd = (int)h.dn_v_dim(), s.nk = (int)h.dn_k_heads, s.nv = (int)h.dn_v_heads;
    s.nh = (int)h.n_head, s.nkv = (int)h.n_head_kv, s.hd = (int)h.head_dim, s.idim = (int)h.idx_dim;
    s.blk = (int)h.idx_block;
    s.eps = h.rms_eps, s.base = h.rope_base;
    s.ch = 2 * s.kd + s.vd;
    s.ost = std::max(s.vd, s.nh * s.hd);  // output of attention or DeltaNet per token
    return s;
}

void Qwen4ExpProgram::alloc_buffers() {
    alloc_streams();
    alloc_step_buffers();
    setup_kv();
    alloc_attention_buffers();
    alloc_moe_buffers();
    router_dev_.assign(h_.n_layer, nullptr);
}

void Qwen4ExpProgram::alloc_streams() {
    b_.R[0] = g_.alloc_b(dm_.hd_all);
    b_.R[1] = g_.alloc_b(dm_.hd_all);
    b_.xn = g_.alloc_b(dm_.hd_all);
    b_.lo = g_.alloc_b(h_.hc_rank);
    b_.gate = g_.alloc_b(dm_.hd_all);
    b_.mixed = g_.alloc_b(dm_.d);
    b_.qmix = g_.alloc_q8_b(dm_.d);
    if (hc8_) {
        b_.qxn = g_.alloc_q8_b(dm_.hd_all);
        b_.qlo = g_.alloc_q8_b(h_.hc_rank);
    }
    b_.inject_a = g_.alloc_b(32);
    b_.inject_f = g_.alloc_b(32);
    b_.y = g_.alloc_b(dm_.d);
}

void Qwen4ExpProgram::alloc_step_buffers() {
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
    b_.t_idxq = g_.alloc_b(h_.idx_heads * s.idim);
    b_.t_k = g_.alloc_b(s.nkv * s.hd);
    b_.t_v = g_.alloc_b(s.nkv * s.hd);
    b_.t_rk = g_.alloc_b(s.idim);
}

// KV cache in int8: bytes per position and layer; from 0.4 GiB in total (ctx about 31K) in pinned RAM (TONA_KV_HOST=0/1
// forces). In RAM the cache leaves 10-21 more expert slots: at ctx 32768 (0.42 GiB) the agent set gains +1.6 % (code)
// and +2.0 % (text), which outweighs the -1.4 % prefill of a cold 15K-token prompt; at ctx 16384 RAM gains nothing
// (research/measurements/2026-11-02-kv-staging.md, 2026-11-02-kv-threshold.md)
void Qwen4ExpProgram::setup_kv() {
    const Dims& s = dm_;
    g_.kv_row_ = (int64_t)s.nkv * 2 * s.hd + (int64_t)s.nkv * 2 * (s.hd / 32) * (int64_t)sizeof(float);
    int n_qsa = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) n_qsa += h_.is_qsa(l);
    g_.kv_bytes_ = (size_t)n_qsa * g_.max_ctx_ * g_.kv_row_;
    const int64_t force = env::integer("TONA_KV_HOST", -1);
    g_.kv_host_ = force >= 0 ? force == 1 : g_.kv_bytes_ > (size_t(2) << 30) / 5;
    g_.sstride_ = g_.max_ctx_ / s.blk + 1;
}

void Qwen4ExpProgram::alloc_attention_buffers() {
    const Dims& s = dm_;
    b_.scores = g_.alloc_b(g_.sstride_);
    b_.cells = reinterpret_cast<int32_t*>(g_.alloc_b(MAX_CELLS));
    // TONA_ATTN=old: old QSA attention, one block per (token, head); TONA_ATTN=wave: split-K with a wave per head;
    // otherwise split-K on WMMA, one block per KV group
    const char* mode = env::text("TONA_ATTN");
    attn_old_ = mode && std::string_view(mode) == "old";
    attn_wave_ = mode && std::string_view(mode) == "wave";
    if (!attn_old_ && s.hd != 256) throw std::runtime_error("gpu::Mega: split attention expects head dim 256");
    if (attn_old_) return;
    const size_t floats = (size_t)MEGA_BMAX * s.nh * ATT_SPL * (s.hd + 2);
    b_.attn_part = g_.alloc(floats);
    g_.rows_[reinterpret_cast<uintptr_t>(b_.attn_part)] = {
        floats * sizeof(float), (int64_t)s.nh * ATT_SPL * (s.hd + 2) * (int64_t)sizeof(float)};
}

// MoE: UMAX places per token for the union, the shared expert behind them; the PLE vectors; logits
void Qwen4ExpProgram::alloc_moe_buffers() {
    const Dims& s = dm_;
    b_.act_n = UMAX * s.ff + s.sff;
    b_.eg = g_.alloc_b(b_.act_n);
    b_.eu = g_.alloc_b(b_.act_n);
    b_.act = g_.alloc_b(b_.act_n);
    b_.q_act = g_.alloc_q8_b(b_.act_n);
    b_.ed = g_.alloc_b((size_t)UMAX * s.d);
    b_.sh_d = g_.alloc_b(s.d);
    b_.sh_inp = g_.alloc_b(32);
    b_.ple_k = g_.alloc_b(s.hd_all);
    b_.ple_v = g_.alloc_b(s.d);
    b_.ple_g = g_.alloc_b(s.hd_all);
    b_.ple_n = g_.alloc_b(s.hd_all);
    b_.ple_hist = g_.alloc_state((size_t)((h_.ple_conv - 1) * h_.ple_ngram + MEGA_BMAX) * s.hd_all);
    g_.e_ = g_.alloc_b(2 * s.d);
    g_.logits_ = g_.alloc_b(h_.vocab);
    g_.logits_host_.resize((size_t)MEGA_BMAX * h_.vocab);
    g_.logits1_.resize(h_.vocab);
    b_.router_out = g_.alloc_b(expert_cap(g_.h_.n_expert));
    b_.pred_out = g_.alloc_b(expert_cap(g_.h_.n_expert));
}

}  // namespace tona::gpu
