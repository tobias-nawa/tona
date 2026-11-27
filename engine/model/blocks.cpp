#include "model/blocks.h"

namespace tona {

DeltaNetWeights bind_deltanet_weights(Binder& b, const std::string& s, const DeltaNetDims& D) {
    const uint64_t kd = D.k_dim(), vd = D.v_dim(), nv = D.v_heads;
    DeltaNetWeights w;
    w.qkv = b.need(s + "attn_qkv.weight", {D.d, 2 * kd + vd});
    w.gate = b.need(s + "attn_gate.weight", {D.d, vd});
    w.beta = b.need(s + "ssm_beta.weight", {D.d, nv});
    w.alpha = b.need(s + "ssm_alpha.weight", {D.d, nv});
    w.conv1d = b.need(s + "ssm_conv1d.weight", {D.conv, 2 * kd + vd});
    w.dt_bias = b.need(s + "ssm_dt.bias", {nv});
    w.a = b.need(s + "ssm_a", {nv});
    w.ssm_norm = b.need(s + "ssm_norm.weight", {D.head_dim});
    w.ssm_out = b.need(s + "ssm_out.weight", {vd, D.d});
    return w;
}

GatedAttentionWeights bind_attention_weights(Binder& b, const std::string& s, const AttentionDims& D) {
    const uint64_t hd = D.head_dim;
    GatedAttentionWeights w;
    // With a gate, attn_q holds per head [q | gate] interleaved
    w.q = b.need(s + "attn_q.weight", {D.d, (D.gate ? 2 : 1) * D.n_head * hd});
    w.k = b.need(s + "attn_k.weight", {D.d, D.n_head_kv * hd});
    w.v = b.need(s + "attn_v.weight", {D.d, D.n_head_kv * hd});
    w.o = b.need(s + "attn_output.weight", {D.n_head * hd, D.d});
    if (D.qk_norm) {
        w.q_norm = b.need(s + "attn_q_norm.weight", {hd});
        w.k_norm = b.need(s + "attn_k_norm.weight", {hd});
    }
    if (D.idx_heads) {
        w.idx_q = b.need(s + "indexer.q_proj.weight", {D.d, D.idx_heads * D.idx_dim});
        w.idx_k = b.need(s + "indexer.k_proj.weight", {D.d, D.idx_dim});
        w.idx_q_norm = b.need(s + "indexer.q_norm.weight", {D.idx_dim});
        w.idx_k_norm = b.need(s + "indexer.k_norm.weight", {D.idx_dim});
    }
    return w;
}

MoeWeights bind_moe_weights(Binder& b, const std::string& s, const MoeDims& D) {
    const uint64_t in = D.latent ? D.latent : D.d;  // width the experts work on
    MoeWeights w;
    w.router = b.need(s + "ffn_gate_inp.weight", {D.d, D.n_expert});
    if (D.router_bias) w.router_bias = b.need(s + "exp_probs_b.bias", {D.n_expert});
    if (D.gated) w.gate_exps = b.need(s + "ffn_gate_exps.weight", {in, D.ff, D.n_expert});
    w.up_exps = b.need(s + "ffn_up_exps.weight", {in, D.ff, D.n_expert});
    w.down_exps = b.need(s + "ffn_down_exps.weight", {D.ff, in, D.n_expert});
    if (D.latent) {
        w.latent_down = b.need(s + "ffn_latent_down.weight", {D.d, D.latent});
        w.latent_up = b.need(s + "ffn_latent_up.weight", {D.latent, D.d});
    }
    if (D.shared_gate) w.shexp_gate_inp = b.need(s + "ffn_gate_inp_shexp.weight", {D.d});
    if (D.shared_ff) {
        if (D.gated) w.shexp_gate = b.need(s + "ffn_gate_shexp.weight", {D.d, D.shared_ff});
        w.shexp_up = b.need(s + "ffn_up_shexp.weight", {D.d, D.shared_ff});
        w.shexp_down = b.need(s + "ffn_down_shexp.weight", {D.shared_ff, D.d});
    }
    return w;
}

Mamba2Weights bind_mamba2_weights(Binder& b, const std::string& s, const Mamba2Dims& D) {
    Mamba2Weights w;
    w.in_proj = b.need(s + "ssm_in.weight", {D.d, D.in_proj_dim()});
    w.in_scale = b.optional(s + "ssm_in.scale", {1});
    w.out_proj = b.need(s + "ssm_out.weight", {D.inner, D.d});
    w.out_scale = b.optional(s + "ssm_out.scale", {1});
    w.conv1d = b.need(s + "ssm_conv1d.weight", {D.conv, D.conv_dim()});
    w.conv1d_bias = b.need(s + "ssm_conv1d.bias", {D.conv_dim()});
    w.dt_bias = b.need(s + "ssm_dt.bias", {D.heads});
    w.a = b.need(s + "ssm_a", {1, D.heads});
    w.dskip = b.need(s + "ssm_d", {1, D.heads});
    w.norm = b.need(s + "ssm_norm.weight", {D.inner / D.groups, D.groups});
    return w;
}

}  // namespace tona
