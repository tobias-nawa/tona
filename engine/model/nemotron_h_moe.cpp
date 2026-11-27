#include "model/nemotron_h_moe.h"

#include <algorithm>
#include <format>
#include <variant>

namespace tona {

namespace {

const char* const ARCH_NAME = "nemotron_h_moe";
std::string prefix() { return std::string(ARCH_NAME) + "."; }

// The per-layer arrays of the GGUF; every one has one value per block
std::vector<uint32_t> layer_array(const Binder& b, const std::string& key, uint32_t n_layer) {
    const auto v = b.gguf().get_int_array(prefix() + key);
    b.check(v.size() == n_layer, std::format("{} has {} values for {} blocks", key, v.size(), n_layer));
    std::vector<uint32_t> out;
    for (int64_t x : v) {
        b.check(x >= 0 && x <= UINT32_MAX, std::format("{} holds {}", key, x));
        out.push_back((uint32_t)x);
    }
    return out;
}

void read_scalars(const Binder& b, NemotronHparams& h) {
    const std::string p = prefix();
    h.n_layer = b.u32(p + "block_count");
    h.d = b.u32(p + "embedding_length");
    h.rms_eps = (float)b.gguf().get_float(p + "attention.layer_norm_rms_epsilon");
    h.native_context = b.gguf().has(p + "context_length") ? b.u32(p + "context_length") : 0;
    h.n_head = b.u32(p + "attention.head_count");
    h.head_dim = b.u32(p + "attention.key_length");
    h.ssm_inner = b.u32(p + "ssm.inner_size");
    h.ssm_heads = b.u32(p + "ssm.time_step_rank");
    h.ssm_state = b.u32(p + "ssm.state_size");
    h.ssm_groups = b.u32(p + "ssm.group_count");
    h.ssm_conv = b.u32(p + "ssm.conv_kernel");
    h.latent = b.gguf().has(p + "moe_latent_size") ? b.u32(p + "moe_latent_size") : 0;
    h.shexp_ff = b.u32(p + "expert_shared_feed_forward_length");
    h.n_expert_max = b.u32(p + "expert_count");
    h.expert_scale = (float)b.gguf().get_float(p + "expert_weights_scale");
    h.n_head_kv = layer_array(b, "attention.head_count_kv", h.n_layer);
    h.ff = layer_array(b, "feed_forward_length", h.n_layer);
    h.expert_ff = layer_array(b, "expert_feed_forward_length", h.n_layer);
    h.expert_used = layer_array(b, "expert_used_count", h.n_layer);
}

// Assumptions of the engine that do not follow from single keys: the routing rule and the shapes the oracle and the
// kernels are written for
void check_assumptions(const Binder& b, const NemotronHparams& h) {
    const std::string p = prefix();
    // 2: sigmoid (llama.cpp LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID); the softmax router is another policy
    b.check(b.u32(p + "expert_gating_func") == 2, "expert_gating_func is not 2 (sigmoid)");
    b.check(std::get<bool>(b.gguf().kv().at(p + "expert_weights_norm").v), "expert_weights_norm is false");
    b.check(!b.gguf().has(p + "expert_group_count") || b.u32(p + "expert_group_count") == 1,
            "grouped expert selection is not supported");
    b.check(b.u32(p + "expert_shared_count") == 1, "expert_shared_count is not 1");
    b.check(b.u32(p + "attention.value_length") == h.head_dim, "value_length != key_length");
    b.check(h.ssm_heads > 0 && h.ssm_inner % h.ssm_heads == 0, "ssm.inner_size is not a multiple of the heads");
    b.check(h.ssm_groups > 0 && h.ssm_heads % h.ssm_groups == 0 && h.ssm_inner % h.ssm_groups == 0,
            "Mamba-2: heads or inner size are not a multiple of the groups");
    b.check(h.ssm_conv >= 2, "ssm.conv_kernel is below 2");
    b.check(h.n_head > 0 && h.head_dim > 0, "attention heads or head size are 0");
    b.check(h.n_expert_max > 0, "expert_count is 0");
}

// What the arrays say a block is: attention has KV heads, a MoE has a feed-forward width, Mamba-2 has neither
BoltBlockKind kind_of(const Binder& b, const NemotronHparams& h, uint32_t l) {
    const bool kv = h.n_head_kv[l] > 0, ff = h.ff[l] > 0;
    b.check(!(kv && ff), std::format("block {} has KV heads and a feed-forward width", l));
    return kv ? BoltBlockKind::Attention : ff ? BoltBlockKind::Moe : BoltBlockKind::Mamba2;
}

const char* kind_tensor(BoltBlockKind k) {
    return k == BoltBlockKind::Mamba2      ? "ssm_in.weight"
           : k == BoltBlockKind::Attention ? "attn_q.weight"
                                           : "ffn_gate_inp.weight";
}

NemotronLayer bind_moe_layer(Binder& b, const NemotronHparams& h, const std::string& s, uint32_t l, NemotronLayer L) {
    const TensorInfo* router = b.gguf().find(s + "ffn_gate_inp.weight");
    b.check(router && router->dims.size() == 2, std::format("block {}: ffn_gate_inp.weight is missing", l));
    L.n_expert = (uint32_t)router->dims[1];
    L.n_used = h.expert_used[l];
    L.expert_ff = h.expert_ff[l];
    b.check(L.n_used > 0 && L.n_used <= L.n_expert && L.n_expert <= h.n_expert_max,
            std::format("block {}: {} experts, top {}, at most {}", l, L.n_expert, L.n_used, h.n_expert_max));
    b.check(L.expert_ff > 0, std::format("block {}: expert_feed_forward_length is 0", l));
    L.moe = bind_moe_weights(b, s, {h.d, L.n_expert, L.expert_ff, h.shexp_ff, false, false, h.latent, true});
    return L;
}

NemotronLayer bind_layer(Binder& b, const NemotronHparams& h, uint32_t l) {
    const std::string s = std::format("blk.{}.", l);
    NemotronLayer L;
    L.kind = kind_of(b, h, l);
    L.attn_norm = b.need(s + "attn_norm.weight", {h.d});
    // The tensors present must agree with the arrays, so a changed layer pattern stops the load, not the first token
    for (BoltBlockKind k : {BoltBlockKind::Mamba2, BoltBlockKind::Attention, BoltBlockKind::Moe})
        b.check((b.gguf().find(s + kind_tensor(k)) != nullptr) == (k == L.kind),
                std::format("block {}: the arrays say {} but the tensors say otherwise", l, kind_tensor(L.kind)));
    switch (L.kind) {
        case BoltBlockKind::Mamba2:
            L.ssm = bind_mamba2_weights(b, s, h.mamba2());
            break;
        case BoltBlockKind::Attention:
            L.n_kv = h.n_head_kv[l];
            b.check(h.n_head % L.n_kv == 0, std::format("block {}: Q heads are not a multiple of KV heads", l));
            L.attn = bind_attention_weights(b, s, {h.d, h.n_head, L.n_kv, h.head_dim, false, false, 0, 0});
            break;
        case BoltBlockKind::Moe:
            L = bind_moe_layer(b, h, s, l, L);
            break;
    }
    return L;
}

}  // namespace

NemotronHMoeModel::NemotronHMoeModel(const std::string& path) : ModelBase(ARCH, path) {
    const Gguf& g = gguf();
    Binder b(g, ARCH_NAME);
    b.check(detect_arch(g) == ARCH,
            std::format("architecture {} instead of {}", g.get_string("general.architecture"), ARCH_NAME));
    NemotronHparams& h = hp_;
    read_scalars(b, h);
    check_assumptions(b, h);

    tok_embd_ = g.find("token_embd.weight");
    b.check(tok_embd_ && tok_embd_->dims.size() == 2 && tok_embd_->dims[0] == h.d,
            "token_embd is missing or has the wrong shape");
    h.vocab = (uint32_t)tok_embd_->dims[1];
    b.check(!g.has(prefix() + "vocab_size") || b.u32(prefix() + "vocab_size") == h.vocab,
            "vocab_size differs from the rows of token_embd");
    b.mark_used("token_embd.weight");
    output_norm_ = b.need("output_norm.weight", {h.d});
    output_ = b.need("output.weight", {h.d, h.vocab});

    for (uint32_t l = 0; l < h.n_layer; l++) {
        layers_.push_back(bind_layer(b, h, l));
        ordinal_.push_back((int)moe_index_.size());
        if (layers_.back().kind == BoltBlockKind::Moe) moe_index_.push_back(l);
    }
    b.check(!moe_index_.empty(), "no MoE block");

    // Nothing may be left over: an unknown tensor means the description is incomplete
    b.expect_all_used();
    set_dims({h.vocab, h.d, h.n_layer, h.native_context, 0.0f, h.rms_eps});  // no RoPE: the attention has no positions
}

uint32_t NemotronHMoeModel::count(BoltBlockKind k) const {
    return (uint32_t)std::count_if(layers_.begin(), layers_.end(), [k](const NemotronLayer& L) { return L.kind == k; });
}

WeightPlan NemotronHMoeModel::plan() const {
    WeightPlan w;
    for (const auto& t : gguf().tensors()) {
        if (t.name == "token_embd.weight")
            w.embedding += t.bytes;
        else if (t.name.contains("_exps."))
            w.experts += t.bytes;
        else
            w.dense += t.bytes;
    }
    w.dense_lm_head = output_->bytes;
    uint64_t n_experts = 0;
    for (const NemotronLayer& L : layers_) {
        if (L.kind != BoltBlockKind::Moe) continue;
        n_experts += L.n_expert;
        // Per layer: the bytes of its experts divided by their count, times the experts one token uses
        const uint64_t layer_bytes = L.moe.up_exps->bytes + L.moe.down_exps->bytes;
        w.experts_per_token += layer_bytes / L.n_expert * L.n_used;
    }
    // Mean over all experts of all layers. The constructor rejects a model without a MoE block, so the max never
    // changes the value; it shows the analyzer that the divisor is not zero.
    w.expert_bytes = w.experts / std::max<uint64_t>(n_experts, 1);
    return w;
}

MoeShape NemotronHMoeModel::moe_shape() const {
    uint32_t ff = 0;
    for (const NemotronLayer& L : layers_) ff = std::max(ff, L.expert_ff);
    uint32_t used = 0;
    for (const NemotronLayer& L : layers_) used = std::max(used, L.n_used);
    return {(uint32_t)moe_index_.size(), hp_.d, hp_.vocab, hp_.n_expert_max, used, ff, hp_.shexp_ff, 0.0f};
}

// Nothing besides the tensors: the attention has no positions (no RoPE, so no YaRN table), and the routing constants
// come from the file
void NemotronHMoeModel::fingerprint(Hasher&) const {}

std::string NemotronHMoeModel::describe() const {
    return std::format(
        "nemotron_h_moe: {} blocks ({} Mamba-2, {} attention, {} MoE), d {}, up to {} experts (top {}), latent {}, "
        "vocab {}",
        hp_.n_layer, count(BoltBlockKind::Mamba2), count(BoltBlockKind::Attention), count(BoltBlockKind::Moe), hp_.d,
        hp_.n_expert_max, moe_shape().n_expert_used, hp_.latent, hp_.vocab);
}

}  // namespace tona
