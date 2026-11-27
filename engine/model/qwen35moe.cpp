#include "model/qwen35moe.h"

#include <format>

#include "backend/rope_yarn.h"

namespace tona {

namespace {

const char* const ARCH_NAME = "qwen35moe";
std::string prefix() { return std::string(ARCH_NAME) + "."; }

void read_scalars(const Binder& b, Qwen35Hparams& h) {
    const std::string p = prefix();
    const uint32_t blocks = b.u32(p + "block_count");
    h.n_mtp = b.gguf().has(p + "nextn_predict_layers") ? b.u32(p + "nextn_predict_layers") : 0;
    b.check(blocks > h.n_mtp, "block_count is not larger than nextn_predict_layers");
    h.n_layer = blocks - h.n_mtp;
    h.d = b.u32(p + "embedding_length");
    h.rms_eps = (float)b.gguf().get_float(p + "attention.layer_norm_rms_epsilon");
    h.full_attn_interval = b.u32(p + "full_attention_interval");
    h.dn_k_heads = b.u32(p + "ssm.group_count");
    h.dn_v_heads = b.u32(p + "ssm.time_step_rank");
    h.dn_head_dim = b.u32(p + "ssm.state_size");
    h.dn_conv = b.u32(p + "ssm.conv_kernel");
    h.n_head = b.u32(p + "attention.head_count");
    h.n_head_kv = b.u32(p + "attention.head_count_kv");
    h.head_dim = b.u32(p + "attention.key_length");
    h.rope_dims = b.u32(p + "rope.dimension_count");
    h.rope_base = (float)b.gguf().get_float(p + "rope.freq_base");
    h.native_context = b.gguf().has(p + "context_length") ? b.u32(p + "context_length") : 0;
    h.n_expert = b.u32(p + "expert_count");
    h.n_expert_used = b.u32(p + "expert_used_count");
    h.expert_ff = b.u32(p + "expert_feed_forward_length");
    h.shexp_ff = b.u32(p + "expert_shared_feed_forward_length");
}

// Assumptions of the engine that do not follow from single keys
void check_assumptions(const Binder& b, Qwen35Hparams& h) {
    const std::string p = prefix();
    b.check(h.n_mtp <= 1, "more than one MTP layer is not supported");
    b.check(h.full_attn_interval > 0, "full_attention_interval is 0");
    b.check(b.u32(p + "attention.value_length") == h.head_dim, "value_length != key_length");
    b.check(h.dn_head_dim > 0 && b.u32(p + "ssm.inner_size") == (uint64_t)h.dn_v_heads * h.dn_head_dim,
            "ssm.inner_size != V heads x head dim");
    b.check(h.dn_k_heads > 0 && h.dn_v_heads % h.dn_k_heads == 0, "DeltaNet: V heads are not a multiple of K heads");
    b.check(h.n_head_kv > 0 && h.n_head % h.n_head_kv == 0, "attention: Q heads are not a multiple of KV heads");
    b.check(h.n_expert_used <= h.n_expert, "more active than available experts");
    // The oracle and the kernels rotate with one position for all sections: text only (no image or video positions)
    const auto sections = b.gguf().get_int_array(p + "rope.dimension_sections");
    b.check(sections.size() == h.rope_sections.size(), "rope.dimension_sections: expected 4 values");
    uint32_t pairs = 0;
    for (size_t i = 0; i < sections.size(); i++) pairs += h.rope_sections[i] = (uint32_t)sections[i];
    b.check(2 * pairs == h.rope_dims, "rope.dimension_sections do not add up to rope.dimension_count / 2");
    b.check(h.rope_dims <= h.head_dim && h.rope_dims % 2 == 0, "rope.dimension_count does not fit the head");
}

Qwen35Layer bind_layer(Binder& b, const Qwen35Hparams& h, uint32_t l) {
    const std::string s = std::format("blk.{}.", l);
    Qwen35Layer L;
    L.attn_norm = b.need(s + "attn_norm.weight", {h.d});
    L.post_attn_norm = b.need(s + "post_attention_norm.weight", {h.d});
    // The tensors present say what the layer is; the interval must agree, so a changed pattern stops the load
    const bool has_attn = b.gguf().find(s + "attn_q.weight") != nullptr;
    b.check(has_attn == h.is_attn(l), std::format("layer {} has {} tensors but full_attention_interval {} says {}", l,
                                                  has_attn ? "attention" : "DeltaNet", h.full_attn_interval,
                                                  h.is_attn(l) ? "attention" : "DeltaNet"));
    if (has_attn) {
        L.kind = Qwen35LayerKind::GatedAttn;
        L.attn = bind_attention_weights(b, s, h.attention());
    } else {
        L.dn = bind_deltanet_weights(b, s, h.deltanet());
    }
    L.moe = bind_moe_weights(b, s, h.moe());
    return L;
}

Qwen35Mtp bind_mtp(Binder& b, const Qwen35Hparams& h) {
    const uint32_t l = h.n_layer;
    const std::string s = std::format("blk.{}.", l);
    Qwen35Mtp m;
    m.layer.kind = Qwen35LayerKind::GatedAttn;
    m.layer.attn_norm = b.need(s + "attn_norm.weight", {h.d});
    m.layer.post_attn_norm = b.need(s + "post_attention_norm.weight", {h.d});
    m.layer.attn = bind_attention_weights(b, s, h.attention());
    m.layer.moe = bind_moe_weights(b, s, h.moe());
    m.eh_proj = b.need(s + "nextn.eh_proj.weight", {2 * (uint64_t)h.d, h.d});
    m.enorm = b.need(s + "nextn.enorm.weight", {h.d});
    m.hnorm = b.need(s + "nextn.hnorm.weight", {h.d});
    m.shared_head_norm = b.need(s + "nextn.shared_head_norm.weight", {h.d});
    return m;
}

}  // namespace

Qwen35MoeModel::Qwen35MoeModel(const std::string& path) : ModelBase(ARCH, path) {
    const Gguf& g = gguf();
    Binder b(g, ARCH_NAME);
    b.check(detect_arch(g) == ARCH,
            std::format("architecture {} instead of {}", g.get_string("general.architecture"), ARCH_NAME));
    Qwen35Hparams& h = hp_;
    read_scalars(b, h);
    check_assumptions(b, h);

    tok_embd_ = g.find("token_embd.weight");
    b.check(tok_embd_ && tok_embd_->dims.size() == 2 && tok_embd_->dims[0] == h.d,
            "token_embd is missing or has the wrong shape");
    h.vocab = (uint32_t)tok_embd_->dims[1];
    b.mark_used("token_embd.weight");
    output_norm_ = b.need("output_norm.weight", {h.d});
    output_ = b.need("output.weight", {h.d, h.vocab});

    for (uint32_t l = 0; l < h.n_layer; l++) layers_.push_back(bind_layer(b, h, l));
    if (h.n_mtp) mtp_ = bind_mtp(b, h);

    // Nothing may be left over: an unknown tensor means the description is incomplete
    b.expect_all_used();
    set_dims({h.vocab, h.d, h.n_layer, h.native_context, h.rope_base, h.rms_eps});
}

WeightPlan Qwen35MoeModel::plan() const {
    WeightPlan w;
    const std::string mtp_prefix = std::format("blk.{}.", hp_.n_layer);
    for (const auto& t : gguf().tensors()) {
        if (t.name == "token_embd.weight") {
            w.embedding += t.bytes;
        } else if (t.name.starts_with(mtp_prefix)) {
            w.mtp += t.bytes;  // read only when the MTP head runs
        } else if (t.name.contains("_exps.")) {
            w.experts += t.bytes;
        } else {
            w.dense += t.bytes;
        }
    }
    w.dense_lm_head = output_->bytes;
    w.expert_bytes = w.experts / ((uint64_t)hp_.n_layer * hp_.n_expert);
    w.experts_per_token = w.expert_bytes * hp_.n_expert_used * hp_.n_layer;
    return w;
}

MoeShape Qwen35MoeModel::moe_shape() const {
    MoeShape s{hp_.n_layer,       hp_.d,         hp_.vocab,    hp_.n_expert,
               hp_.n_expert_used, hp_.expert_ff, hp_.shexp_ff, hp_.rope_base};
    s.qsa_interval = hp_.full_attn_interval;  // the gated attention layers hold the KV cache, without an indexer pool
    return s;
}

// YaRN rotates the keys of the attention layers by other angles: a checkpoint of one mode is wrong in the other. Off
// adds nothing.
void Qwen35MoeModel::fingerprint(Hasher& h) const {
    if (const rope::YarnTable& y = rope::yarn_active(hp_.rope_base); y.factor > 1.0) {
        h.value(y.mul);
        h.value(y.mscale);
    }
}

std::string Qwen35MoeModel::describe() const {
    return std::format("qwen35moe: {} layers ({} gated attention) + {} MTP, d {}, {} experts (top {}), vocab {}",
                       hp_.n_layer, hp_.n_layer / hp_.full_attn_interval, hp_.n_mtp, hp_.d, hp_.n_expert,
                       hp_.n_expert_used, hp_.vocab);
}

}  // namespace tona
