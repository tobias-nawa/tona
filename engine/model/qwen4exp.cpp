#include "model/qwen4exp.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <set>
#include <stdexcept>

#include "backend/env.h"
#include "backend/rope_yarn.h"
#include "model/blocks.h"
#include "model/ple.h"

namespace tona {

namespace {

const char* const ARCH_NAME = "qwen4exp";
std::string prefix() { return std::string(ARCH_NAME) + "."; }

void read_scalars(const Binder& b, Hparams& h) {
    const std::string p = prefix();
    h.n_layer = b.u32(p + "block_count");
    h.d = b.u32(p + "embedding_length");
    h.rms_eps = (float)b.gguf().get_float(p + "attention.layer_norm_rms_epsilon");
    h.n_hc = b.u32(p + "hyper_connection.count");
    h.hc_rank = b.u32(p + "hyper_connection.low_rank");
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
    h.idx_heads = b.u32(p + "attention.indexer.head_count");
    h.idx_dim = b.u32(p + "attention.indexer.key_length");
    h.idx_top_k = b.u32(p + "attention.indexer.top_k");
    h.n_expert = b.u32(p + "expert_count");
    h.n_expert_used = b.u32(p + "expert_used_count");
    h.expert_ff = b.u32(p + "expert_feed_forward_length");
    h.shexp_ff = b.u32(p + "expert_shared_feed_forward_length");
    h.ple_dim = b.u32(p + "embedding_length_per_layer_input");
    h.ple_ngram = b.u32(p + "ple.ngram_size");
    h.ple_heads = b.u32(p + "ple.heads_per_ngram") * (h.ple_ngram - 1);
    h.ple_conv = b.u32(p + "ple.conv_kernel");
    h.ple_eos = b.u32(p + "ple.eos_token_id");
    h.ple_image_token = b.u32(p + "ple.image_token_id");
}

// Assumptions of the engine that do not follow from single keys
void check_assumptions(const Binder& b, const Hparams& h) {
    const std::string p = prefix();
    b.check(b.u32(p + "attention.value_length") == h.head_dim, "value_length != key_length");
    b.check(b.u32(p + "ssm.inner_size") == h.dn_v_dim(), "ssm.inner_size != V heads x head dim");
    b.check(h.dn_v_heads % h.dn_k_heads == 0, "DeltaNet: V heads are not a multiple of K heads");
    b.check(h.n_head % h.n_head_kv == 0, "QSA: Q heads are not a multiple of KV heads");
    b.check(h.idx_heads == 4 && h.idx_dim == 128,
            "QSA: the indexer kernels of the megakernel score 4 heads of 128 values");
    b.check(h.n_expert_used <= h.n_expert, "more active than available experts");
    b.check(h.ple_heads * h.ple_dim == h.d, "PLE: heads x row width != d");
}

void read_compress_ratios(const Binder& b, Hparams& h) {
    const auto ratios = b.gguf().get_int_array(prefix() + "attention.compress_ratios");
    b.check(ratios.size() == h.n_layer, "compress_ratios has the wrong length");
    h.idx_block = 0;
    for (uint32_t l = 0; l < h.n_layer; l++) {
        // QSA layers carry the indexer block size, DeltaNet layers 0
        b.check((ratios[l] != 0) == h.is_qsa(l),
                std::format("compress_ratios[{}] does not match the layer pattern", l));
        if (ratios[l]) {
            b.check(h.idx_block == 0 || h.idx_block == ratios[l], "inconsistent indexer block size");
            h.idx_block = (uint32_t)ratios[l];
        }
    }
}

// TONA_IDX_TOPK=<tokens>: how many tokens each query may attend to through the indexer (default: the model's value,
// 2048). Rounded down to whole blocks, at least one block. The kernels hold MEGA_MAX_CELLS cells per token; a value
// that does not fit stops the start in the backend (gpu/mega_init.cpp).
void apply_idx_topk_env(Hparams& h) {
    h.idx_top_k_gguf = h.idx_top_k;
    const int64_t want = env::integer("TONA_IDX_TOPK", 0);
    if (want <= 0) return;
    h.idx_top_k = (uint32_t)std::max<int64_t>(want / h.idx_block, 1) * h.idx_block;
    std::fprintf(stderr, "TONA_IDX_TOPK=%lld: indexer top-k %u tokens (%u blocks) instead of %u\n", (long long)want,
                 h.idx_top_k, h.idx_top_k / h.idx_block, h.idx_top_k_gguf);
}

void read_ple_tables(const Binder& b, Hparams& h) {
    const std::string p = prefix();
    const auto ple_layers = b.gguf().get_int_array(p + "ple.layers");
    b.check(ple_layers.size() == 1, "PLE in more than one layer is not supported");
    h.ple_layer = (uint32_t)ple_layers[0];
    const auto mult = b.gguf().get_int_array(p + "ple.layer_multipliers");
    b.check(mult.size() == h.ple_ngram, "ple.layer_multipliers: wrong length");
    b.check(h.ple_ngram >= 2 && h.ple_ngram <= h.ple_mult.size(), "ple.ngram_size: expected 2 to 3 (ple_mult)");
    for (size_t i = 0; i < 3; i++) h.ple_mult[i] = mult[i];
    for (auto v : b.gguf().get_int_array(p + "ple.head_vocab_sizes")) h.ple_head_vocab.push_back((uint64_t)v);
    for (auto v : b.gguf().get_int_array(p + "ple.head_offsets")) h.ple_head_offset.push_back((uint64_t)v);
    b.check(h.ple_head_vocab.size() == h.ple_heads && h.ple_head_offset.size() == h.ple_heads, "PLE head tables");
    for (uint32_t i = 1; i < h.ple_heads; i++)
        b.check(h.ple_head_offset[i] == h.ple_head_offset[i - 1] + h.ple_head_vocab[i - 1],
                "PLE offsets are not cumulative");
}

Hparams read_hparams(const Binder& b) {
    Hparams h;
    read_scalars(b, h);
    check_assumptions(b, h);
    read_compress_ratios(b, h);
    read_ple_tables(b, h);
    return h;
}

// Sizes shared by the tensor shapes of one layer
struct Dims {
    uint64_t d, hc, r, E, ff, sff;
};

Dims dims_of(const Hparams& h, uint64_t n_expert) {
    return {h.d, (uint64_t)h.n_hc * h.d, h.hc_rank, n_expert, h.expert_ff, h.shexp_ff};
}

GatedResidual bind_gr(Binder& b, const std::string& s, const Dims& D, uint32_t n_hc, const char* kind) {
    return GatedResidual{b.need(s + "hc_" + kind + "_norm.weight", {D.hc}),
                         b.need(s + "hc_" + kind + "_down.weight", {D.hc, D.r}),
                         b.need(s + "hc_" + kind + "_up.weight", {D.r, D.hc}),
                         b.need(s + "hc_" + kind + "_inject.weight", {D.hc, n_hc})};
}

// The shared block binders return their own weight structs; the Coder's flat Layer takes the pointers over
void bind_qsa(Binder& b, const std::string& s, const Hparams& h, Layer& L) {
    const AttentionDims A{h.d, h.n_head, h.n_head_kv, h.head_dim, true, true, h.idx_heads, h.idx_dim};
    const GatedAttentionWeights w = bind_attention_weights(b, s, A);
    L.q = w.q, L.k = w.k, L.v = w.v, L.o = w.o, L.q_norm = w.q_norm, L.k_norm = w.k_norm;
    L.idx_q = w.idx_q, L.idx_k = w.idx_k, L.idx_q_norm = w.idx_q_norm, L.idx_k_norm = w.idx_k_norm;
}

void bind_deltanet(Binder& b, const std::string& s, const Hparams& h, Layer& L) {
    const DeltaNetWeights w = bind_deltanet_weights(b, s, {h.d, h.dn_k_heads, h.dn_v_heads, h.dn_head_dim, h.dn_conv});
    L.qkv = w.qkv, L.gate = w.gate, L.beta = w.beta, L.alpha = w.alpha, L.conv1d = w.conv1d;
    L.dt_bias = w.dt_bias, L.a = w.a, L.ssm_norm = w.ssm_norm, L.ssm_out = w.ssm_out;
}

void bind_moe(Binder& b, const std::string& s, const Dims& D, Layer& L) {
    const MoeWeights w = bind_moe_weights(b, s, {D.d, D.E, D.ff, D.sff, true, true, 0, false});
    L.router = w.router, L.gate_exps = w.gate_exps, L.up_exps = w.up_exps, L.down_exps = w.down_exps;
    L.shexp_gate_inp = w.shexp_gate_inp, L.shexp_gate = w.shexp_gate, L.shexp_up = w.shexp_up;
    L.shexp_down = w.shexp_down;
}

void bind_ple(Binder& b, const std::string& s, const Hparams& h, const Dims& D, Layer& L) {
    L.ple_key = b.need(s + "ple_key.weight", {D.d, D.hc});
    L.ple_value = b.need(s + "ple_value.weight", {D.d, D.d});
    L.ple_conv1d = b.need(s + "ple_conv1d.weight", {h.ple_conv, D.hc});
    L.ple_norm_key = b.need(s + "ple_norm_key.weight", {D.hc});
    L.ple_norm_query = b.need(s + "ple_norm_query.weight", {D.hc});
    L.ple_norm_conv = b.need(s + "ple_norm_conv.weight", {D.hc});
}

Layer bind_layer(Binder& b, const Hparams& h, const Dims& D, uint32_t l) {
    const std::string s = std::format("blk.{}.", l);
    Layer L;
    L.gr_attn = bind_gr(b, s, D, h.n_hc, "attn");
    L.gr_ffn = bind_gr(b, s, D, h.n_hc, "ffn");
    if (h.is_qsa(l))
        bind_qsa(b, s, h, L);
    else
        bind_deltanet(b, s, h, L);
    bind_moe(b, s, D, L);
    if (l == h.ple_layer) bind_ple(b, s, h, D, L);
    return L;
}

CommonDims common_dims(const Gguf& g, const Hparams& h) {
    const std::string key = prefix() + "context_length";
    const uint32_t ctx = g.has(key) ? (uint32_t)g.get_uint(key) : 0;
    return {h.vocab, h.d, h.n_layer, ctx, h.rope_base, h.rms_eps};
}

}  // namespace

Qwen4ExpModel::Qwen4ExpModel(const std::string& path) : ModelBase(ARCH, path) {
    const Gguf& g = gguf();
    Binder b(g, ARCH_NAME);
    b.check(detect_arch(g) == ARCH,
            std::format("architecture {} instead of {}", g.get_string("general.architecture"), ARCH_NAME));
    hp_ = read_hparams(b);
    Hparams& h = hp_;
    apply_idx_topk_env(h);

    const Dims D = dims_of(h, h.n_expert);
    tok_embd_ = g.find("token_embd.weight");
    b.check(tok_embd_ && tok_embd_->dims.size() == 2 && tok_embd_->dims[0] == D.d,
            "token_embd is missing or has the wrong shape");
    h.vocab = (uint32_t)tok_embd_->dims[1];
    b.mark_used("token_embd.weight");
    output_ = b.need("output.weight", {D.d, h.vocab});
    gr_head_ = {b.need("output_hc_norm.weight", {D.hc}), b.need("output_hc_down.weight", {D.hc, D.r}),
                b.need("output_hc_up.weight", {D.r, D.hc}), nullptr};
    const uint64_t ple_rows = h.ple_head_offset.back() + h.ple_head_vocab.back();
    ple_embd_ = g.find("per_layer_token_embd.weight");
    b.check(
        ple_embd_ && ple_embd_->dims.size() == 2 && ple_embd_->dims[0] == h.ple_dim && ple_embd_->dims[1] >= ple_rows,
        "per_layer_token_embd is missing or too small");
    b.mark_used("per_layer_token_embd.weight");

    for (uint32_t l = 0; l < h.n_layer; l++) layers_.push_back(bind_layer(b, h, D, l));
    for (const Layer& L : layers_) {
        MoeWeights w;
        w.router = L.router, w.gate_exps = L.gate_exps, w.up_exps = L.up_exps, w.down_exps = L.down_exps;
        w.shexp_gate_inp = L.shexp_gate_inp, w.shexp_gate = L.shexp_gate, w.shexp_up = L.shexp_up;
        w.shexp_down = L.shexp_down;
        moe_.push_back(w);
    }

    // Nothing may be left over: an unknown tensor means the description is incomplete
    b.expect_all_used();
    set_dims(common_dims(g, h));
}

MtpHead::MtpHead(const std::string& path, const Hparams& h) : gguf(path) {
    const Gguf& g = gguf;
    const std::string p = prefix();
    Binder b(g, ARCH_NAME);
    b.check(g.get_string("general.architecture") == ARCH_NAME, "MTP: architecture is not qwen4exp");
    b.check(b.u32(p + "nextn_predict_layers") == 1, "MTP: exactly one layer expected");
    b.check(b.u32(p + "block_count") == h.n_layer + 1 && b.u32(p + "embedding_length") == h.d &&
                b.u32(p + "attention.head_count") == h.n_head && b.u32(p + "attention.head_count_kv") == h.n_head_kv &&
                b.u32(p + "attention.key_length") == h.head_dim &&
                b.u32(p + "expert_feed_forward_length") == h.expert_ff &&
                b.u32(p + "expert_shared_feed_forward_length") == h.shexp_ff &&
                b.u32(p + "hyper_connection.count") == h.n_hc && b.u32(p + "hyper_connection.low_rank") == h.hc_rank &&
                b.u32(p + "attention.indexer.top_k") == h.idx_top_k_gguf,
            "MTP: shapes do not match the target model");
    n_expert = b.u32(p + "expert_count");
    n_used = b.u32(p + "expert_used_count");
    rope_base = (float)g.get_float(p + "rope.freq_base");
    const Dims D = dims_of(h, n_expert);
    const std::string s = std::format("blk.{}.", h.n_layer);
    L.gr_attn = bind_gr(b, s, D, h.n_hc, "attn");
    L.gr_ffn = bind_gr(b, s, D, h.n_hc, "ffn");
    bind_qsa(b, s, h, L);
    bind_moe(b, s, D, L);
    eh_proj = b.need(s + "nextn.eh_proj.weight", {2 * D.d, D.d});
    enorm = b.need(s + "nextn.enorm.weight", {D.d});
    hnorm = b.need(s + "nextn.hnorm.weight", {D.hc});
    gr_head = {b.need(s + "nextn.hc_head_norm.weight", {D.hc}), b.need(s + "nextn.hc_head_down.weight", {D.hc, D.r}),
               b.need(s + "nextn.hc_head_up.weight", {D.r, D.hc}), nullptr};
    // Embedding and LM head are included, but the target model's are used for the computation
    b.mark_used("token_embd.weight"), b.mark_used("output.weight");
    b.expect_all_used("MTP: ");
}

WeightPlan Qwen4ExpModel::plan() const {
    WeightPlan w;
    for (const auto& t : gguf().tensors()) {
        if (t.name == "token_embd.weight")
            w.embedding += t.bytes;
        else if (t.name == "per_layer_token_embd.weight")
            w.ple_table += t.bytes;
        else if (t.name.contains("_exps."))
            w.experts += t.bytes;
        else {
            w.dense += t.bytes;
            if (t.name.contains("hc_")) w.dense_hc += t.bytes;
        }
    }
    w.dense_lm_head = output_->bytes;
    w.expert_bytes = w.experts / ((uint64_t)hp_.n_layer * hp_.n_expert);
    w.experts_per_token = w.expert_bytes * hp_.n_expert_used * hp_.n_layer;
    w.ple_row_bytes = ple_embd_->bytes / ple_embd_->dims[1];
    return w;
}

MoeShape Qwen4ExpModel::moe_shape() const {
    MoeShape s{hp_.n_layer,       hp_.d,         hp_.vocab,    hp_.n_expert,
               hp_.n_expert_used, hp_.expert_ff, hp_.shexp_ff, hp_.rope_base};
    s.n_hc = hp_.n_hc;
    s.ple_ngram = hp_.ple_ngram, s.ple_heads = hp_.ple_heads, s.ple_dim = hp_.ple_dim;
    s.qsa_interval = hp_.full_attn_interval, s.idx_block = hp_.idx_block, s.idx_dim = hp_.idx_dim;
    return s;
}

// The same bytes in the same order as the fingerprint of the prompt cache hashed them before the model layer was split
void Qwen4ExpModel::fingerprint(Hasher& h) const {
    // A different top-k gives other hidden states, so other KV rows: only the default keeps the old fingerprint
    if (hp_.idx_top_k != hp_.idx_top_k_gguf) h.value(hp_.idx_top_k);
    // YaRN rotates the keys of the QSA layers by other angles: a checkpoint of one mode is wrong in the other. Off
    // keeps the old fingerprint
    if (const rope::YarnTable& y = rope::yarn_active(hp_.rope_base); y.factor > 1.0) {
        h.value(y.mul);
        h.value(y.mscale);
    }
}

std::string Qwen4ExpModel::describe() const {
    return std::format("qwen4exp: {} layers ({} QSA), d {}, {} experts (top {}), vocab {}", hp_.n_layer,
                       hp_.n_layer / hp_.full_attn_interval, hp_.d, hp_.n_expert, hp_.n_expert_used, hp_.vocab);
}

void Qwen4ExpModel::rows_at(const int32_t* seq, int64_t i, int64_t* out) const { ple::rows_at(hp_, seq, i, out); }

}  // namespace tona
