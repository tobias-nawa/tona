// The qwen35moe model module and its CPU oracle on a tiny hand-written file (random F32 weights): binding, refusals of
// a wrong shape, an unknown tensor and a changed layer pattern, and the oracle's basic behaviour (finite logits, state
// and cache in use, the trace). The real file is checked by tona-model and tona-ref, see the measurement notes.
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cpu/ref.h"
#include "model/model_registry.h"
#include "model/qwen35moe.h"
#include "tests/test_support.h"

using tona::test::GgufWriter;
using tona::test::throws;

namespace {

using Dims = std::vector<uint64_t>;
using Spec = std::vector<std::pair<std::string, Dims>>;

// d 8, 4 layers (attention at 1 and 3) + MTP blk.4, 2 K / 4 V heads of 4, 2 Q / 1 KV heads of 4, 4 experts (top 2)
constexpr uint64_t D = 8, VOCAB = 16;

void add_moe(Spec& s, const std::string& p) {
    s.push_back({p + "attn_norm.weight", {D}});
    s.push_back({p + "post_attention_norm.weight", {D}});
    s.push_back({p + "ffn_gate_inp.weight", {D, 4}});
    s.push_back({p + "ffn_gate_exps.weight", {D, 4, 4}});
    s.push_back({p + "ffn_up_exps.weight", {D, 4, 4}});
    s.push_back({p + "ffn_down_exps.weight", {4, D, 4}});
    s.push_back({p + "ffn_gate_inp_shexp.weight", {D}});
    s.push_back({p + "ffn_gate_shexp.weight", {D, 4}});
    s.push_back({p + "ffn_up_shexp.weight", {D, 4}});
    s.push_back({p + "ffn_down_shexp.weight", {4, D}});
}

void add_attention(Spec& s, const std::string& p) {
    s.push_back({p + "attn_q.weight", {D, 16}});
    s.push_back({p + "attn_k.weight", {D, 4}});
    s.push_back({p + "attn_v.weight", {D, 4}});
    s.push_back({p + "attn_output.weight", {8, D}});
    s.push_back({p + "attn_q_norm.weight", {4}});
    s.push_back({p + "attn_k_norm.weight", {4}});
}

void add_deltanet(Spec& s, const std::string& p) {
    s.push_back({p + "attn_qkv.weight", {D, 32}});
    s.push_back({p + "attn_gate.weight", {D, 16}});
    s.push_back({p + "ssm_beta.weight", {D, 4}});
    s.push_back({p + "ssm_alpha.weight", {D, 4}});
    s.push_back({p + "ssm_conv1d.weight", {4, 32}});
    s.push_back({p + "ssm_dt.bias", {4}});
    s.push_back({p + "ssm_a", {4}});
    s.push_back({p + "ssm_norm.weight", {4}});
    s.push_back({p + "ssm_out.weight", {16, D}});
}

Spec tiny_tensors() {
    Spec s{{"token_embd.weight", {D, VOCAB}}, {"output_norm.weight", {D}}, {"output.weight", {D, VOCAB}}};
    for (int l = 0; l < 4; l++) {
        const std::string p = "blk." + std::to_string(l) + ".";
        (l % 2 == 1 ? add_attention : add_deltanet)(s, p);
        add_moe(s, p);
    }
    add_attention(s, "blk.4.");
    add_moe(s, "blk.4.");
    s.push_back({"blk.4.nextn.eh_proj.weight", {2 * D, D}});
    s.push_back({"blk.4.nextn.enorm.weight", {D}});
    s.push_back({"blk.4.nextn.hnorm.weight", {D}});
    s.push_back({"blk.4.nextn.shared_head_norm.weight", {D}});
    return s;
}

struct Keys {
    uint32_t interval = 2;
    std::vector<int32_t> sections{1, 1, 0, 0};
};

// Deterministic values: norms near 1, the decay rate negative, everything else small
std::vector<float> values_of(const std::string& name, uint64_t n, uint64_t& seed) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const float u = (float)((seed >> 40) & 0xFFFF) / 65535.0f - 0.5f;  // [-0.5, 0.5]
        if (name.contains("norm"))
            x = 1.0f + 0.1f * u;
        else if (name.contains("ssm_a"))
            x = -0.5f;
        else
            x = 0.6f * u;
    }
    return v;
}

std::string write_model(const tona::test::TempDir& dir, const std::string& file, const Spec& tensors,
                        const Keys& k = {}) {
    GgufWriter w;
    w.header(tensors.size(), 23);
    w.kv_str("general.architecture", "qwen35moe");
    w.kv_u32("qwen35moe.block_count", 5);
    w.kv_u32("qwen35moe.nextn_predict_layers", 1);
    w.kv_u32("qwen35moe.embedding_length", D);
    w.kv_u32("qwen35moe.context_length", 128);
    w.kv_f32("qwen35moe.attention.layer_norm_rms_epsilon", 1e-6f);
    w.kv_u32("qwen35moe.full_attention_interval", k.interval);
    w.kv_u32("qwen35moe.ssm.group_count", 2);
    w.kv_u32("qwen35moe.ssm.time_step_rank", 4);
    w.kv_u32("qwen35moe.ssm.state_size", 4);
    w.kv_u32("qwen35moe.ssm.conv_kernel", 4);
    w.kv_u32("qwen35moe.ssm.inner_size", 16);
    w.kv_u32("qwen35moe.attention.head_count", 2);
    w.kv_u32("qwen35moe.attention.head_count_kv", 1);
    w.kv_u32("qwen35moe.attention.key_length", 4);
    w.kv_u32("qwen35moe.attention.value_length", 4);
    w.kv_u32("qwen35moe.rope.dimension_count", 4);
    w.kv_f32("qwen35moe.rope.freq_base", 10000.0f);
    w.kv_arr_i32("qwen35moe.rope.dimension_sections", k.sections);
    w.kv_u32("qwen35moe.expert_count", 4);
    w.kv_u32("qwen35moe.expert_used_count", 2);
    w.kv_u32("qwen35moe.expert_feed_forward_length", 4);
    w.kv_u32("qwen35moe.expert_shared_feed_forward_length", 4);
    uint64_t off = 0;
    for (const auto& [n, d] : tensors) {
        uint64_t el = 1;
        for (uint64_t x : d) el *= x;
        w.tensor(n, d, 0, off);
        off += (el * 4 + 31) / 32 * 32;
    }
    w.align();
    uint64_t seed = 12345;
    for (const auto& [n, d] : tensors) {
        uint64_t el = 1;
        for (uint64_t x : d) el *= x;
        const auto v = values_of(n, el, seed);
        const size_t at = w.b.size();
        w.b.resize(at + (el * 4 + 31) / 32 * 32, 0);
        std::memcpy(w.b.data() + at, v.data(), el * 4);
    }
    const std::string path = (dir.path() / file).string();
    w.save(path);
    return path;
}

void test_module_facts(const tona::Qwen35MoeModel& m);

void test_binding(const tona::test::TempDir& dir) {
    const auto base = tona::load_model(write_model(dir, "ok.gguf", tiny_tensors()));
    const auto& m = tona::model_as<tona::Qwen35MoeModel>(*base);
    EXPECT(m.arch() == tona::ArchId::Qwen35Moe);
    EXPECT(m.hp().n_layer == 4 && m.hp().n_mtp == 1 && m.hp().vocab == VOCAB);
    EXPECT(m.layers().size() == 4);
    for (uint32_t l = 0; l < 4; l++) {
        const auto kind = m.layers()[l].kind;
        EXPECT(kind == (l % 2 == 1 ? tona::Qwen35LayerKind::GatedAttn : tona::Qwen35LayerKind::DeltaNet));
    }
    EXPECT(m.layers()[0].dn.qkv && !m.layers()[0].attn.q && m.layers()[1].attn.q && !m.layers()[1].dn.qkv);
    EXPECT(m.mtp().eh_proj && m.mtp().layer.attn.q && m.mtp().shared_head_norm);
    test_module_facts(m);
}

void test_module_facts(const tona::Qwen35MoeModel& m) {
    EXPECT(m.dims().d_model == D && m.dims().n_layer == 4 && m.dims().native_context == 128);
    EXPECT(m.chat_family() == tona::ChatFamily::Qwen35 && m.mtp_source() == tona::MtpSource::EmbeddedInModel);
    EXPECT(m.describe().contains("qwen35moe: 4 layers (2 gated attention) + 1 MTP"));
    tona::Hasher h;
    m.fingerprint(h);
    EXPECT(h.digest() == tona::Hasher::BASIS);  // YaRN off: the key adds nothing

    // The weight plan keeps the MTP layer out of the trunk's numbers
    const tona::WeightPlan w = m.plan();
    EXPECT(w.mtp > 0 && w.experts > 0 && w.dense > 0 && w.embedding == D * VOCAB * 4);
    EXPECT(w.expert_bytes * 4 * 4 == w.experts && w.experts_per_token == w.expert_bytes * 2 * 4);
}

void test_refusals(const tona::test::TempDir& dir) {
    Spec wrong = tiny_tensors();
    for (auto& [n, d] : wrong)
        if (n == "blk.1.attn_q.weight") d = {D, 8};  // the reading without the gate
    EXPECT(throws([&] { tona::load_model(write_model(dir, "shape.gguf", wrong)); }));

    Spec extra = tiny_tensors();
    extra.push_back({"blk.2.unknown.weight", {4}});
    EXPECT(throws([&] { tona::load_model(write_model(dir, "extra.gguf", extra)); }));

    Spec missing = tiny_tensors();
    missing.pop_back();  // shared_head_norm
    EXPECT(throws([&] { tona::load_model(write_model(dir, "missing.gguf", missing)); }));

    // The tensors present say attention at 1 and 3; an interval of 3 says 2: the load stops, not the first token
    Keys other;
    other.interval = 3;
    EXPECT(throws([&] { tona::load_model(write_model(dir, "interval.gguf", tiny_tensors(), other)); }));

    Keys sections;
    sections.sections = {1, 1, 1, 0};
    EXPECT(throws([&] { tona::load_model(write_model(dir, "rope.gguf", tiny_tensors(), sections)); }));
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

void test_oracle(const tona::test::TempDir& dir) {
    const auto base = tona::load_model(write_model(dir, "ok.gguf", tiny_tensors()));
    const auto oracle = tona::ref::make_oracle(*base);
    EXPECT(oracle->max_ctx() == 128 && oracle->pos() == 0);

    tona::ref::Trace trace;
    const std::vector<float> first = oracle->step(3, &trace);
    EXPECT(first.size() == VOCAB && all_finite(first));
    EXPECT(trace.residual.size() == 4 && trace.residual[0].size() == D);  // one stream
    for (const auto& e : trace.experts) EXPECT(e.size() == 2 && e[0] != e[1] && e[0] < 4 && e[1] < 4);

    // The same token at the next position sees the state and the cache: another answer
    const std::vector<float> second = oracle->step(3);
    EXPECT(all_finite(second) && oracle->pos() == 2);
    EXPECT(first != second);

    // A reset starts over, bit for bit
    (*oracle).reset();
    EXPECT(oracle->pos() == 0 && oracle->step(3) == first);
    EXPECT(throws([&] { oracle->step((int32_t)VOCAB); }));
    EXPECT(throws([&] { oracle->step(-1); }));
}

}  // namespace

int main() {
    return tona::test::run("qwen35moe", [] {
        const tona::test::TempDir dir("qwen35moe");
        test_binding(dir);
        test_refusals(dir);
        test_oracle(dir);
    });
}
