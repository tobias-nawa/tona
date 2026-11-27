// The nemotron_h_moe model module and its CPU oracle on a tiny hand-written file (random F32 weights): binding with
// uneven blocks (different expert counts, widths and top-k per layer), refusals of a wrong shape, an unknown tensor, a
// changed layer pattern and a wrong routing rule, the oracle's basic behaviour, and the Mamba-2 step against numbers
// computed by hand. The real file is checked by tona-model and tona-ref, see the measurement notes.
#include <cmath>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "cpu/ref.h"
#include "cpu/ref_mamba2.h"
#include "model/model_registry.h"
#include "model/nemotron_h_moe.h"
#include "tests/test_support.h"

using tona::test::GgufWriter;
using tona::test::throws;

namespace {

using Dims = std::vector<uint64_t>;
using Spec = std::vector<std::pair<std::string, Dims>>;

// d 8, 5 blocks: Mamba-2, MoE (4 experts, top 2, ff 4), attention (1 KV head), MoE (6 experts, top 3, ff 2), Mamba-2.
// Mamba-2: inner 8 = 2 heads x 4, 2 groups, state 2, conv 4. Latent 4, shared expert ff 6, vocab 16.
constexpr uint64_t D = 8, VOCAB = 16, LATENT = 4, SHEXP = 6;

void add_mamba(Spec& s, const std::string& p) {
    s.push_back({p + "attn_norm.weight", {D}});
    s.push_back({p + "ssm_in.weight", {D, 26}});
    s.push_back({p + "ssm_in.scale", {1}});
    s.push_back({p + "ssm_out.weight", {8, D}});
    s.push_back({p + "ssm_out.scale", {1}});
    s.push_back({p + "ssm_conv1d.weight", {4, 16}});
    s.push_back({p + "ssm_conv1d.bias", {16}});
    s.push_back({p + "ssm_dt.bias", {2}});
    s.push_back({p + "ssm_a", {1, 2}});
    s.push_back({p + "ssm_d", {1, 2}});
    s.push_back({p + "ssm_norm.weight", {4, 2}});
}

void add_attention(Spec& s, const std::string& p) {
    s.push_back({p + "attn_norm.weight", {D}});
    s.push_back({p + "attn_q.weight", {D, 8}});
    s.push_back({p + "attn_k.weight", {D, 4}});
    s.push_back({p + "attn_v.weight", {D, 4}});
    s.push_back({p + "attn_output.weight", {8, D}});
}

void add_moe(Spec& s, const std::string& p, uint64_t experts, uint64_t ff) {
    s.push_back({p + "attn_norm.weight", {D}});
    s.push_back({p + "ffn_gate_inp.weight", {D, experts}});
    s.push_back({p + "exp_probs_b.bias", {experts}});
    s.push_back({p + "ffn_up_exps.weight", {LATENT, ff, experts}});
    s.push_back({p + "ffn_down_exps.weight", {ff, LATENT, experts}});
    s.push_back({p + "ffn_latent_down.weight", {D, LATENT}});
    s.push_back({p + "ffn_latent_up.weight", {LATENT, D}});
    s.push_back({p + "ffn_up_shexp.weight", {D, SHEXP}});
    s.push_back({p + "ffn_down_shexp.weight", {SHEXP, D}});
}

Spec tiny_tensors() {
    Spec s{{"token_embd.weight", {D, VOCAB}}, {"output_norm.weight", {D}}, {"output.weight", {D, VOCAB}}};
    add_mamba(s, "blk.0.");
    add_moe(s, "blk.1.", 4, 4);
    add_attention(s, "blk.2.");
    add_moe(s, "blk.3.", 6, 2);
    add_mamba(s, "blk.4.");
    return s;
}

struct Keys {
    uint32_t gating = 2;
    std::vector<int32_t> kv{0, 0, 1, 0, 0};
};

// Deterministic values: norms near 1, the decay rate negative, the router bias small, everything else small
std::vector<float> values_of(const std::string& name, uint64_t n, uint64_t& seed) {
    std::vector<float> v(n);
    for (auto& x : v) {
        seed = seed * 6364136223846793005ull + 1442695040888963407ull;
        const float u = (float)((seed >> 40) & 0xFFFF) / 65535.0f - 0.5f;  // [-0.5, 0.5]
        if (name.contains("norm"))
            x = 1.0f + 0.1f * u;
        else if (name.contains("ssm_a"))
            x = -0.5f;
        else if (name.contains(".scale"))
            x = 1.0f + u;
        else
            x = 0.6f * u;
    }
    return v;
}

std::string write_model(const tona::test::TempDir& dir, const std::string& file, const Spec& tensors,
                        const Keys& k = {}) {
    GgufWriter w;
    w.header(tensors.size(), 27);
    w.kv_str("general.architecture", "nemotron_h_moe");
    w.kv_u32("nemotron_h_moe.block_count", 5);
    w.kv_u32("nemotron_h_moe.context_length", 128);
    w.kv_u32("nemotron_h_moe.embedding_length", D);
    w.kv_arr_i32("nemotron_h_moe.feed_forward_length", {0, 4, 0, 2, 0});
    w.kv_u32("nemotron_h_moe.attention.head_count", 2);
    w.kv_arr_i32("nemotron_h_moe.attention.head_count_kv", k.kv);
    w.kv_f32("nemotron_h_moe.attention.layer_norm_rms_epsilon", 1e-5f);
    w.kv_u32("nemotron_h_moe.attention.key_length", 4);
    w.kv_u32("nemotron_h_moe.attention.value_length", 4);
    w.kv_u32("nemotron_h_moe.expert_count", 6);
    w.kv_u32("nemotron_h_moe.expert_group_count", 1);
    w.kv_u32("nemotron_h_moe.expert_gating_func", k.gating);
    w.kv_u32("nemotron_h_moe.vocab_size", VOCAB);
    w.kv_u32("nemotron_h_moe.ssm.conv_kernel", 4);
    w.kv_u32("nemotron_h_moe.ssm.state_size", 2);
    w.kv_u32("nemotron_h_moe.ssm.group_count", 2);
    w.kv_u32("nemotron_h_moe.ssm.inner_size", 8);
    w.kv_u32("nemotron_h_moe.ssm.time_step_rank", 2);
    w.kv_arr_i32("nemotron_h_moe.expert_feed_forward_length", {0, 4, 0, 2, 0});
    w.kv_arr_i32("nemotron_h_moe.expert_used_count", {0, 2, 0, 3, 0});
    w.kv_u32("nemotron_h_moe.expert_shared_feed_forward_length", SHEXP);
    w.kv_u32("nemotron_h_moe.expert_shared_count", 1);
    w.kv_bool("nemotron_h_moe.expert_weights_norm", true);
    w.kv_f32("nemotron_h_moe.expert_weights_scale", 5.0f);
    w.kv_u32("nemotron_h_moe.moe_latent_size", LATENT);
    w.kv_u32("general.quantization_version", 2);
    uint64_t off = 0;
    for (const auto& [n, d] : tensors) {
        uint64_t el = 1;
        for (uint64_t x : d) el *= x;
        w.tensor(n, d, 0, off);
        off += (el * 4 + 31) / 32 * 32;
    }
    w.align();
    uint64_t seed = 4242;
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

using tona::BoltBlockKind;

void test_uneven(const tona::NemotronHMoeModel& m);
void test_module_facts(const tona::NemotronHMoeModel& m);
void test_plan(const tona::NemotronHMoeModel& m);

void test_binding(const tona::test::TempDir& dir) {
    const auto base = tona::load_model(write_model(dir, "ok.gguf", tiny_tensors()));
    const auto& m = tona::model_as<tona::NemotronHMoeModel>(*base);
    EXPECT(m.arch() == tona::ArchId::NemotronHMoe && m.hp().n_layer == 5 && m.hp().vocab == VOCAB);
    const BoltBlockKind want[] = {BoltBlockKind::Mamba2, BoltBlockKind::Moe, BoltBlockKind::Attention,
                                  BoltBlockKind::Moe, BoltBlockKind::Mamba2};
    for (uint32_t l = 0; l < 5; l++) EXPECT(m.layers()[l].kind == want[l] && m.layers()[l].attn_norm);
    EXPECT(m.count(BoltBlockKind::Mamba2) == 2 && m.count(BoltBlockKind::Attention) == 1 &&
           m.count(BoltBlockKind::Moe) == 2);
    EXPECT(m.layers()[0].ssm.in_proj && m.layers()[0].ssm.in_scale && !m.layers()[0].moe.router);
    EXPECT(m.layers()[2].attn.q && m.layers()[2].n_kv == 1 && !m.layers()[2].attn.q_norm);
    test_uneven(m);
    test_module_facts(m);
    test_plan(m);
}

// The blocks are uneven: the experts of a layer are bound with their own count, width and top-k
void test_uneven(const tona::NemotronHMoeModel& m) {
    EXPECT(m.layers()[1].n_expert == 4 && m.layers()[1].n_used == 2 && m.layers()[1].expert_ff == 4);
    EXPECT(m.layers()[3].n_expert == 6 && m.layers()[3].n_used == 3 && m.layers()[3].expert_ff == 2);
    EXPECT(m.layers()[1].moe.latent_down && !m.layers()[1].moe.gate_exps && m.layers()[1].moe.router_bias);
}

void test_module_facts(const tona::NemotronHMoeModel& m) {
    EXPECT(m.dims().d_model == D && m.dims().n_layer == 5 && m.dims().native_context == 128);
    EXPECT(m.chat_family() == tona::ChatFamily::Nemotron && m.mtp_source() == tona::MtpSource::None);
    // The trunk as the prefill walks it: 5 blocks, 2 of them MoE (blocks 1 and 3), the KV cache in block 2 only
    EXPECT(m.n_blocks() == 5 && m.moe_shape().n_layer == 2);
    EXPECT(m.moe_block(0) == 1 && m.moe_block(1) == 3);
    const int ordinal[] = {-1, 0, -1, 1, -1};
    for (uint32_t l = 0; l < 5; l++) EXPECT(m.moe_ordinal(l) == ordinal[l] && m.has_kv(l) == (l == 2));
    EXPECT(m.describe().contains("5 blocks (2 Mamba-2, 1 attention, 2 MoE)"));
    tona::Hasher h;
    m.fingerprint(h);
    EXPECT(h.digest() == tona::Hasher::BASIS);

    // The MoE blocks only, indexed in order
    const tona::MoeShape s = m.moe_shape();
    EXPECT(s.n_layer == 2 && s.n_expert == 6 && s.n_expert_used == 3 && s.expert_ff == 4 && s.shexp_ff == SHEXP);
    EXPECT(m.moe_layer(0) == 1 && m.moe_layer(1) == 3 && &m.moe_weights(1) == &m.layers()[3].moe);
}

void test_plan(const tona::NemotronHMoeModel& m) {
    // Weight plan: the bytes per token follow each layer's own top-k and count
    const tona::WeightPlan w = m.plan();
    const uint64_t layer1 = (4 * 4 * 4 + 4 * 4 * 4) * 4ull, layer3 = (4 * 2 * 6 + 2 * 4 * 6) * 4ull;
    EXPECT(w.experts == layer1 + layer3 && w.embedding == D * VOCAB * 4);
    EXPECT(w.experts_per_token == layer1 / 4 * 2 + layer3 / 6 * 3 && w.expert_bytes == (layer1 + layer3) / 10);
}

void test_refusals(const tona::test::TempDir& dir) {
    Spec wrong = tiny_tensors();
    for (auto& [n, d] : wrong)
        if (n == "blk.1.ffn_up_exps.weight") d = {LATENT, 5, 4};  // ff 5 instead of the 4 the array says
    EXPECT(throws([&] { tona::load_model(write_model(dir, "shape.gguf", wrong)); }));

    Spec extra = tiny_tensors();
    extra.push_back({"blk.2.unknown.weight", {4}});
    EXPECT(throws([&] { tona::load_model(write_model(dir, "extra.gguf", extra)); }));

    Spec missing = tiny_tensors();
    missing.pop_back();  // ssm_norm of the last block
    EXPECT(throws([&] { tona::load_model(write_model(dir, "missing.gguf", missing)); }));

    // The arrays say attention at 1; the tensors say attention at 2: the load stops, not the first token
    Keys other;
    other.kv = {0, 1, 0, 0, 0};
    EXPECT(throws([&] { tona::load_model(write_model(dir, "pattern.gguf", tiny_tensors(), other)); }));

    Keys softmax;
    softmax.gating = 1;  // the softmax router is another policy
    EXPECT(throws([&] { tona::load_model(write_model(dir, "gating.gguf", tiny_tensors(), softmax)); }));

    Spec many = tiny_tensors();  // more experts in a layer than expert_count
    for (auto& [n, d] : many) {
        if (n == "blk.3.ffn_gate_inp.weight") d = {D, 7};
        if (n == "blk.3.exp_probs_b.bias") d = {7};
        if (n == "blk.3.ffn_up_exps.weight") d = {LATENT, 2, 7};
        if (n == "blk.3.ffn_down_exps.weight") d = {2, LATENT, 7};
    }
    EXPECT(throws([&] { tona::load_model(write_model(dir, "many.gguf", many)); }));
}

bool all_finite(const std::vector<float>& v) {
    for (float x : v)
        if (!std::isfinite(x)) return false;
    return true;
}

// One entry per MoE block with its own top-k, distinct experts inside that layer's own count
void check_experts(const tona::ref::Trace& trace) {
    EXPECT(trace.experts.size() == 2 && trace.experts[0].size() == 2 && trace.experts[1].size() == 3);
    for (const auto& e : trace.experts)
        for (size_t i = 0; i < e.size(); i++)
            for (size_t j = 0; j < i; j++) EXPECT(e[i] != e[j]);
    for (int e : trace.experts[0]) EXPECT(e >= 0 && e < 4);
    for (int e : trace.experts[1]) EXPECT(e >= 0 && e < 6);
}

void test_oracle(const tona::test::TempDir& dir) {
    const auto base = tona::load_model(write_model(dir, "ok.gguf", tiny_tensors()));
    const auto oracle = tona::ref::make_oracle(*base);
    EXPECT(oracle->max_ctx() == 128 && oracle->pos() == 0);

    tona::ref::Trace trace;
    const std::vector<float> first = oracle->step(3, &trace);
    EXPECT(first.size() == VOCAB && all_finite(first));
    EXPECT(trace.residual.size() == 5 && trace.residual[0].size() == D);  // one stream, one entry per block
    check_experts(trace);

    // The same token at the next position sees the Mamba-2 state, the conv state and the cache: another answer
    const std::vector<float> second = oracle->step(3);
    EXPECT(all_finite(second) && oracle->pos() == 2 && first != second);

    // A reset starts over, bit for bit
    (*oracle).reset();
    EXPECT(oracle->pos() == 0 && oracle->step(3) == first);
    EXPECT(throws([&] { oracle->step((int32_t)VOCAB); }));
    EXPECT(throws([&] { oracle->step(-1); }));
}

// Two heads of two values, state 2, two groups with one head each; every value is small enough to compute by hand
void test_ssm_step_by_hand() {
    const float x[] = {1, 2, 3, 4}, B[] = {1, 0, 0, 1}, C[] = {1, 1, 1, 2}, dt[] = {0, 0}, a[] = {-1, -2},
                d[] = {0.5f, 0.25f};
    const tona::ref::SsmStepInputs in{x, B, C, dt, a, d, 2, 2, 2, 2};
    std::vector<float> state(8, 0.0f), y(4);
    const double ln2 = std::log(2.0);  // softplus(0)

    // First step from a zero state: s = B * x * ln2, y = s . C + x * d
    tona::ref::ssm_step(in, state.data(), y.data());
    const double want1[] = {1 * ln2 + 0.5, 2 * ln2 + 1.0, 3 * ln2 * 2 + 0.75, 4 * ln2 * 2 + 1.0};
    for (int i = 0; i < 4; i++) EXPECT(std::fabs(y[i] - want1[i]) < 1e-5);
    EXPECT(std::fabs(state[0] - ln2) < 1e-6 && state[1] == 0.0f && state[4] == 0.0f);  // head 0, value 0 / head 1
    EXPECT(std::fabs(state[5] - 3 * ln2) < 1e-6);  // head 1, value 0: B[1] * 3 * ln2

    // Second step: the state decays by exp(ln2 * a) = 1/2 for head 0 and 1/4 for head 1, then takes the new input
    tona::ref::ssm_step(in, state.data(), y.data());
    const double want2[] = {(1 * ln2 * 0.5 + 1 * ln2) + 0.5, (2 * ln2 * 0.5 + 2 * ln2) + 1.0,
                            (3 * ln2 * 0.25 + 3 * ln2) * 2 + 0.75, (4 * ln2 * 0.25 + 4 * ln2) * 2 + 1.0};
    for (int i = 0; i < 4; i++) EXPECT(std::fabs(y[i] - want2[i]) < 1e-5);
}

}  // namespace

int main() {
    return tona::test::run("nemotron_h", [] {
        const tona::test::TempDir dir("nemotron_h");
        test_binding(dir);
        test_refusals(dir);
        test_oracle(dir);
        test_ssm_step_by_hand();
    });
}
