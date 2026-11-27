// Tests for the architecture-neutral model layer: architecture detection, weight types, the shape-checked binder, the
// shared block binders (with tiny hand-written files) and the model registry's refusals.
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "model/arch.h"
#include "model/binder.h"
#include "model/blocks.h"
#include "model/hasher.h"
#include "model/model_base.h"
#include "model/model_registry.h"
#include "tests/test_support.h"

using tona::test::GgufWriter;
using tona::test::throws;

namespace {

using Dims = std::vector<uint64_t>;

// A GGUF with F32 tensors of the given shapes (zeros) and the architecture name
std::string write_file(const tona::test::TempDir& dir, const std::string& name, const std::string& arch,
                       const std::vector<std::pair<std::string, Dims>>& tensors, uint32_t type = 0) {
    GgufWriter w;
    w.header(tensors.size(), 1);
    w.kv_str("general.architecture", arch);
    uint64_t off = 0;
    uint64_t total = 0;
    for (const auto& [n, d] : tensors) {
        uint64_t el = 1;
        for (uint64_t x : d) el *= x;
        const uint64_t bytes = (el * 4 + 31) / 32 * 32;
        w.tensor(n, d, type, off);
        off += bytes, total += bytes;
    }
    w.align();
    w.b.resize(w.b.size() + total, 0);
    const std::string path = (dir.path() / name).string();
    w.save(path);
    return path;
}

void test_detect_arch(const tona::test::TempDir& dir) {
    using tona::ArchId;
    const struct {
        const char* name;
        ArchId id;
    } known[] = {
        {"qwen4exp", ArchId::Qwen4Exp}, {"qwen35moe", ArchId::Qwen35Moe}, {"nemotron_h_moe", ArchId::NemotronHMoe}};
    for (const auto& k : known) {
        tona::Gguf g(write_file(dir, "a.gguf", k.name, {{"x", {4}}}));
        EXPECT(tona::detect_arch(g) == k.id);
        EXPECT(tona::arch_name(k.id) == k.name);
    }
    tona::Gguf bad(write_file(dir, "b.gguf", "llama", {{"x", {4}}}));
    try {
        tona::detect_arch(bad);
        EXPECT(false);
    } catch (const std::exception& e) {
        // The message names the unknown and every known architecture
        const std::string m = e.what();
        EXPECT(m.contains("llama") && m.contains("qwen4exp") && m.contains("nemotron_h_moe"));
    }
    EXPECT(tona::arch_needs(ArchId::Qwen4Exp).indexer && !tona::arch_needs(ArchId::Qwen4Exp).full_attention);
    EXPECT(tona::arch_needs(ArchId::NemotronHMoe).recurrent == tona::RecurrentKind::Mamba2);
    EXPECT(tona::arch_needs(ArchId::Qwen35Moe).embedded_mtp);
}

void test_weight_types(const tona::test::TempDir& dir) {
    GgufWriter w;
    w.header(4, 1);
    w.kv_str("general.architecture", "qwen4exp");
    w.tensor("token_embd.weight", {32}, 8, 128);  // read on the host: not listed, although Q8_0 is
    w.tensor("first_q8.weight", {32}, 8, 0);
    w.tensor("f32.weight", {4}, 0, 64);
    w.tensor("second_q8.weight", {32}, 8, 96);
    w.align();
    w.b.resize(w.b.size() + 300, 0);
    w.save(dir.path() / "t.gguf");
    tona::Gguf g((dir.path() / "t.gguf").string());
    const auto types = tona::weight_types(g);
    EXPECT(types.size() == 2);
    EXPECT(types[0].type == 0 && types[0].first_tensor == "f32.weight");
    EXPECT(types[1].type == 8 && types[1].first_tensor == "first_q8.weight");
}

void test_hasher() {
    uint8_t a = 'a';
    tona::Hasher h;
    h.value(a);
    EXPECT(h.digest() == 0xaf63dc4c8601ec8cull);  // FNV-1a 64 of "a", the value server::fnv gives
    EXPECT(tona::Hasher().digest() == tona::Hasher::BASIS);
}

void test_binder(const tona::test::TempDir& dir) {
    tona::Gguf g(write_file(dir, "b.gguf", "tiny", {{"m.weight", {4, 2}}, {"extra", {3}}}));
    tona::Binder b(g, "tiny");
    EXPECT(b.need("m.weight", {4, 2}) != nullptr);
    EXPECT(b.optional("nothing", {1}) == nullptr);
    EXPECT(throws([&] { b.need("m.weight", {2, 4}); }));
    EXPECT(throws([&] { b.need("absent", {1}); }));
    EXPECT(throws([&] { b.optional("extra", {4}); }));  // present with another shape: an error, not "absent"
    // "extra" was never bound successfully, so the file has a tensor nobody asked for
    EXPECT(throws([&] { b.expect_all_used(); }));
    b.mark_used("extra");
    b.expect_all_used();
    try {
        b.fail("boom");
    } catch (const std::exception& e) {
        EXPECT(std::string(e.what()) == "tiny: boom");
    }
}

std::vector<std::pair<std::string, Dims>> mamba2_tensors() {
    // d 4, inner 8, 2 heads, state 2, 2 groups, conv 4: conv_dim 16, in_proj 26
    return {{"blk.0.ssm_in.weight", {4, 26}}, {"blk.0.ssm_out.weight", {8, 4}}, {"blk.0.ssm_conv1d.weight", {4, 16}},
            {"blk.0.ssm_conv1d.bias", {16}},  {"blk.0.ssm_dt.bias", {2}},       {"blk.0.ssm_a", {1, 2}},
            {"blk.0.ssm_d", {1, 2}},          {"blk.0.ssm_norm.weight", {4, 2}}};
}

void test_mamba2(const tona::test::TempDir& dir) {
    const tona::Mamba2Dims D{4, 8, 2, 2, 2, 4};
    EXPECT(D.conv_dim() == 16 && D.in_proj_dim() == 26);
    tona::Gguf g(write_file(dir, "m.gguf", "tiny", mamba2_tensors()));
    tona::Binder b(g, "tiny");
    const auto w = tona::bind_mamba2_weights(b, "blk.0.", D);
    EXPECT(w.in_proj && w.norm && !w.in_scale && !w.out_scale);
    b.expect_all_used();

    auto with_scales = mamba2_tensors();
    with_scales.push_back({"blk.0.ssm_in.scale", {1}});
    with_scales.push_back({"blk.0.ssm_out.scale", {1}});
    tona::Gguf g2(write_file(dir, "m2.gguf", "tiny", with_scales));
    tona::Binder b2(g2, "tiny");
    const auto w2 = tona::bind_mamba2_weights(b2, "blk.0.", D);
    EXPECT(w2.in_scale && w2.out_scale);
    b2.expect_all_used();

    auto wrong = mamba2_tensors();
    wrong[0].second = {4, 25};
    tona::Gguf g3(write_file(dir, "m3.gguf", "tiny", wrong));
    tona::Binder b3(g3, "tiny");
    EXPECT(throws([&] { tona::bind_mamba2_weights(b3, "blk.0.", D); }));
}

void test_moe(const tona::test::TempDir& dir) {
    // Latent MoE without gate tensor: d 4, latent 2, ff 3, 5 experts, shared ff 6, router bias
    tona::Gguf g(write_file(dir, "e.gguf", "tiny",
                            {{"blk.1.ffn_gate_inp.weight", {4, 5}},
                             {"blk.1.exp_probs_b.bias", {5}},
                             {"blk.1.ffn_up_exps.weight", {2, 3, 5}},
                             {"blk.1.ffn_down_exps.weight", {3, 2, 5}},
                             {"blk.1.ffn_latent_down.weight", {4, 2}},
                             {"blk.1.ffn_latent_up.weight", {2, 4}},
                             {"blk.1.ffn_up_shexp.weight", {4, 6}},
                             {"blk.1.ffn_down_shexp.weight", {6, 4}}}));
    tona::Binder b(g, "tiny");
    tona::MoeDims D{4, 5, 3, 6, false, false, 2, true};
    const auto w = tona::bind_moe_weights(b, "blk.1.", D);
    EXPECT(w.router_bias && w.latent_down && w.latent_up && !w.gate_exps && !w.shexp_gate && w.shexp_up);
    b.expect_all_used();
    // The same file read as a gated MoE without latent space is refused
    tona::Binder b2(g, "tiny");
    EXPECT(throws([&] { tona::bind_moe_weights(b2, "blk.1.", {4, 5, 3, 6, true, true, 0, false}); }));
}

void test_attention_and_deltanet(const tona::test::TempDir& dir) {
    // Gated attention with QK norm and no indexer: d 4, 2 heads, 1 KV head, head dim 2
    tona::Gguf g(write_file(dir, "a.gguf", "tiny",
                            {{"blk.3.attn_q.weight", {4, 8}},
                             {"blk.3.attn_k.weight", {4, 2}},
                             {"blk.3.attn_v.weight", {4, 2}},
                             {"blk.3.attn_output.weight", {4, 4}},
                             {"blk.3.attn_q_norm.weight", {2}},
                             {"blk.3.attn_k_norm.weight", {2}}}));
    tona::Binder b(g, "tiny");
    const auto w = tona::bind_attention_weights(b, "blk.3.", {4, 2, 1, 2, true, true, 0, 0});
    EXPECT(w.q_norm && !w.idx_q);
    b.expect_all_used();
    // Without the gate the q tensor would be [4, 4]: the shape check separates the two readings
    tona::Binder b2(g, "tiny");
    EXPECT(throws([&] { tona::bind_attention_weights(b2, "blk.3.", {4, 2, 1, 2, false, true, 0, 0}); }));

    tona::Gguf g2(write_file(dir, "d.gguf", "tiny",
                             {{"blk.0.attn_qkv.weight", {4, 2 * 4 + 8}},
                              {"blk.0.attn_gate.weight", {4, 8}},
                              {"blk.0.ssm_beta.weight", {4, 4}},
                              {"blk.0.ssm_alpha.weight", {4, 4}},
                              {"blk.0.ssm_conv1d.weight", {4, 16}},
                              {"blk.0.ssm_dt.bias", {4}},
                              {"blk.0.ssm_a", {4}},
                              {"blk.0.ssm_norm.weight", {2}},
                              {"blk.0.ssm_out.weight", {8, 4}}}));
    tona::Binder b3(g2, "tiny");
    // d 4, 2 K heads, 4 V heads, head dim 2, conv 4
    const auto dn = tona::bind_deltanet_weights(b3, "blk.0.", {4, 2, 4, 2, 4});
    EXPECT(dn.qkv && dn.ssm_out);
    b3.expect_all_used();
}

// A module for the downcast test; it never loads a file
class FakeModel final : public tona::ModelBase {
public:
    static constexpr tona::ArchId ARCH = tona::ArchId::Qwen35Moe;
    explicit FakeModel(const std::string& path) : ModelBase(ARCH, path) {}
    tona::WeightPlan plan() const override { return {}; }
    tona::MoeShape moe_shape() const override { return {}; }
    const tona::MoeWeights& moe_weights(uint32_t) const override { return moe_; }
    tona::T embedding_table() const override { return nullptr; }
    tona::T lm_head() const override { return nullptr; }
    tona::ChatFamily chat_family() const override { return tona::ChatFamily::Qwen3Coder; }
    tona::MtpSource mtp_source() const override { return tona::MtpSource::None; }
    void fingerprint(tona::Hasher&) const override {}
    std::string describe() const override { return "fake"; }

private:
    tona::MoeWeights moe_;
};

class OtherFake final : public tona::ModelBase {
public:
    static constexpr tona::ArchId ARCH = tona::ArchId::NemotronHMoe;
    explicit OtherFake(const std::string& path) : ModelBase(ARCH, path) {}
    tona::WeightPlan plan() const override { return {}; }
    tona::MoeShape moe_shape() const override { return {}; }
    const tona::MoeWeights& moe_weights(uint32_t) const override { return moe_; }
    tona::T embedding_table() const override { return nullptr; }
    tona::T lm_head() const override { return nullptr; }
    tona::ChatFamily chat_family() const override { return tona::ChatFamily::Nemotron; }
    tona::MtpSource mtp_source() const override { return tona::MtpSource::None; }
    void fingerprint(tona::Hasher&) const override {}
    std::string describe() const override { return "other"; }

private:
    tona::MoeWeights moe_;
};

void test_model_as_and_registry(const tona::test::TempDir& dir) {
    const std::string path = write_file(dir, "f.gguf", "qwen35moe", {{"x", {4}}});
    const FakeModel fake(path);
    const tona::ModelBase& base = fake;
    EXPECT(&tona::model_as<FakeModel>(base) == &fake);
    EXPECT(throws([&] { tona::model_as<OtherFake>(base); }));
    EXPECT(base.gguf().find("x") != nullptr);

    // A file of an architecture that lacks the module's keys fails in the module, with its prefix (nemotron_h_moe
    // fails in its module; the qwen35moe case follows)
    EXPECT(throws([&] { tona::load_model(write_file(dir, "n.gguf", "nemotron_h_moe", {{"x", {4}}})); }));
    try {
        tona::load_model(path);
        EXPECT(false);
    } catch (const std::exception& e) {
        EXPECT(std::string(e.what()).contains("qwen35moe.block_count"));
    }
    EXPECT(throws([&] { tona::load_model(write_file(dir, "g.gguf", "llama", {{"x", {4}}})); }));
    // A file of the Coder's architecture that lacks the Coder's keys fails in the module, with its prefix
    try {
        tona::load_model(write_file(dir, "h.gguf", "qwen4exp", {{"x", {4}}}));
        EXPECT(false);
    } catch (const std::exception& e) {
        EXPECT(std::string(e.what()).contains("qwen4exp.block_count"));
    }
}

}  // namespace

int main() {
    return tona::test::run("model_arch", [] {
        const tona::test::TempDir dir("model-arch");
        test_detect_arch(dir);
        test_weight_types(dir);
        test_hasher();
        test_binder(dir);
        test_mamba2(dir);
        test_moe(dir);
        test_attention_and_deltanet(dir);
        test_model_as_and_registry(dir);
    });
}
