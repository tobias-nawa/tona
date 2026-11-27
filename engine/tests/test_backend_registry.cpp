// Backend selection: "auto" takes the first backend whose probe passes and which serves the architecture, a name takes
// only that backend, an unsupported architecture or weight type is refused with the reason, and every failure names
// the candidates
#include <cstdio>
#include <stdexcept>
#include <string>

#include "backend/registry.h"
#include "tests/test_support.h"

namespace reg = tona::backend::registry;
using tona::ArchId;

namespace {

bool no(std::string& why) {
    why = "no such device";
    return false;
}
bool yes(std::string&) { return true; }
std::unique_ptr<tona::backend::Backend> none(const tona::ModelBase&, const tona::backend::CreateOptions&) {
    return nullptr;
}
constexpr uint32_t F32 = 0, Q8_0 = 8, IQ2_XXS = 16;

std::vector<reg::ArchSupport> coder_only() { return {{ArchId::Qwen4Exp, {F32, Q8_0}, reg::Tier::Reference}}; }

// The message of the exception pick(name, arch) throws, or "" if it does not throw
std::string error_of(const char* name, ArchId arch = ArchId::Qwen4Exp) {
    try {
        reg::pick(name, arch);
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

// A model that only exists to carry an architecture and a tensor directory
class FakeModel final : public tona::ModelBase {
public:
    FakeModel(ArchId arch, const std::string& path) : ModelBase(arch, path) {}
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

// A GGUF with one tensor of one block of the given name and ggml type (zeros) and one F32 norm
std::string write_model(const tona::test::TempDir& dir, const std::string& tensor, uint32_t type) {
    tona::test::GgufWriter w;
    w.header(2, 1);
    w.kv_str("general.architecture", "qwen4exp");
    w.tensor(tensor, {tona::ggml_type_info(type)->block_elems}, type, 0);
    w.tensor("blk.0.attn_norm.weight", {4}, F32, 96);
    w.align();
    w.b.resize(w.b.size() + 96 + 16, 0);
    const std::string path = (dir.path() / ("m" + std::to_string(type) + ".gguf")).string();
    w.save(path);
    return path;
}

// Registered in order: auto skips a backend whose probe fails and takes the first that passes
void test_auto_takes_the_first_working() {
    EXPECT(error_of("auto").contains("without a backend"));
    reg::add({"first", no, coder_only(), none});
    EXPECT(error_of("auto").contains("first: no such device"));
    reg::add({"second", yes, coder_only(), none});
    reg::add({"third", yes, coder_only(), none});
    EXPECT(reg::pick("auto", ArchId::Qwen4Exp).name == "second");
    EXPECT(reg::pick("", ArchId::Qwen4Exp).name == "second");
}

void test_names_pick_one_or_list_the_candidates() {
    EXPECT(reg::pick("third", ArchId::Qwen4Exp).name == "third");
    EXPECT(error_of("first").contains("first: no such device"));
    const std::string unknown = error_of("other");
    EXPECT(unknown.contains("unknown backend \"other\""));
    EXPECT(unknown.contains("first, second, third"));
}

void test_a_name_registers_once() {
    bool twice = false;
    try {
        reg::add({"second", yes, coder_only(), none});
    } catch (const std::logic_error&) {
        twice = true;
    }
    EXPECT(twice);
    EXPECT(reg::entries().size() == 3);
}

// A backend refuses an architecture it does not list, and auto moves on to the one that lists it
void test_architecture_refusal() {
    const std::string e = error_of("auto", ArchId::NemotronHMoe);
    EXPECT(e.contains("no usable backend"));
    for (const char* name : {"first", "second", "third"})
        EXPECT(e.contains(std::string(name) + ": no support for nemotron_h_moe"));
    EXPECT(error_of("second", ArchId::Qwen35Moe).contains("second: no support for qwen35moe"));

    reg::add({"bolt-only", yes, {{ArchId::NemotronHMoe, {F32, IQ2_XXS}, reg::Tier::Maintained}}, none});
    EXPECT(reg::pick("auto", ArchId::NemotronHMoe).name == "bolt-only");
    EXPECT(reg::pick("auto", ArchId::Qwen4Exp).name == "second");  // the new entry does not take the Coder
}

// create() names the first tensor whose type the backend has no kernel for
void test_weight_type_refusal(const tona::test::TempDir& dir) {
    const FakeModel ok(ArchId::Qwen4Exp, write_model(dir, "blk.1.ffn_up_exps.weight", Q8_0));
    EXPECT(reg::create("second", ok, {}) == nullptr);  // the fake backend returns no backend, without throwing

    const FakeModel bad(ArchId::Qwen4Exp, write_model(dir, "blk.1.ffn_up_exps.weight", IQ2_XXS));
    try {
        reg::create("second", bad, {});
        EXPECT(false);
    } catch (const std::runtime_error& e) {
        EXPECT(std::string(e.what()) ==
               "blk.1.ffn_up_exps.weight is IQ2_XXS; second has no kernel for it for qwen4exp");
    }
}

// A type list built from a table (the Metal backend's kernel suffixes): exactly its types
void test_quant_set_from_a_table() {
    const reg::QuantSet s(std::set<uint32_t>{0, 16, 1019});
    EXPECT(s.contains(16) && s.contains(1019) && !s.contains(12) && s.types().size() == 3);
}

void test_support_matrix() {
    const std::string m = reg::support_matrix();
    EXPECT(m.contains("second") && m.contains("tier 1"));
    EXPECT(m.contains("bolt-only") && m.contains("nemotron_h_moe") && m.contains("tier 2") && m.contains("IQ2_XXS"));
}

}  // namespace

int main() {
    return tona::test::run("backend registry", [] {
        const tona::test::TempDir dir("backend-registry");
        test_auto_takes_the_first_working();
        test_names_pick_one_or_list_the_candidates();
        test_a_name_registers_once();
        test_architecture_refusal();
        test_weight_type_refusal(dir);
        test_quant_set_from_a_table();
        test_support_matrix();
    });
}
