#include "model/arch.h"

#include <algorithm>
#include <array>
#include <format>
#include <stdexcept>

namespace tona {

namespace {

struct ArchRow {
    ArchId id;
    std::string_view name;
    ArchNeeds needs;
};

// The one table of architectures: a new architecture adds one row here
constexpr std::array<ArchRow, 3> ARCHS{{
    {ArchId::Qwen4Exp, "qwen4exp", {RecurrentKind::DeltaNet, false, true, false, true, false}},
    {ArchId::Qwen35Moe, "qwen35moe", {RecurrentKind::DeltaNet, true, false, false, false, true}},
    {ArchId::NemotronHMoe, "nemotron_h_moe", {RecurrentKind::Mamba2, true, false, true, false, false}},
}};

const ArchRow& row(ArchId id) {
    for (const ArchRow& r : ARCHS)
        if (r.id == id) return r;
    throw std::logic_error("unknown ArchId");
}

}  // namespace

const ArchNeeds& arch_needs(ArchId id) { return row(id).needs; }

std::string_view arch_name(ArchId id) { return row(id).name; }

ArchId detect_arch(const Gguf& g) {
    const std::string name = g.get_string("general.architecture");
    std::string known;
    for (const ArchRow& r : ARCHS) {
        if (r.name == name) return r.id;
        known += (known.empty() ? "" : ", ") + std::string(r.name);
    }
    throw std::runtime_error(std::format("architecture {} is not supported (known: {})", name, known));
}

std::vector<WeightType> weight_types(const Gguf& g) {
    std::vector<WeightType> out;
    for (const TensorInfo& t : g.tensors()) {
        // The embedding tables are read row by row on the host (model/quant.cpp dequantizes every type there), so no
        // backend needs a kernel for their type
        if (t.name == "token_embd.weight" || t.name == "per_layer_token_embd.weight") continue;
        const bool seen = std::any_of(out.begin(), out.end(), [&](const WeightType& w) { return w.type == t.type; });
        if (!seen) out.push_back({t.type, t.name});
    }
    std::sort(out.begin(), out.end(), [](const WeightType& a, const WeightType& b) { return a.type < b.type; });
    return out;
}

}  // namespace tona
