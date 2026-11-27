#include "backend/registry.h"

#include <format>
#include <stdexcept>

namespace tona::backend::registry {

namespace {

std::vector<Entry>& list() {
    static std::vector<Entry> l;
    return l;
}

const char* tier_name(Tier t) {
    switch (t) {
        case Tier::Reference:
            return "tier 1";
        case Tier::Maintained:
            return "tier 2";
        case Tier::Experimental:
            return "tier 3";
    }
    return "?";
}

std::string type_name(uint32_t type) { return ggml_type_info(type)->name; }

}  // namespace

const ArchSupport* Entry::support(ArchId a) const {
    for (const ArchSupport& s : archs)
        if (s.arch == a) return &s;
    return nullptr;
}

void add(Entry e) {
    for (const Entry& x : list())
        if (x.name == e.name) throw std::logic_error("backend registered twice: " + e.name);
    list().push_back(std::move(e));
}

const std::vector<Entry>& entries() { return list(); }

const Entry& pick(std::string_view name, ArchId arch) {
    const bool any = name.empty() || name == "auto";
    std::string tried;
    for (const Entry& e : list()) {
        if (!any && e.name != name) continue;
        std::string why;
        if (!e.support(arch))
            tried += std::format("\n  {}: no support for {}", e.name, arch_name(arch));
        else if (!e.probe(why))
            tried += "\n  " + e.name + ": " + why;
        else
            return e;
    }
    if (list().empty()) throw std::runtime_error("this program was built without a backend");
    if (tried.empty()) {
        std::string names;
        for (const Entry& e : list()) names += (names.empty() ? "" : ", ") + e.name;
        throw std::runtime_error("unknown backend \"" + std::string(name) + "\" (built in: " + names + ")");
    }
    throw std::runtime_error("no usable backend:" + tried);
}

void check_weight_types(const Entry& e, const ModelBase& m) {
    const ArchSupport* s = e.support(m.arch());
    if (!s) throw std::runtime_error(std::format("{}: no support for {}", e.name, arch_name(m.arch())));
    for (const WeightType& w : weight_types(m.gguf()))
        if (!s->types.contains(w.type))
            throw std::runtime_error(std::format("{} is {}; {} has no kernel for it for {}", w.first_tensor,
                                                 type_name(w.type), e.name, arch_name(m.arch())));
}

std::unique_ptr<Backend> create(std::string_view name, const ModelBase& m, const CreateOptions& o) {
    const Entry& e = pick(name, m.arch());
    check_weight_types(e, m);
    return e.create(m, o);
}

std::string support_matrix() {
    std::string out;
    for (const Entry& e : list())
        for (const ArchSupport& s : e.archs) {
            std::string types;
            for (uint32_t t : s.types.types()) types += (types.empty() ? "" : " ") + type_name(t);
            out += std::format("{:<12} {:<16} {}  types: {}\n", e.name, arch_name(s.arch), tier_name(s.tier), types);
        }
    return out;
}

}  // namespace tona::backend::registry
