// Backend selection at start (docs/backends/README.md#selection): every backend compiled into a program registers here,
// explicitly from main (no static initializers, so the linker drops nothing and the order is fixed). "auto" takes the
// first registered backend whose probe accepts this machine and which serves the model's architecture.
//
// A backend declares which architectures it serves, with which ggml weight types (the types it has a kernel for) and
// at which tier (docs/backends/README.md#tiers). An unsupported pair stops at load with the reason, not at the first
// matvec (docs/architecture/model-architectures.md section 3.2).
#pragma once
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <memory>
#include <set>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "backend/backend.h"
#include "backend/options.h"
#include "model/model_base.h"

namespace tona::backend::registry {

enum class Tier { Reference = 1, Maintained = 2, Experimental = 3 };

// The ggml types a backend has a kernel for, for one architecture
class QuantSet {
public:
    QuantSet(std::initializer_list<uint32_t> types) : types_(types) {}
    explicit QuantSet(std::set<uint32_t> types) : types_(std::move(types)) {}

    bool contains(uint32_t type) const { return types_.contains(type); }
    const std::set<uint32_t>& types() const { return types_; }

private:
    std::set<uint32_t> types_;
};

struct ArchSupport {
    ArchId arch;
    QuantSet types;
    Tier tier;
};

struct Entry {
    std::string name;                 // as info().name, e.g. "hip-gfx1200"
    bool (*probe)(std::string& why);  // usable on this machine; otherwise why not
    std::vector<ArchSupport> archs;
    // Builds the backend; the downcast to the architecture's own model type lives here (model_as<M>), never in pass
    // code
    std::function<std::unique_ptr<Backend>(const ModelBase&, const CreateOptions&)> create;

    const ArchSupport* support(ArchId a) const;
};

void add(Entry e);
const std::vector<Entry>& entries();

// The entry for name ("auto" or empty: first whose probe passes and which serves arch); throws std::runtime_error
// naming every candidate and why it was not taken
const Entry& pick(std::string_view name, ArchId arch);

// Throws naming the first tensor of the model whose type the entry has no kernel for (for the model's architecture)
void check_weight_types(const Entry& e, const ModelBase& m);

// pick, then check_weight_types, then the entry's create
std::unique_ptr<Backend> create(std::string_view name, const ModelBase& m, const CreateOptions& o);

// The support matrix (backend x architecture x tier x types) as text, for tona-server --list-backends
std::string support_matrix();

}  // namespace tona::backend::registry
