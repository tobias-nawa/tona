// Opens a model file as the module of its architecture. This and the backend registries are the only places that
// switch over ArchId (tools/gates/arch_isolation.py).
#pragma once

#include <memory>
#include <string>

#include "model/model_base.h"

namespace tona {

// Reads general.architecture and builds the matching module; throws with the reason if the architecture is unknown or
// has no module yet, or if the file does not fit the module (missing or misshapen tensor)
std::unique_ptr<ModelBase> load_model(const std::string& path);

// For tools that work on one architecture only: load_model, then require module M (throws "this needs a qwen4exp
// model, not ..." otherwise)
template <class M>
std::unique_ptr<M> load_model_as(const std::string& path) {
    std::unique_ptr<ModelBase> base = load_model(path);
    model_as<M>(*base);
    return std::unique_ptr<M>(static_cast<M*>(base.release()));
}

}  // namespace tona
