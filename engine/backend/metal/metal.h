// Metal backend (Apple silicon): entry points for the registry (backend/registry.h). Plain C++; the implementation
// is Objective-C++ (backend.mm).
#pragma once
#include <cstdint>
#include <memory>
#include <set>
#include <string>

#include "backend/backend.h"
#include "backend/options.h"

namespace tona {
class Qwen4ExpModel;
}

namespace tona::metal {

bool probe(std::string& why);       // a Metal device is present
std::set<uint32_t> weight_types();  // the ggml types with a kernel (weights.h)
std::unique_ptr<backend::Backend> create(const Qwen4ExpModel& m, const backend::CreateOptions& o);

}  // namespace tona::metal
