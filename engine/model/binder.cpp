#include "model/binder.h"

#include <format>
#include <stdexcept>

namespace tona {

std::string dims_str(const std::vector<uint64_t>& d) {
    std::string s = "[";
    for (size_t i = 0; i < d.size(); i++) s += (i ? ", " : "") + std::to_string(d[i]);
    return s + "]";
}

void Binder::fail(const std::string& what) const { throw std::runtime_error(arch_ + ": " + what); }

T Binder::need(const std::string& name, const std::vector<uint64_t>& dims) {
    T t = g_.find(name);
    if (!t) fail(std::format("tensor {} is missing", name));
    if (t->dims != dims) fail(std::format("{} has shape {}, expected {}", name, dims_str(t->dims), dims_str(dims)));
    used_.insert(name);
    return t;
}

T Binder::optional(const std::string& name, const std::vector<uint64_t>& dims) {
    return g_.find(name) ? need(name, dims) : nullptr;
}

void Binder::expect_all_used(const std::string& what) const {
    for (const auto& t : g_.tensors())
        if (!used_.contains(t.name)) fail(std::format("{}unexpected tensor {} {}", what, t.name, dims_str(t.dims)));
}

uint32_t Binder::u32(const std::string& key) const {
    const uint64_t v = g_.get_uint(key);
    if (v > UINT32_MAX) fail(std::format("{} = {} is too large", key, v));
    return (uint32_t)v;
}

}  // namespace tona
