// Shape-checked tensor binding shared by every architecture module. Rule of the engine: an unknown tensor or a wrong
// shape stops the load, so a model that is not fully understood never runs.
#pragma once

#include <cstdint>
#include <set>
#include <string>
#include <vector>

#include "model/gguf.h"

namespace tona {

using T = const TensorInfo*;

std::string dims_str(const std::vector<uint64_t>& d);

class Binder {
public:
    // arch is the message prefix of every error ("qwen4exp: tensor x is missing")
    Binder(const Gguf& g, std::string arch) : g_(g), arch_(std::move(arch)) {}

    // Shapes in ggml order: matrices are [input, output]. Throws if missing or of another shape.
    T need(const std::string& name, const std::vector<uint64_t>& dims);
    // The tensor if present (then its shape must match, as in need()), else null
    T optional(const std::string& name, const std::vector<uint64_t>& dims);
    // Marks a tensor as used without binding it (tensors the engine reads by name elsewhere)
    void mark_used(const std::string& name) { used_.insert(name); }
    // Throws on the first tensor of the file that no binder asked for; what is the message prefix after the arch name
    void expect_all_used(const std::string& what = "") const;

    const Gguf& gguf() const { return g_; }
    const std::string& arch() const { return arch_; }
    // "<arch>: <what>" as an exception message
    [[noreturn]] void fail(const std::string& what) const;
    void check(bool ok, const std::string& what) const {
        if (!ok) fail(what);
    }
    // Metadata as u32; throws if the key is missing or does not fit
    uint32_t u32(const std::string& key) const;

private:
    const Gguf& g_;
    std::string arch_;
    std::set<std::string> used_;
};

}  // namespace tona
