// Directory layout of a release installation (see docs/use/update.md)
#pragma once

#include <filesystem>
#include <string>

namespace tona::update {

struct Layout {
    std::filesystem::path root;  // ~/.local/share/tona

    std::filesystem::path versions() const { return root / "versions"; }
    std::filesystem::path version(const std::string& v) const { return versions() / v; }
    std::filesystem::path runtime(const std::string& lock_id) const { return root / "runtime" / ("rocm-" + lock_id); }
    std::filesystem::path runtime_parent() const { return root / "runtime"; }
    std::filesystem::path staging() const { return root / "staging"; }
    std::filesystem::path current() const { return root / "current"; }
    std::filesystem::path previous() const { return root / "previous"; }
    std::filesystem::path state() const { return root / "state.json"; }
    std::filesystem::path lock() const { return root / ".lock"; }
};

}  // namespace tona::update
