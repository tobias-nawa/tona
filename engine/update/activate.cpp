#include "update/activate.h"

#include <algorithm>
#include <charconv>
#include <set>
#include <vector>

#include "update/fsutil.h"

namespace fs = std::filesystem;

namespace tona::update {

namespace {

constexpr std::string_view kVersionsPrefix = "versions/";

bool parse_version(std::string_view v, int (&out)[3]) {
    for (int& part : out) {
        const size_t dot = v.find('.');
        const std::string_view tok = v.substr(0, dot);
        const auto r = std::from_chars(tok.data(), tok.data() + tok.size(), part);
        if (tok.empty() || r.ec != std::errc() || r.ptr != tok.data() + tok.size()) return false;
        v = dot == std::string_view::npos ? std::string_view() : v.substr(dot + 1);
    }
    return v.empty();
}

void remove_quietly(const fs::path& p) {
    std::error_code ec;
    fs::remove_all(p, ec);
}

std::set<std::string> used_runtimes(const Layout& layout) {
    std::set<std::string> used;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(layout.versions(), ec)) {
        const std::string target = read_link((e.path() / "lib").string());
        used.insert(fs::path(target).filename().string());
    }
    return used;
}

void prune_runtimes(const Layout& layout) {
    const std::set<std::string> used = used_runtimes(layout);
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(layout.runtime_parent(), ec)) {
        if (!used.contains(e.path().filename().string())) remove_quietly(e.path());
    }
}

}  // namespace

bool version_less(std::string_view a, std::string_view b) {
    int x[3], y[3];
    if (!parse_version(a, x) || !parse_version(b, y)) return false;
    return std::lexicographical_compare(std::begin(x), std::end(x), std::begin(y), std::end(y));
}

std::string linked_version(const Layout& layout, const std::string& link_name) {
    const std::string target = read_link((layout.root / link_name).string());
    if (!target.starts_with(kVersionsPrefix)) return "";
    return target.substr(kVersionsPrefix.size());
}

void activate_version(const Layout& layout, const std::string& version) {
    const std::string old = linked_version(layout, "current");
    if (!old.empty() && old != version)
        set_symlink_atomic(std::string(kVersionsPrefix) + old, layout.previous().string());
    set_symlink_atomic(std::string(kVersionsPrefix) + version, layout.current().string());
}

bool swap_current_previous(const Layout& layout) {
    const std::string cur = linked_version(layout, "current");
    const std::string prev = linked_version(layout, "previous");
    if (prev.empty() || !fs::is_directory(layout.version(prev))) return false;
    // previous first: if we die in between, `current` is still the version that ran before
    if (!cur.empty()) set_symlink_atomic(std::string(kVersionsPrefix) + cur, layout.previous().string());
    set_symlink_atomic(std::string(kVersionsPrefix) + prev, layout.current().string());
    return true;
}

void prune(const Layout& layout) {
    const std::string cur = linked_version(layout, "current");
    const std::string prev = linked_version(layout, "previous");
    std::vector<std::string> others;
    std::error_code ec;
    for (const auto& e : fs::directory_iterator(layout.versions(), ec)) {
        const std::string name = e.path().filename().string();
        if (name != cur && name != prev) others.push_back(name);
    }
    std::sort(others.begin(), others.end(), [](const auto& a, const auto& b) { return version_less(b, a); });
    const size_t protected_count = (cur.empty() ? 0 : 1) + (prev.empty() || prev == cur ? 0 : 1);
    const size_t keep = kKeepVersions > protected_count ? kKeepVersions - protected_count : 0;
    for (size_t i = keep; i < others.size(); i++) remove_quietly(layout.version(others[i]));
    prune_runtimes(layout);
}

}  // namespace tona::update
