#include "core/space.h"

#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <format>
#include <fstream>
#include <set>
#include <sstream>

#include "core/catalog.h"
#include "core/model_fetch.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

// The install step (fetch_model) takes a file with the pinned size and hashes it instead of fetching it
bool file_here(const fs::path& file, const ModelFile& f) {
    std::error_code ec;
    return file_marked_verified(file, f) || (fs::is_regular_file(file, ec) && fs::file_size(file, ec) == f.size);
}

// The folder that exists, going up from `p`
fs::path existing_part(fs::path p) {
    std::error_code ec;
    while (!p.empty() && !fs::exists(p, ec) && p != p.parent_path()) p = p.parent_path();
    return p;
}

dev_t device_of(const std::string& path) {
    struct stat st{};
    return ::stat(existing_part(path).c_str(), &st) == 0 ? st.st_dev : 0;
}

bool real_filesystem(const std::string& type) {
    static const std::set<std::string> kinds = {"ext4", "xfs", "btrfs", "f2fs", "exfat", "ntfs3", "ntfs", "zfs"};
    return kinds.contains(type);
}

// Languages the old model is strong in and the new one is not: what the swap costs
std::string weaker_in(const std::string& from, const std::string& to) {
    const offer::ModelCard *a = offer::model_card(from), *b = offer::model_card(to);
    std::string out;
    if (!a || !b) return out;
    for (const offer::Language& l : offer::languages()) {
        const offer::LanguageVerdict *va = a->verdict(l.id), *vb = b->verdict(l.id);
        if (va && vb && va->level == offer::Level::Strong && vb->level != offer::Level::Strong)
            out += (out.empty() ? "" : ", ") + l.label;
    }
    return out;
}

std::string names_of(const std::vector<std::string>& ids) {
    std::string out;
    for (const std::string& id : ids) {
        const offer::ModelCard* c = offer::model_card(id);
        out += (out.empty() ? "" : ", ") + (c ? c->name : id);
    }
    return out;
}

}  // namespace

uint64_t entry_bytes(const std::string& id) {
    const CatalogEntry* e = find_entry(id);
    uint64_t n = 0;
    if (e)
        for (const ModelFile& f : e->files) n += f.size;
    return n;
}

LoadSplit entry_split(const std::string& id, const std::string& models_dir) {
    LoadSplit n;
    const CatalogEntry* e = id.empty() ? nullptr : find_entry(id);
    if (!e) return n;
    for (const ModelFile& f : e->files)
        (!models_dir.empty() && file_here(model_dir(models_dir, *e) / f.name(), f) ? n.here : n.download) += f.size;
    return n;
}

LoadSplit load_split(const Settings& s, const std::string& models_dir) {
    std::vector<std::string> ids = models_of(s);
    if (s.drafter != "none") ids.push_back(drafter_entry_id(s.drafter, s.model));
    LoadSplit n;
    for (const std::string& id : ids) {
        const LoadSplit one = entry_split(id, models_dir);
        n.download += one.download, n.here += one.here;
    }
    return n;
}

LoadSplit model_split(const std::string& id, const std::string& models_dir) {
    LoadSplit n = entry_split(id, models_dir);
    const std::string d = default_drafter(id);
    if (d != "none") {
        const LoadSplit x = entry_split(drafter_entry_id(d, id), models_dir);
        n.download += x.download, n.here += x.here;
    }
    return n;
}

uint64_t model_bytes(const std::string& id) {
    const std::string d = default_drafter(id);
    return entry_bytes(id) + (d == "none" ? 0 : entry_bytes(drafter_entry_id(d, id)));
}

std::string load_text(const LoadSplit& n) {
    std::string t = size_text(static_cast<double>(n.download));
    if (n.here) t += std::format(" ({} already here, verified on install)", size_text(static_cast<double>(n.here)));
    return t;
}

uint64_t bytes_to_load(const Settings& s, const std::string& models_dir) { return load_split(s, models_dir).download; }

double seconds_to_load(uint64_t bytes, double mbit_per_s) {
    return static_cast<double>(bytes) * 8 / (mbit_per_s * 1e6);
}

std::string size_text(double bytes) {
    const double gb = bytes / 1e9;
    return gb >= 1000 ? std::format("{:.1f} TB", gb / 1000) : std::format("{:.1f} GB", gb);
}

std::string duration_text(double seconds) {
    const long min = std::lround(seconds / 60);
    if (min < 1) return "< 1 min";
    if (min < 60) return std::format("≈ {} min", min);
    return min % 60 == 0 ? std::format("≈ {} h", min / 60) : std::format("≈ {} h {} min", min / 60, min % 60);
}

std::vector<Place> find_places(const std::string& current_dir) {
    std::vector<Place> out;
    std::set<dev_t> seen = {device_of(current_dir)};
    std::ifstream in("/proc/mounts");
    for (std::string line; std::getline(in, line);) {
        std::istringstream f(line);
        std::string dev, mnt, type;
        f >> dev >> mnt >> type;
        // /proc/mounts writes a space in a path as \040
        for (size_t p; (p = mnt.find("\\040")) != std::string::npos;) mnt.replace(p, 4, " ");
        if (!real_filesystem(type) || mnt.starts_with("/boot") || ::access(mnt.c_str(), W_OK) != 0) continue;
        const dev_t d = device_of(mnt);
        struct statvfs st{};
        if (!seen.insert(d).second || statvfs(mnt.c_str(), &st) != 0) continue;
        out.push_back({mnt, static_cast<uint64_t>(st.f_bavail) * st.f_frsize});
    }
    return out;
}

bool fits(const Settings& s, const std::string& models_dir, uint64_t free_bytes) {
    return bytes_to_load(s, models_dir) + kBufferBytes <= free_bytes;
}

namespace {

// Smaller lists that fit, the longest first (at most two): "only the Coder, Bolt later"
void add_shorter_list(const Settings& s, const Selection& sel, const std::string& cls, const SpaceContext& ctx,
                      std::vector<SpaceOption>& out) {
    const size_t n = sel.models.size();
    for (size_t keep = n; keep-- > 1 && out.size() < 2;) {
        for (unsigned mask = 1; mask < (1U << n) && out.size() < 2; mask++) {
            if (static_cast<size_t>(__builtin_popcount(mask)) != keep) continue;
            std::vector<std::string> now, later;
            for (size_t i = 0; i < n; i++) (mask >> i & 1U ? now : later).push_back(sel.models[i]);
            Selection part;
            FlowError err;
            if (!selection_from_names({{}, sel.languages, now}, cls, part, err)) continue;
            if (!fits(apply_selection(s, part), ctx.models_dir, ctx.free_bytes)) continue;
            out.push_back(
                {std::format("{} {}, {} {}", "Download only:", names_of(part.models), "later:", names_of(later)), "",
                 part, ""});
        }
    }
}

void add_swaps(const Settings& s, const Selection& sel, const std::string& cls, const SpaceContext& ctx,
               std::vector<SpaceOption>& out) {
    for (const TaskModels& tm : task_models(sel, cls)) {
        if (!tm.alternative || !contains(sel.models, tm.recommended->id) || contains(sel.models, tm.alternative->id))
            continue;
        if (tm.alternative->size_bytes >= tm.recommended->size_bytes) continue;
        Selection swapped = sel;
        toggle_model(swapped, tm.recommended->id, cls);
        toggle_model(swapped, tm.alternative->id, cls);
        if (!fits(apply_selection(s, swapped), ctx.models_dir, ctx.free_bytes)) continue;
        const std::string weaker = weaker_in(tm.recommended->id, tm.alternative->id);
        const std::string size = size_text(static_cast<double>(tm.alternative->size_bytes));
        const std::string label =
            std::format("Replace {} by {} ({})", tm.recommended->name, tm.alternative->name, size);
        out.push_back({label, weaker.empty() ? "" : std::string("weaker in ") + weaker, swapped, ""});
    }
}

void add_places(const Settings& s, const Selection& sel, const SpaceContext& ctx, std::vector<SpaceOption>& out) {
    for (const Place& p : ctx.places) {
        const std::string dir = p.path + "/tona";
        if (!fits(apply_selection(s, sel), dir + "/models", p.free_bytes)) continue;
        out.push_back({std::format("{} {} {}", "Another place:", p.path,
                                   std::format("({} {})", size_text(static_cast<double>(p.free_bytes)), "free")),
                       "", sel, dir});
    }
}

}  // namespace

std::vector<SpaceOption> space_options(const Settings& s, const Selection& sel, const std::string& cls,
                                       const SpaceContext& ctx) {
    std::vector<SpaceOption> out;
    add_shorter_list(s, sel, cls, ctx, out);
    add_places(s, sel, ctx, out);
    add_swaps(s, sel, cls, ctx, out);
    return out;
}

}  // namespace tona::install
