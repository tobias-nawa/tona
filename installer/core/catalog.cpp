#include "core/catalog.h"

#include "core/model_fetch.h"

namespace fs = std::filesystem;

namespace tona::install {

double CatalogEntry::size_gb() const {
    double n = 0;
    for (const ModelFile& f : files) n += static_cast<double>(f.size);
    return n / 1e9;
}

std::string drafter_entry_id(const std::string& drafter, const std::string& model) {
    if (drafter == "mtp") {
        const CatalogEntry* m = find_entry(model);
        return m && !m->mtp.empty() ? m->mtp : "mtp-q8_0";
    }
    return drafter == "dflash" ? "dflash-q8_0" : "";
}

bool pinned(const CatalogEntry& e) {
    const auto is_commit = [](const std::string& r) {
        return r.size() == 40 && r.find_first_not_of("0123456789abcdef") == std::string::npos;
    };
    for (const ModelFile& f : e.files)
        if (!e.repo_of(f).contains('/') || !is_commit(e.revision_of(f))) return false;
    return !e.files.empty();
}

const CatalogEntry* find_entry(const std::string& id) {
    for (const CatalogEntry& e : embedded_catalog())
        if (e.id == id) return &e;
    return nullptr;
}

const ProfileBlob* find_profile(const std::string& id) {
    for (const ProfileBlob& p : embedded_profiles())
        if (p.id == id) return &p;
    return nullptr;
}

fs::path model_dir(const std::string& models_dir, const CatalogEntry& e) { return fs::path(models_dir) / e.id; }

bool model_present(const std::string& models_dir, const CatalogEntry& e) {
    for (const ModelFile& f : e.files)
        if (!file_marked_verified(model_dir(models_dir, e) / f.name(), f)) return false;
    return true;
}

std::vector<ModelInfo> release_catalog(const std::string& models_dir) {
    std::vector<ModelInfo> out;
    for (const CatalogEntry& e : embedded_catalog()) {
        ModelInfo m;
        m.id = e.id, m.title = e.title, m.kind = e.kind, m.license = e.license, m.quality = e.quality;
        m.first_file = e.files.front().name(), m.size_gb = e.size_gb(), m.tested = e.tested;
        m.present = model_present(models_dir, e);
        out.push_back(std::move(m));
    }
    return out;
}

std::vector<ModelInfo> catalog_of(const Target& t) {
    return t.release ? release_catalog(t.models_dir) : load_catalog(t.repo);
}

}  // namespace tona::install
