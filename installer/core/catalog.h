// The models and expert profiles that tona carries. tools/gen_installer_data.py turns toolchain/models.toml
// and toolchain/profiles/ into C++ at build time, so revision, size and SHA-256 of every file are part of the signed
// program: nothing about what gets downloaded is read from the machine or a server.
#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

#include "core/plan.h"
#include "core/steps.h"

namespace tona::install {

struct ModelFile {
    std::string path;  // in the Hugging Face repository
    uint64_t size = 0;
    std::string sha256;  // 64 lowercase hex
    // Optional: a file that already is pinned in another repository (byte-equal) comes from there instead of being
    // uploaded twice; save_as renames it so the split loader finds it next to part 1 (engine/model/gguf.cpp).
    std::string repo, revision, save_as;
    std::string source_name() const { return std::filesystem::path(path).filename().string(); }
    std::string name() const { return save_as.empty() ? source_name() : save_as; }  // name on disk
};

struct CatalogEntry {
    std::string id, title, kind, license, quality, repo, revision;
    std::string mtp;  // drafter entry for --mtp: the MTP head must have the model's experts; empty: mtp-q8_0
    bool tested = false;
    std::vector<ModelFile> files;
    double size_gb() const;
    const std::string& repo_of(const ModelFile& f) const { return f.repo.empty() ? repo : f.repo; }
    const std::string& revision_of(const ModelFile& f) const { return f.revision.empty() ? revision : f.revision; }
};

// Every file has a repository and a full commit hash. An entry written before its upload carries a placeholder and
// is never fetched (models.toml, UNPUBLISHED).
bool pinned(const CatalogEntry& e);

struct ProfileBlob {
    std::string id;  // as the model id: toolchain/profiles/<id>.bin
    std::vector<uint8_t> bytes;
};

// Defined in the generated catalog_data.cpp
const std::vector<CatalogEntry>& embedded_catalog();
const std::vector<ProfileBlob>& embedded_profiles();

// Catalog entry of a drafter choice ("mtp", "dflash") for a model entry; empty for "none"
std::string drafter_entry_id(const std::string& drafter, const std::string& model);

const CatalogEntry* find_entry(const std::string& id);
const ProfileBlob* find_profile(const std::string& id);

// Folder of an entry below the models directory
std::filesystem::path model_dir(const std::string& models_dir, const CatalogEntry& e);

// All files are there with the size and a verification mark of their SHA-256 (see model_fetch.h), no hashing
bool model_present(const std::string& models_dir, const CatalogEntry& e);

// The embedded catalog in the form the wizard uses; `present` looks into models_dir
std::vector<ModelInfo> release_catalog(const std::string& models_dir);

// The catalog for the target: the signed one embedded in a release installer, tools/get_model.py in a checkout
std::vector<ModelInfo> catalog_of(const Target& t);

}  // namespace tona::install
