// Fetching a model at its pinned revision, with the same checks as tools/get_model.py: every file has the size and the
// SHA-256 of the signed catalog (catalog.h), a download is resumed from its .part file, a file is hashed once and then
// carries a verification mark next to it (same JSON as get_model.py, so both tools accept each other's files).
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>

#include "core/catalog.h"
#include "core/steps.h"
#include "update/fetch.h"

namespace tona::install {

struct ModelSource {
    std::string base = "https://huggingface.co";
    update::FetchConfig fetch;  // tests allow loopback http here
    int attempts = 5;           // each continues where the last one stopped
};

std::string model_url(const ModelSource& src, const CatalogEntry& e, const ModelFile& f);

// SHA-256 of a whole file as hex; empty if it cannot be read
// `on_bytes` gets the bytes hashed so far, once per chunk
std::string sha256_of_file(const std::filesystem::path& file, const std::function<void(uint64_t)>& on_bytes = {});

// The mark says: this file (size, mtime) has this SHA-256. No hashing.
bool file_marked_verified(const std::filesystem::path& file, const ModelFile& want);

// Fetches all files of `e` into <models_dir>/<id>/ and verifies them. Files found in `from_dir` (may be empty) are
// verified and linked instead. Returns an error text, empty on success. Nothing but a complete, verified file ever
// has its final name.
std::string fetch_model(const CatalogEntry& e, const std::string& models_dir, const ModelSource& src,
                        const std::string& from_dir, const LineFn& out);

}  // namespace tona::install
