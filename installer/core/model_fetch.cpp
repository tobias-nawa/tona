#include "core/model_fetch.h"

#include <sys/stat.h>
#include <sys/statvfs.h>

#include <algorithm>
#include <cstdio>
#include <format>
#include <fstream>
#include <vector>

#include "core/progress.h"
#include "core/space.h"
#include "nlohmann/json.hpp"
#include "update/crypto.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

constexpr size_t kChunk = 16 << 20;
constexpr int kTransferSeconds = 24 * 3600;  // 29 GB on a slow line; a stall is caught by the read timeout

// Bytes of the whole model per phase, for the progress of the screen: `base` is what earlier files already contributed
struct Totals {
    uint64_t dl_base = 0, dl_total = 0, vf_base = 0, vf_total = 0;
};

fs::path mark_of(const fs::path& file) { return file.parent_path() / (file.filename().string() + ".sha256ok"); }

nlohmann::json mark_value(const fs::path& resolved, const ModelFile& want) {
    struct stat st{};
    if (::stat(resolved.c_str(), &st) != 0) return nullptr;
#ifdef __APPLE__
    const timespec& mt = st.st_mtimespec;
#else
    const timespec& mt = st.st_mtim;
#endif
    const int64_t mtime_ns = static_cast<int64_t>(mt.tv_sec) * 1000000000 + mt.tv_nsec;
    return {{"size", static_cast<int64_t>(st.st_size)}, {"mtime_ns", mtime_ns}, {"sha256", want.sha256}};
}

// Hashes the file once and writes the mark (next to `marked`, which may be a link to `file`)
bool verify_and_mark(const fs::path& file, const fs::path& marked, const ModelFile& want, const LineFn& out,
                     Totals& tot) {
    std::error_code ec;
    const fs::path resolved = fs::canonical(file, ec);
    if (ec || fs::file_size(resolved, ec) != want.size) return false;
    out("  verifying SHA-256 of " + want.name() + " ...");
    const auto step = [&](uint64_t n) { report_progress({Phase::Verify, tot.vf_base + n, tot.vf_total}); };
    if (sha256_of_file(resolved, step) != want.sha256) return false;
    tot.vf_base += want.size;
    std::ofstream(mark_of(marked)) << mark_value(resolved, want).dump();
    return true;
}

// Reads from `have` on into the .part file; returns an error text, empty when the file is complete
std::string download_rest(const ModelSource& src, const std::string& url, const fs::path& part, const ModelFile& f,
                          const LineFn& out, const Totals& tot) {
    std::error_code ec;
    uint64_t have = fs::exists(part) ? fs::file_size(part, ec) : 0;
    if (have > f.size) fs::remove(part, ec), have = 0;
    if (have == f.size) return "";
    std::ofstream file(part, std::ios::binary | std::ios::app);
    if (!file) return "cannot write " + part.string();
    update::FetchRequest req{url, 0, f.size - have, kTransferSeconds, have};
    int shown = -1;
    const auto r = update::fetch(
        src.fetch, req,
        [&](const char* d, size_t n) { return static_cast<bool>(file.write(d, static_cast<std::streamsize>(n))); },
        [&](uint64_t so_far) {
            report_progress({Phase::Download, tot.dl_base + have + so_far, tot.dl_total});
            const int pct = static_cast<int>((have + so_far) * 50 / f.size);  // a line per 2 %
            if (pct == shown) return;
            shown = pct;
            out(std::format("  {}: {} / {}", f.name(), size_text(static_cast<double>(have + so_far)),
                            size_text(static_cast<double>(f.size))));
        });
    file.close();
    return r.ok ? "" : r.error;
}

std::string fetch_file(const CatalogEntry& e, const ModelFile& f, const fs::path& dir, const ModelSource& src,
                       const LineFn& out, Totals& tot) {
    const fs::path dest = dir / f.name(), part = fs::path(dest.string() + ".part");
    const std::string url = model_url(src, e, f);
    std::string err;
    for (int attempt = 1; attempt <= src.attempts; attempt++) {
        out(std::format("fetching {} ({}){}", f.name(), size_text(static_cast<double>(f.size)),
                        attempt > 1 ? ", resuming" : ""));
        err = download_rest(src, url, part, f, out, tot);
        if (err.empty()) break;
        out("  interrupted: " + err);
    }
    if (!err.empty()) return f.name() + ": " + err;
    std::error_code ec;
    tot.dl_base += f.size;
    if (!verify_and_mark(part, part, f, out, tot)) {
        fs::remove(part, ec);
        return f.name() + ": size or SHA-256 does not match the pinned revision (the partial file was removed)";
    }
    fs::rename(part, dest, ec);
    if (ec) return f.name() + ": " + ec.message();
    fs::rename(mark_of(part), mark_of(dest), ec);  // the mark was made for the .part name; same file, same bytes
    return ec ? f.name() + ": " + ec.message() : "";
}

std::string link_file(const ModelFile& f, const fs::path& src, const fs::path& dest, const LineFn& out, Totals& tot) {
    if (!verify_and_mark(src, dest, f, out, tot)) return src.string() + ": does not match the pinned revision";
    std::error_code ec;
    fs::remove(dest, ec);
    fs::create_symlink(fs::canonical(src), dest, ec);
    if (ec) return dest.string() + ": " + ec.message();
    out("linked  " + f.name() + " -> " + fs::canonical(src).string());
    return "";
}

uint64_t missing_bytes(const CatalogEntry& e, const fs::path& dir) {
    uint64_t need = 0;
    std::error_code ec;
    for (const ModelFile& f : e.files) {
        if (file_marked_verified(dir / f.name(), f)) continue;
        const fs::path part = fs::path((dir / f.name()).string() + ".part");
        const uint64_t have = fs::exists(part) ? fs::file_size(part, ec) : 0;
        need += f.size - std::min(have, f.size);
    }
    return need;
}

// Every file not yet marked is hashed once; one not on disk (and not in `from_dir`) is downloaded first
Totals totals_of(const CatalogEntry& e, const fs::path& dir, const std::string& from_dir) {
    Totals t;
    for (const ModelFile& f : e.files) {
        if (file_marked_verified(dir / f.name(), f)) continue;
        t.vf_total += f.size;
        const fs::path a = fs::path(from_dir) / f.name(), b = fs::path(from_dir) / f.source_name();
        const bool local = fs::exists(dir / f.name()) || (!from_dir.empty() && (fs::exists(a) || fs::exists(b)));
        if (!local) t.dl_total += f.size;
    }
    return t;
}

}  // namespace

std::string model_url(const ModelSource& src, const CatalogEntry& e, const ModelFile& f) {
    return src.base + "/" + e.repo_of(f) + "/resolve/" + e.revision_of(f) + "/" + f.path;
}

std::string sha256_of_file(const fs::path& file, const std::function<void(uint64_t)>& on_bytes) {
    std::ifstream in(file, std::ios::binary);
    if (!in) return {};
    update::Sha256 h;
    std::vector<char> buf(kChunk);
    uint64_t hashed = 0;
    while (in.read(buf.data(), static_cast<std::streamsize>(buf.size())) || in.gcount() > 0) {
        h.update(buf.data(), static_cast<size_t>(in.gcount()));
        hashed += static_cast<uint64_t>(in.gcount());
        if (on_bytes) on_bytes(hashed);
    }
    const auto d = h.finish();
    return update::to_hex(d.data(), d.size());
}

bool file_marked_verified(const fs::path& file, const ModelFile& want) {
    std::error_code ec;
    const fs::path resolved = fs::canonical(file, ec);
    if (ec) return false;
    std::ifstream in(mark_of(file));
    if (!in) return false;
    const auto got = nlohmann::json::parse(in, nullptr, false);
    return !got.is_discarded() && got == mark_value(resolved, want);
}

std::string fetch_model(const CatalogEntry& e, const std::string& models_dir, const ModelSource& src,
                        const std::string& from_dir, const LineFn& out) {
    if (!pinned(e)) return e.id + ": not published yet (no pinned repository and revision in this installer)";
    const fs::path dir = model_dir(models_dir, e);
    std::error_code ec;
    fs::create_directories(dir, ec);
    if (ec) return dir.string() + ": " + ec.message();
    struct statvfs vfs{};
    const uint64_t need = missing_bytes(e, dir);
    if (::statvfs(dir.c_str(), &vfs) == 0 && static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize < need)
        return std::format("not enough free space in {}: {} needed", dir.string(),
                           size_text(static_cast<double>(need)));
    Totals tot = totals_of(e, dir, from_dir);
    for (const ModelFile& f : e.files) {
        const fs::path dest = dir / f.name();
        if (file_marked_verified(dest, f) || (fs::exists(dest) && verify_and_mark(dest, dest, f, out, tot))) {
            out("ok      " + f.name());
            continue;
        }
        // an existing download may carry the name on disk or the name in its repository
        fs::path from = fs::path(from_dir) / f.name();
        if (!fs::exists(from)) from = fs::path(from_dir) / f.source_name();
        const std::string err = !from_dir.empty() && fs::exists(from) ? link_file(f, from, dest, out, tot)
                                                                      : fetch_file(e, f, dir, src, out, tot);
        if (!err.empty()) return err;
    }
    return "";
}

}  // namespace tona::install
