// Prompt cache: stashed conversations in RAM and checkpoint files on the SSD

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <format>
#include <fstream>

#include "backend/budget.h"
#include "backend/stall.h"
#include "server/checkpoint_file.h"
#include "server/generator.h"

namespace tona::server {

namespace {

// Fingerprint of all tensors (name, type, shape, offset), what else the model says changes its tokens or states
// (ModelBase::fingerprint) and the state sizes. Another model, another quantization or another context size gives
// another value (KV in RAM or VRAM does not change the content).
uint64_t model_fingerprint(const ModelBase& m, const backend::Backend& mg) {
    Hasher h;
    for (const TensorInfo& t : m.gguf().tensors()) {
        h.bytes(t.name.data(), t.name.size());
        h.value(t.type);
        h.bytes(t.dims.data(), t.dims.size() * sizeof(uint64_t));
        h.value(t.offset);
    }
    m.fingerprint(h);
    // Sizes and layout of the states: if a commit changes the KV row or the states, old files no longer fit
    const uint64_t sizes[4] = {mg.state_bytes(), mg.dring_bytes(), mg.kv_row_bytes(), DISK_VERSION};
    h.bytes(sizes, sizeof sizes);
    return h.digest();
}

void protect_file(const std::string& path) {
    std::error_code ec;
    std::filesystem::permissions(path, std::filesystem::perms::owner_read | std::filesystem::perms::owner_write,
                                 std::filesystem::perm_options::replace, ec);
}

}  // namespace

void Generator::disk_scan() {
    fingerprint_ = model_fingerprint(m_, mg_);
    std::error_code ec;
    const bool created = std::filesystem::create_directories(disk_dir_, ec);
    if (std::string why; !secure_checkpoint_dir(disk_dir_, created, why)) {
        fprintf(stderr, "Error: SSD prompt cache off: %s is %s\n", disk_dir_.c_str(), why.c_str());
        disk_dir_.clear();
        return;
    }
    for (const auto& f : std::filesystem::directory_iterator(disk_dir_, ec)) {
        if (f.path().extension() != ".bin" || f.is_symlink()) continue;
        OnDisk d{f.path().string(), {}, true};
        if (read_checkpoint_tokens(d.path, fingerprint_, d.tokens, d.logits))
            protect_file(d.path), disk_.push_back(std::move(d));
    }
    fprintf(stderr, "Checkpoints on the SSD (%s): %zu matching\n", disk_dir_.c_str(), disk_.size());
}

// Makes room: deletes the longest unused files (modification time, renewed on read) until the limit holds
void Generator::disk_prune(const std::string& keep_path) {
    std::vector<std::pair<std::filesystem::file_time_type, std::string>> files;
    uintmax_t total = 0;
    std::error_code ec;
    for (const auto& f : std::filesystem::directory_iterator(disk_dir_, ec))
        if (f.path().extension() == ".bin")
            files.emplace_back(f.last_write_time(), f.path().string()), total += f.file_size();
    std::sort(files.begin(), files.end());
    for (const auto& [t, p] : files) {
        if (total <= disk_gb_ * 1e9 || p == keep_path) break;
        total -= std::filesystem::file_size(p, ec);
        std::filesystem::remove(p, ec);
        std::erase_if(disk_, [&](const OnDisk& d) { return d.path == p; });
    }
}

void Generator::disk_write(const Stashed& s) {
    if (disk_dir_.empty()) return;
    const auto& h = s.snap;
    const std::string path = checkpoint_path(disk_dir_, fingerprint_, h.tokens);
    // The name is a 64-bit hash of the tokens: an existing valid file under it is the same prompt (nothing to do) or a
    // collision (must not be overwritten, a client could aim at another prompt's checkpoint)
    for (const OnDisk& d : disk_)
        if (d.path == path || d.tokens == h.tokens) return;
    if (!write_checkpoint(path + ".tmp", fingerprint_, s.logits, h)) {
        fprintf(stderr, "Checkpoint not written: %s\n", path.c_str());
        std::filesystem::remove(path + ".tmp");
        return;
    }
    std::error_code rc;
    std::filesystem::rename(path + ".tmp", path, rc);
    if (rc) {
        fprintf(stderr, "Checkpoint not renamed: %s (%s)\n", path.c_str(), rc.message().c_str());
        std::filesystem::remove(path + ".tmp", rc);
        return;
    }
    disk_.push_back({path, h.tokens, !s.logits.empty()});
    disk_prune(path);
}

// Checks the head and all sizes before anything is allocated (broken or foreign file: false, the caller deletes it)
bool Generator::disk_read(const OnDisk& d, Stashed& s) const {
    if (!read_checkpoint(d.path, {fingerprint_, d.tokens, m_.dims().vocab, mg_.state_bytes()}, s.logits, s.snap) ||
        !mg_.snap_fits(s.snap))
        return false;
    std::error_code ec;
    std::filesystem::last_write_time(d.path, std::filesystem::file_time_type::clock::now(), ec);
    return true;
}

// Checkpoint of the running conversation into RAM; short conversations are not worth it (recomputed in under 1 s).
// A stashed end replaces older ends it continues and inherits their count: the main history keeps its hits, while
// the changing ends of review requests (they continue no older end) stay at 0 and are evicted first.
void Generator::stash_current(bool ckpt, bool tail) {
    if (slots_ < 2 || !mg_.has_state() || mg_.snap_tokens().size() < 1024) return;
    const auto& t = mg_.snap_tokens();
    for (Stashed& s : stash_)
        if (s.snap.tokens == t) {
            s.used = ++clock_, s.ckpt |= ckpt;
            return;
        }
    const double need = mg_.stash_bytes() + snap_logits().size() * sizeof(float);
    if (!can_stash(need)) return;  // before the older ends are dropped: they stay if the new one cannot be kept
    int hits = 0;
    std::erase_if(stash_, [&](const Stashed& s) {
        const bool old = (!s.ckpt || (tail && s.tail)) && s.used != keep_ && s.snap.tokens.size() < t.size() &&
                         std::equal(s.snap.tokens.begin(), s.snap.tokens.end(), t.begin());
        if (old) hits = std::max(hits, s.hits);
        return old;
    });
    make_room(need);
    Stashed* dst = &stash_.emplace_back();
    dst->ckpt = ckpt, dst->tail = tail, dst->hits = hits;
    const auto t0 = std::chrono::steady_clock::now();
    try {
        mg_.stash(dst->snap);
    } catch (const backend::CopyStalled& e) {
        // The copy stalled and the backend released the stash (backend/stall.h); the conversation is computed again
        // when it comes back, as if it had not fitted (can_stash)
        fprintf(stderr, "Conversation of %zu tokens not stashed: %s\n", mg_.snap_tokens().size(), e.what());
        stash_.pop_back();
        return;
    } catch (...) {
        stash_.pop_back();  // whatever else went wrong, a snapshot that may be incomplete is not kept
        throw;
    }
    const double stash_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    dst->logits = snap_logits();
    const auto& h = dst->snap;
    fprintf(stderr,
            "Conversation of %zu tokens stashed: %.0f MB (state %.0f, KV %.0f, pool %.0f, drafter %.0f) in %.2f s\n",
            h.tokens.size(), h.bytes() / 1e6, h.state.size() / 1e6, h.kv.size() / 1e6, h.pool.size() / 1e6,
            h.dring.size() / 1e6, stash_s);
    dst->used = ++clock_;
}

// Protected is what was fetched before and used in the last 16 operations (stash, fetch; ~2-3 per request);
// otherwise entries of old conversations with hits would stay forever
bool Generator::evict_one() {
    auto prot = [&](const Stashed& s) { return s.hits > 0 && clock_ - s.used <= 16; };
    auto worse = [&](const Stashed& a, const Stashed& b) {
        return std::pair(prot(a), a.used) < std::pair(prot(b), b.used);
    };
    auto victim = stash_.end();
    for (auto it = stash_.begin(); it != stash_.end(); ++it)
        if (it->used != keep_ && (victim == stash_.end() || worse(*it, *victim))) victim = it;
    if (victim == stash_.end()) return false;
    stash_.erase(victim);
    return true;
}

// All entries, or with `evictable` only those evict_one may drop (not the one being fetched, keep_)
double Generator::stash_held_bytes(bool evictable) const {
    double b = 0;
    for (const Stashed& s : stash_)
        if (!evictable || s.used != keep_) b += (double)s.snap.bytes() + s.logits.size() * sizeof(float);
    return b;
}

void Generator::trim_stash() {
    while (!stash_.empty() && ((int)stash_.size() > std::max(0, slots_ - 1) || stash_held_bytes() > ram_gb_ * 1e9) &&
           evict_one()) {
    }
}

// Decides before the copy is made whether a stash of `bytes` fits (backend::stash_fits). A conversation that does not
// fit is not stored at all: its next request is computed from scratch (a stash is a cache, and an OOM kill loses every
// conversation). The reason is logged once per server run.
bool Generator::can_stash(double bytes) {
    // The entry being fetched cannot make room: counting it would let a stash through that leaves less than the reserve
    const double held = stash_held_bytes(true), free = backend::free_ram_bytes();
    if (!backend::stash_fits(bytes, ram_gb_, free, held)) {
        if (!stash_skip_logged_)
            fprintf(
                stderr,
                "Conversation of %zu tokens not stashed: needs %.1f GB, budget %.1f GB (TONA_PROMPT_RAM_GB), free RAM "
                "%.1f GB; further skips are not logged\n",
                mg_.snap_tokens().size(), bytes / 1e9, ram_gb_, free / 1e9);
        stash_skip_logged_ = true;
        return false;
    }
    return true;
}

// Drops older entries until `bytes` more fit the budget and the free RAM
void Generator::make_room(double bytes) {
    // The entry being fetched (keep_) stays; evict_one reports when nothing else is left
    while ((stash_held_bytes() + bytes > ram_gb_ * 1e9 ||
            bytes + backend::stash_reserve_gb() * 1e9 > backend::free_ram_bytes()) &&
           evict_one()) {
    }
}

}  // namespace tona::server
