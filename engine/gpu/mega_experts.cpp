// Expert placement at start-up: the expert profile, the initial contents of the VRAM cache and the pinned stage in
// host RAM.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <numeric>
#include <string>
#include <string_view>

#include "backend/budget.h"
#include "backend/env.h"
#include "backend/expert_profile.h"
#include "backend/expert_tiers.h"
#include "backend/omp_guard.h"
#include "gpu/expert_parts.h"
#include "gpu/gtt.h"
#include "gpu/mega.h"
#include "gpu/mega_util.h"
#include "model/progress.h"

namespace tona::gpu {

// TONA_EXPERT_PROFILE=file: use counts per (layer, expert), counted per token (see write_profile for the format)
void Mega::load_profile() {
    const char* path = env::text("TONA_EXPERT_PROFILE");
    if (!path) return;
    profile_ = backend::read_expert_profile(path, experts_, "gpu::Mega");
}

// Fill the VRAM cache before the first pass with the experts of each layer that the profile counts most often
void Mega::prefill_cache() {
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        LayerCache& C = cache_[l];
        const MoeWeights& S = m_.moe_weights(l);
        const int E = (int)experts_.experts(l);
        const std::vector<int> ord = backend::experts_by_count(profile_.data() + experts_.offset(l), E);
        const ExpertParts P = expert_parts(S, C);
        const auto& parts = P.t;
        const auto& offs = P.off;
        std::vector<uint8_t> buf(C.slot_bytes * 16);
        const int n = std::min<int>(C.n_slots, (int)C.free_slots.size());
        for (int i0 = 0; i0 < n; i0 += 16) {
            const int m = std::min(16, n - i0);
            backend::OmpErrors err;  // a failing pread must reach the caller, not end the process
#pragma omp parallel for collapse(2) schedule(static)
            for (int i = 0; i < m; i++)
                for (int k = 0; k < P.n; k++)
                    err.run([&] {
                        m_.gguf().read(*parts[k], (size_t)ord[i0 + i] * bytes_per_expert(parts[k]),
                                       buf.data() + i * C.slot_bytes + offs[k], bytes_per_expert(parts[k]));
                    });
            err.rethrow();
            for (int i = 0; i < m; i++) {
                const int e = ord[i0 + i], slot = C.free_slots.back();
                C.free_slots.pop_back();
                elastic_.require_live(C, (int)l, slot);
                HIPCHECK(hipMemcpy(C.slots + (size_t)slot * C.slot_bytes, buf.data() + i * C.slot_bytes, C.slot_bytes,
                                   hipMemcpyHostToDevice));
                C.add_back(e, slot);  // the most frequent first, so the least frequent leave first
            }
        }
        progress::add(1);
    }
}

// With a profile (unless TONA_TIERS=0): pin by usage when at least 70 % of the experts fit (backend/expert_tiers.h). On
// by default since 2026-11-15: +40 to +100 % decode and +70 to +90 % prefill at the 26 GB class, nothing pinned (as
// before) where less than 70 % fit (research/measurements/2026-11-15-expert-tiers.md).

namespace {
// The context whose KV in RAM is counted as filled later with TONA_TIERS: the installer's context below 32 GB, decided
// with the page-cache floor (backend::TIERED_CACHE_MIN_GB) by the sweep in
// research/measurements/2026-11-15-expert-tiers.md
constexpr double kTiersKvCtx = 65536;
bool tiers_on(const std::vector<uint32_t>& profile) { return env::integer("TONA_TIERS", 1) != 0 && !profile.empty(); }
}  // namespace

// Bytes of all experts of all layers
size_t Mega::all_expert_bytes() const {
    size_t all = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) all += cache_[l].slot_bytes * experts_.experts(l);
    return all;
}

// Bytes the pinned stage may take with TONA_RAM_GB: the limit minus what is already held (buffers, KV in RAM), the
// slack, the minimum for stashed sequences and the page cache floor. Without TONA_TIERS that is all or none
// (backend/budget.h); with it a part counts if it is at least backend::MIN_PARTIAL_PIN_FRACTION of the experts.
// TONA_TIERS_CACHE_GB: the page cache floor (default CACHE_MIN_GB), for the sweep of the measurement.
int64_t Mega::pin_budget_from_limit() const {
    const bool tiers = tiers_on(profile_);
    const double floor_gb =
        tiers ? env::number("TONA_TIERS_CACHE_GB", backend::TIERED_CACHE_MIN_GB) : backend::CACHE_MIN_GB;
    // The pinned buffers of the KV cache and of the further sequences exist but are filled later (hipHostMalloc does
    // not touch its pages): ram_held_bytes() shows 0.25 GB here and the server holds 2 to 4 GB once loaded. With
    // TONA_TIERS they count now, otherwise the page cache floor is not kept (ctx 140000: 23.5 GB pinned in a 26 GB
    // limit, 0.6 GB left for the cache; research/measurements/2026-11-15-expert-tiers.md, step 3)
    const int64_t kv_counted =
        kv_host_
            ? (int64_t)((double)kv_bytes_ * kv_split_.host_frac * std::min<double>(max_ctx_, kTiersKvCtx) / max_ctx_)
            : 0;
    const int64_t later = tiers ? (int64_t)pin_bytes_ + kv_counted : 0;
    int64_t budget =
        (int64_t)((backend::ram_limit_gb() - backend::RAM_SLACK_GB - backend::prompt_reserve_gb() - floor_gb) * 1e9) -
        backend::ram_held_bytes() - later;
    const size_t all = all_expert_bytes();
    if (tiers ? !backend::partial_pin_worthwhile(
                    budget, all, env::number("TONA_TIERS_MIN_FRACTION", backend::MIN_PARTIAL_PIN_FRACTION))
              : budget < (int64_t)all)
        budget = 0;
    // scripts/bench-decode.sh does not read this line
    if (env::text("TONA_VRAM_LOG"))
        std::fprintf(
            stderr,
            "Pin budget: limit %.1f GB, held %.2f GB, filled later %.2f GB, all experts %.2f GB -> %s (%.2f GB)\n",
            backend::ram_limit_gb(), backend::ram_held_bytes() / 1e9, later / 1e9, all / 1e9,
            budget == 0             ? "none"
            : (size_t)budget >= all ? "all"
                                    : "part",
            std::min<double>(budget, all) / 1e9);
    return budget;
}

// Pinned stage: as many experts as RAM is left (TONA_PIN_GB; otherwise with TONA_RAM_GB all or none, otherwise
// MemTotal - 34 GiB), at most what is left of the GTT if the host memory is GTT (gpu/gtt.h)
namespace {
// TONA_PIN_OVERFLOW: "register" (default): the pinned experts that do not fit the GTT are registered anonymous
// memory (userptr); "cap": the pinned experts are capped to the GTT (the rest is read from the page cache)
bool overflow_registered() {
    return std::string_view(env::text("TONA_PIN_OVERFLOW") ? env::text("TONA_PIN_OVERFLOW") : "register") != "cap";
}
// state of one pin_experts() run (pin_layer is called only from there, one thread)
int64_t g_gtt_left = INT64_MAX;
size_t g_registered_bytes = 0;
int64_t pin_overflow_cap(int64_t budget) { return overflow_registered() ? budget : cap_pin_to_gtt(budget); }
}  // namespace

int64_t Mega::pin_budget() const {
    const double gb = env::number("TONA_PIN_GB", -1.0);
    if (gb >= 0) return pin_overflow_cap((int64_t)(gb * 1e9));
    if (backend::ram_limit_gb() >= 0) return pin_overflow_cap(pin_budget_from_limit());
    std::ifstream in("/proc/meminfo");
    std::string key;
    long long kb = 0;
    if (!(in >> key >> kb)) kb = 0;  // "MemTotal: <n> kB"
    return pin_overflow_cap(kb * 1024 - (34LL << 30));
}

// Which experts per layer fit `budget` bytes: by usage over all layers with TONA_TIERS and a profile when only part of
// them fit (one ranking, so the layers differ in count), else the same number per layer, the most frequent by the
// profile (without a profile: all, or by number). Empty: none
std::vector<std::vector<int>> Mega::choose_pinned(int64_t budget) const {
    const int E = (int)experts_.max_experts();
    std::vector<std::vector<int>> chosen(h_.n_layer);
    if (tiers_on(profile_) && (size_t)budget < all_expert_bytes()) {
        std::vector<size_t> bytes(h_.n_layer);
        for (uint32_t l = 0; l < h_.n_layer; l++) bytes[l] = cache_[l].slot_bytes;
        return backend::select_pinned(experts_, profile_, bytes, cache_[0].n_slots, budget);
    }
    size_t per_layer = 0;
    for (const auto& C : cache_) per_layer += C.slot_bytes;
    const int n = (int)std::clamp<int64_t>(budget / (int64_t)std::max<size_t>(per_layer, 1), 0, E);
    if (n == 0) return {};
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        const int El = (int)experts_.experts(l), k = std::min<int>(n, El);
        const std::vector<int> ord =
            backend::experts_by_count(profile_.empty() ? nullptr : profile_.data() + experts_.offset(l), El);
        // If there is no room for all: the next ones after those already in the VRAM cache
        const int first = k < El && !profile_.empty() ? std::min(cache_[l].n_slots, El - k) : 0;
        chosen[l].assign(ord.begin() + first, ord.begin() + first + k);
    }
    return chosen;
}

void Mega::pin_experts() {
    pin_.assign(h_.n_layer, {});
    for (uint32_t l = 0; l < h_.n_layer; l++) pin_[l].idx.assign(experts_.experts(l), -1);
    const int64_t budget = pin_budget();
    if (budget <= 0) return;
    const std::vector<std::vector<int>> chosen = choose_pinned(budget);
    if (chosen.empty()) return;
    const auto t0 = std::chrono::steady_clock::now();
    g_gtt_left = overflow_registered() ? gtt_room() : INT64_MAX;
    g_registered_bytes = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) pin_layer(l, chosen[l]);
    if (g_registered_bytes) {
        std::fprintf(stderr,
                     "host memory: %.1f GB of the pinned experts do not fit the GTT and are registered as userptr\n",
                     g_registered_bytes / 1e9);
        log_registered_memory(g_registered_bytes);
    }
    size_t n_pinned = 0;
    for (const auto& c : chosen) n_pinned += c.size();
    // scripts/bench-decode.sh reads "pinned" from this line
    if (env::text("TONA_MEGA_INFO"))
        std::fprintf(stderr, "gpu::Mega: %zu experts per layer pinned (%.1f GB, %zu in total) in %.1f s\n",
                     n_pinned / h_.n_layer, pin_bytes_ / 1e9, n_pinned, seconds_since<std::chrono::steady_clock>(t0));
}

void Mega::pin_layer(uint32_t l, const std::vector<int>& chosen) {
    const int n = (int)chosen.size();
    const LayerCache& C = cache_[l];
    const MoeWeights& S = m_.moe_weights(l);
    PinLayer& P = pin_[l];
    const size_t sb = C.slot_bytes;
    void* raw;
    const size_t bytes = (size_t)n * sb + sb + kMatvecReadPad;
    if ((int64_t)bytes <= g_gtt_left) {
        HIPCHECK(trace::host_malloc(&raw, bytes, hipHostMallocMapped | hipHostMallocNonCoherent));
        pin_allocs_.push_back(raw);
        g_gtt_left -= (int64_t)bytes;
    } else {
        raw = register_host_memory(bytes);
        g_registered_bytes += bytes;
    }
    uint8_t* dev;
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&dev), raw, 0));
    const int64_t diff = reinterpret_cast<intptr_t>(C.slots) - reinterpret_cast<intptr_t>(dev);
    const int64_t delta = ((diff % (int64_t)sb) + (int64_t)sb) % (int64_t)sb;
    P.host = static_cast<uint8_t*>(raw) + delta;
    P.slot0 = (reinterpret_cast<intptr_t>(dev) + delta - reinterpret_cast<intptr_t>(C.slots)) / (int64_t)sb;
    if (P.slot0 < INT32_MIN || P.slot0 + n > INT32_MAX)
        throw std::runtime_error("gpu::Mega: the pinned buffer is too far from the slots");
    const ExpertParts XP = expert_parts(S, C);
    const auto& parts = XP.t;
    const auto& offs = XP.off;
    backend::OmpErrors err;  // a failing pread must reach the caller, not end the process
#pragma omp parallel for collapse(2) schedule(dynamic, 4)
    for (int j = 0; j < n; j++)
        for (int k = 0; k < XP.n; k++)
            err.run([&] {
                m_.gguf().read(*parts[k], (size_t)chosen[j] * bytes_per_expert(parts[k]), P.host + j * sb + offs[k],
                               bytes_per_expert(parts[k]));
            });
    err.rethrow();
    for (int j = 0; j < n; j++) {
        P.idx[chosen[j]] = j;
        for (int k = 0; k < XP.n; k++)  // now pinned: out of the page cache
            m_.gguf().drop_range(*parts[k], (size_t)chosen[j] * bytes_per_expert(parts[k]), bytes_per_expert(parts[k]));
    }
    pin_bytes_ += (size_t)n * sb;
    progress::add(1);
}

}  // namespace tona::gpu
