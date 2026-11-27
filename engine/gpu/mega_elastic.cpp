#include "gpu/mega_elastic.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <stdexcept>

#include "gpu/mega_extra.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

uint64_t round_up(uint64_t v, uint64_t m) { return (v + m - 1) / m * m; }
uint64_t round_down(uint64_t v, uint64_t m) { return v / m * m; }

}  // namespace

size_t vmm_granule() {
    static const size_t g = [] {
        hipMemAllocationProp prop = {};
        prop.type = hipMemAllocationTypePinned;
        prop.location.type = hipMemLocationTypeDevice;
        prop.location.id = 0;
        size_t gran = 0;
        if (hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityRecommended) != hipSuccess ||
            gran == 0)
            return (size_t)2 << 20;
        return gran;
    }();
    return g;
}

uint8_t* ElasticPool::open(const ElasticConfig& cfg) {
    cfg_ = cfg;
    prop_.type = hipMemAllocationTypePinned;
    prop_.location.type = hipMemLocationTypeDevice;
    prop_.location.id = 0;
    access_.location = prop_.location;
    access_.flags = hipMemAccessFlagsProtReadWrite;
    size_t gran = 0;
    void* va = nullptr;
    auto fail = [&](const char* what, hipError_t e) -> uint8_t* {
        std::fprintf(stderr, "gpu::Mega: slot pool: %s failed (%s), ordinary slot allocation\n", what,
                     hipGetErrorString(e));
        return nullptr;
    };
    hipError_t e = hipMemGetAllocationGranularity(&gran, &prop_, hipMemAllocationGranularityRecommended);
    if (e != hipSuccess || gran == 0) return fail("granularity", e);
    gran_ = gran;
    const size_t core_bytes = round_up(cfg.core_bytes, gran_), total = core_bytes + round_up(cfg.va_bytes, gran_);
    e = hipMemAddressReserve(&va, total, gran_, nullptr, 0);
    if (e != hipSuccess) return fail("address reservation", e);
    base_ = static_cast<uint8_t*>(va);
    e = map_core(core_bytes);
    if (e == hipSuccess) e = hipMemSetAccess(base_, core_bytes, &access_, 1);
    if (e != hipSuccess) {
        unmap_core(core_bytes);
        (void)hipMemAddressFree(va, total);
        base_ = nullptr;
        return fail("mapping the core", e);
    }
    if (cfg.chunk_bytes && std::getenv("TONA_VRAM_LOG"))
        std::fprintf(stderr,
                     "VRAM: the core (slots and prefill workspace, %.0f MB) is built from %zu chunks of %.0f MB\n",
                     core_bytes / 1e6, core_handles_.size(), round_up(cfg.chunk_bytes, gran_) / 1e6);
    ledger_.map(reinterpret_cast<uint64_t>(base_), core_bytes);
    cursor_ = reinterpret_cast<uint64_t>(base_) + core_bytes;
    va_end_ = cursor_ + round_up(cfg.va_bytes, gran_);
    return base_;
}

void ElasticPool::release() {
    if (!base_) return;
    (void)hipDeviceSynchronize();
    for (Generation& g : gens_)
        if (g.live)
            for (ElasticBlock& b : g.blocks) unmap_block(b);
    const uint64_t core_end = reinterpret_cast<uint64_t>(base_) + round_up(cfg_.core_bytes, gran_);
    unmap_core(core_end - reinterpret_cast<uint64_t>(base_));
    (void)hipMemAddressFree(base_, va_end_ - reinterpret_cast<uint64_t>(base_));
    base_ = nullptr;
}

// The core as one handle, or as chunks of cfg_.chunk_bytes (a multiple of the granule): every hipMemCreate is its own
// buffer object, so the kernel can evict one chunk instead of the whole core. The addresses are contiguous either way.
hipError_t ElasticPool::map_core(size_t core_bytes) {
    size_t chunk = cfg_.chunk_bytes ? round_up(cfg_.chunk_bytes, gran_) : core_bytes;
    chunk = std::min(chunk, core_bytes);
    for (size_t off = 0; off < core_bytes; off += chunk) {
        const size_t len = std::min(chunk, core_bytes - off);
        hipMemGenericAllocationHandle_t h = {};
        hipError_t e = hipMemCreate(&h, len, &prop_, 0);
        if (e != hipSuccess) return e;
        core_handles_.push_back(h);
        e = hipMemMap(base_ + off, len, 0, h, 0);
        if (e != hipSuccess) return e;
    }
    return hipSuccess;
}

// Unmaps what map_core mapped (chunk by chunk) and releases the handles; also after a failed map_core
void ElasticPool::unmap_core(size_t core_bytes) {
    size_t chunk = cfg_.chunk_bytes ? round_up(cfg_.chunk_bytes, gran_) : core_bytes;
    chunk = std::min(chunk, core_bytes);
    for (size_t i = 0; i < core_handles_.size(); i++) {
        const size_t off = i * chunk;
        (void)hipMemUnmap(base_ + off, std::min(chunk, core_bytes - off));
        (void)hipMemRelease(core_handles_[i]);
    }
    core_handles_.clear();
}

void ElasticPool::unmap_block(ElasticBlock& b) {
    (void)hipMemUnmap(reinterpret_cast<void*>(b.map_begin), b.map_end - b.map_begin);
    (void)hipMemRelease(b.handle);
    ledger_.unmap(b.map_begin);
}

// The block of one layer: the first slot index whose address is at or behind the cursor, memory rounded out to granules
bool ElasticPool::grow_layer(LayerCache& C, int layer, ElasticBlock& b) {
    const uint64_t lbase = reinterpret_cast<uint64_t>(C.slots), sb = C.slot_bytes;
    const uint64_t idx = (cursor_ - lbase + sb - 1) / sb;
    const uint64_t a = lbase + idx * sb, bytes = (uint64_t)cfg_.step_slots * sb;
    b.layer = layer, b.first = (int)idx, b.count = cfg_.step_slots;
    b.map_begin = round_down(a, gran_), b.map_end = round_up(a + bytes + cfg_.read_pad, gran_);
    if (idx + (uint64_t)cfg_.step_slots > (uint64_t)INT32_MAX || b.map_end > va_end_) return false;
    if (!ledger_.map(b.map_begin, b.map_end - b.map_begin)) return false;
    const size_t len = b.map_end - b.map_begin;
    void* at = reinterpret_cast<void*>(b.map_begin);
    cursor_ = b.map_end;  // whatever happens next, these addresses are spent
    if (hipMemCreate(&b.handle, len, &prop_, 0) != hipSuccess) return ledger_.unmap(b.map_begin), false;
    if (hipMemMap(at, len, 0, b.handle, 0) != hipSuccess) {
        (void)hipMemRelease(b.handle);
        return ledger_.unmap(b.map_begin), false;
    }
    if (hipMemSetAccess(at, len, &access_, 1) != hipSuccess) return unmap_block(b), false;
    return true;
}

bool ElasticPool::grow(std::vector<LayerCache>& caches, bool front) {
    const uint64_t start = cursor_;
    Generation g;
    g.blocks.resize(caches.size());
    for (size_t l = 0; l < caches.size(); l++)
        if (!grow_layer(caches[l], (int)l, g.blocks[l])) {
            for (size_t k = 0; k < l; k++) unmap_block(g.blocks[k]);
            return false;
        }
    gen_va_ = std::max(gen_va_, cursor_ - start);
    gen_bytes_ = 0;
    for (const ElasticBlock& b : g.blocks) gen_bytes_ += b.map_end - b.map_begin;
    for (size_t l = 0; l < caches.size(); l++)
        add_elastic_block(caches[l], g.blocks[l].first, g.blocks[l].count, front);
    gens_.push_back(std::move(g));
    return true;
}

size_t ElasticPool::shrink(std::vector<LayerCache>& caches) {
    for (size_t gi = gens_.size(); gi-- > 0;) {
        Generation& g = gens_[gi];
        if (!g.live) continue;
        for (const ElasticBlock& b : g.blocks) kill_block(caches[b.layer], b.first);
        (void)hipDeviceSynchronize();  // the caller checked that no pass runs; this also ends copies into these slots
        size_t bytes = 0;
        for (ElasticBlock& b : g.blocks) bytes += b.map_end - b.map_begin, unmap_block(b);
        g.live = false;
        return bytes;
    }
    return 0;
}

void ElasticPool::step(std::vector<LayerCache>& caches, int n, double now, int64_t free_bytes) {
    const bool log = std::getenv("TONA_VRAM_LOG") != nullptr;
    if (n < 0) {
        int done = 0;
        size_t bytes = 0;
        for (int i = 0; i < -n; i++) {
            const size_t b = shrink(caches);
            if (b == 0) break;
            bytes += b, done++;
        }
        gov_->applied(-done);
        count(-done);
        if (log)
            std::fprintf(stderr, "VRAM governor: %.1f s free %.0f MB, gave back %d generations (%.0f MB), %d left\n",
                         now, free_bytes / 1e6, done, bytes / 1e6, live_generations());
        return;
    }
    if (grow(caches, false)) {
        gov_->applied(1);
        count(1);
    } else {
        gov_->no_budget();
    }
    if (log)
        std::fprintf(stderr, "VRAM governor: %.1f s free %.0f MB, took a generation, %d live, %d address ranges left\n",
                     now, free_bytes / 1e6, live_generations(), generations_left());
}

void ElasticPool::require_live(const LayerCache& C, int layer, int slot) const {
    if (!base_) return;  // no governor: the slots are one ordinary allocation
    const uint64_t at = reinterpret_cast<uint64_t>(C.slots) + (uint64_t)slot * C.slot_bytes;
    if (slot >= 0 && ledger_.is_live(at, C.slot_bytes)) return;
    std::fprintf(stderr, "gpu::Mega: refused a write into slot %d of layer %d: its memory is not mapped\n", slot,
                 layer);
    throw std::runtime_error(std::format("gpu::Mega: slot {} of layer {} is not live", slot, layer));
}

int ElasticPool::live_generations() const {
    return (int)std::count_if(gens_.begin(), gens_.end(), [](const Generation& g) { return g.live; });
}

int ElasticPool::generations_left() const {
    if (gen_va_ == 0) return 0;
    return (int)((va_end_ > cursor_ ? va_end_ - cursor_ : 0) / gen_va_);
}

bool ElasticPool::load_fake(const char* path) {
    std::FILE* f = std::fopen(path, "r");
    if (!f) return false;
    char line[128];
    while (std::fgets(line, sizeof line, f)) {
        char* rest = nullptr;
        const double t = std::strtod(line, &rest);
        char* end = nullptr;
        const double mib = std::strtod(rest, &end);
        if (end != rest) fake_.emplace_back(t, (int64_t)(mib * 1048576.0));
    }
    std::fclose(f);
    return !fake_.empty();
}

int64_t ElasticPool::fake_free(double t) const {
    int64_t v = fake_.front().second;
    for (const auto& [at, free_bytes] : fake_)
        if (at <= t) v = free_bytes;
    return v;
}

void ElasticPool::start_governor(const backend::VramMarks& m, int live_steps) {
    backend::VramMarks mk = m;
    mk.step_bytes = (int64_t)gen_bytes_;
    gov_ = std::make_unique<backend::VramGovernor>(mk, live_steps, 0, generations_left());
}

}  // namespace tona::gpu
