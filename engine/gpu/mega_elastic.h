// The elastic part of the expert slot pool (TONA_VRAM_GOVERNOR=1, docs/architecture/expert-tiers.md section 5a): slots
// that live in blocks of virtual memory, so that the pool can return VRAM to the driver between two passes when other
// programs need it, and take it again when it is free. The core slots and the prefill workspace stay one mapping.
//
// The rule that shapes everything: an address range that was unmapped is never mapped again (ROCm/TheRock #8744: a new
// mapping at the same address writes through a stale translation into pages that were given back, 29 of 400 chunks
// wrong on this machine). Growing therefore maps new blocks at never-used addresses, and the slot index follows the
// address: the slot of layer l at address A has the index (A - base of l) / slot bytes, which the kernels take as a
// 64-bit offset.
#pragma once

#include <hip/hip_runtime.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "backend/va_ledger.h"
#include "backend/vram_governor.h"
#include "gpu/mega_cache.h"

namespace tona::gpu {

// Granularity of the virtual-memory API for device memory on this card (hipMemGetAllocationGranularity, recommended):
// every block and chunk is a multiple of it. Asked once; 2 MiB if the runtime does not answer (its documented minimum
// on AMD).
size_t vmm_granule();

struct ElasticBlock {
    int layer = 0, first = 0, count = 0;
    uint64_t map_begin = 0, map_end = 0;  // granule aligned, the memory behind the block (with the read padding)
    hipMemGenericAllocationHandle_t handle = {};
};

struct ElasticConfig {
    size_t core_bytes = 0;   // the mapping of the core slots and the prefill workspace
    size_t va_bytes = 0;     // address space behind it for the blocks (never freed, never reused)
    int step_slots = 4;      // slots per layer in one generation
    size_t read_pad = 64;    // the matvec reads this far behind a slot (kMatvecReadPad)
    size_t chunk_bytes = 0;  // > 0: the core is built from chunks of this size, one buffer object each, else one
};

class ElasticPool {
public:
    ElasticPool() = default;
    ElasticPool(const ElasticPool&) = delete;
    ElasticPool& operator=(const ElasticPool&) = delete;
    ~ElasticPool() { release(); }

    bool on() const { return base_ != nullptr; }
    // Reserves the address space and maps the core. nullptr (and a message) if the runtime refuses: the caller then
    // uses an ordinary allocation.
    uint8_t* open(const ElasticConfig& cfg);
    // Frees everything (device must be idle)
    void release();

    // One more generation: step_slots slots in every layer at fresh addresses. False if the address space is used up or
    // the runtime refuses (what was mapped is unmapped again and stays in the ledger as dead).
    bool grow(std::vector<LayerCache>& caches, bool front);
    // What the governor asked for (n < 0: give back -n generations, n > 0: take one), with the bookkeeping and the log
    // line. The device must be idle and no copy into a slot in flight (the caller checked).
    void step(std::vector<LayerCache>& caches, int n, double now, int64_t free_bytes);
    // Gives the newest live generation back; the device must be idle. Returns the bytes released, 0 if none is live.
    size_t shrink(std::vector<LayerCache>& caches);

    // The hard check of every path that writes into a slot: the slot's bytes must lie in a live range. Throws.
    void require_live(const LayerCache& C, int layer, int slot) const;

    int live_generations() const;
    int generations_left() const;  // that fresh address space still allows
    size_t generation_bytes() const { return gen_bytes_; }
    uint64_t refused() const { return ledger_.refused(); }
    backend::VramGovernor& governor() { return *gov_; }
    void start_governor(const backend::VramMarks& m, int live_steps);
    bool governing() const { return gov_ != nullptr; }
    void count(int steps) { (steps < 0 ? shrunk_ : grown_) += steps < 0 ? -steps : steps; }
    int shrunk() const { return shrunk_; }
    int grown() const { return grown_; }

    // Seconds since the first tick (the engine thread starts after the model is loaded), and the poll interval of the
    // governor (two small sysfs reads every 250 ms)
    double now() {
        const auto t = std::chrono::steady_clock::now();
        if (!started_) t0_ = t, started_ = true;
        return std::chrono::duration<double>(t - t0_).count();
    }
    bool due(double now_s) {
        if (now_s - last_poll_ < 0.25) return false;
        return last_poll_ = now_s, true;
    }
    // TONA_VRAM_FAKE_FREE=<file>: lines "<seconds> <free MiB>"; from then on the governor reads this trace instead of
    // the card (a test without a second program on the GPU). The free VRAM at time t is the last line at or before t.
    bool load_fake(const char* path);
    bool fake_on() const { return !fake_.empty(); }
    int64_t fake_free(double t) const;

private:
    struct Generation {
        std::vector<ElasticBlock> blocks;
        bool live = true;
    };
    bool grow_layer(LayerCache& C, int layer, ElasticBlock& b);
    void unmap_block(ElasticBlock& b);
    hipError_t map_core(size_t core_bytes);
    void unmap_core(size_t core_bytes);

    uint8_t* base_ = nullptr;
    ElasticConfig cfg_;
    uint64_t gran_ = 0, va_end_ = 0, cursor_ = 0, gen_va_ = 0;  // gen_va_: address space of the largest generation
    size_t gen_bytes_ = 0;
    std::vector<hipMemGenericAllocationHandle_t> core_handles_;  // one per chunk of the core
    hipMemAllocationProp prop_ = {};
    hipMemAccessDesc access_ = {};
    backend::VaLedger ledger_;
    std::vector<Generation> gens_;
    std::unique_ptr<backend::VramGovernor> gov_;
    std::chrono::steady_clock::time_point t0_;
    bool started_ = false;
    double last_poll_ = -1;
    int shrunk_ = 0, grown_ = 0;  // generations given back and taken since the start
    std::vector<std::pair<double, int64_t>> fake_;
};

}  // namespace tona::gpu
