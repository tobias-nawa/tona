// The expert cache of one layer in VRAM and the experts pinned in host RAM.
#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "backend/expert_order.h"
#include "backend/route_policy.h"
#include "gpu/kv_place.h"

namespace tona::gpu {

// A run of extra slots that lies in the prefill workspace (mega_extra.h): its slots are usable while no prefill needs
// the memory under them
struct ExtraBlock {
    int first = 0;      // slot index of the first slot (counted from LayerCache::slots, always above the base slots)
    int count = 0;      // slots in the block
    size_t off = 0;     // start of the block in the workspace
    bool live = true;   // false: a prefill uses the memory, the slots are neither free nor hold an expert
    bool dead = false;  // elastic block (mega_elastic.h) whose memory went back to the driver: never live again
};

struct LayerCache : backend::ExpertOrder {
    uint8_t* slots = nullptr;
    size_t slot_bytes = 0, off_up = 0, off_down = 0;
    int n_used = 0;                  // experts per token in this layer (the model's per-layer top-k)
    int n_slots = 0;                 // base slots plus all extra blocks (live or not)
    std::vector<ExtraBlock> extras;  // in the prefill workspace, highest address first
    std::vector<int> pending;        // CPU mode: already read into fill_, enter their slots in the next pass
    std::vector<int> installed;      // put into a slot in this pass (recover takes them back)
    int fill_half = 0;  // half of fill_ that holds pending; the next ones go to the other half, because the
                        // GPU may still copy from this one while the CPU refills
    // How the router logits of this layer become experts (backend/route_policy.h); the architecture's program sets
    // another rule than the Coder's softmax
    backend::RoutePolicy route;
};

// Experts pinned in host RAM that the GPU reads directly over PCIe. The buffer lies so that (buffer - slots) is a
// multiple of the slot size: place j is then slot slot0 + j and the matvec tasks stay unchanged.
struct PinLayer {
    uint8_t* host = nullptr;  // place j at host + j * slot_bytes
    int64_t slot0 = 0;
    std::vector<int32_t> idx;  // expert -> place, -1: not pinned
};

}  // namespace tona::gpu
