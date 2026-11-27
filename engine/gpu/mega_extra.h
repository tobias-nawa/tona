// Extra expert slots in the prefill workspace. A chunk-by-chunk prefill needs the workspace only while it runs (912 MB
// for chunks of 2,048 tokens), so between prefills the memory holds more slots of the expert cache. The workspace lies
// behind the slots in the same allocation: the extra slots get indices above the base slots, the kernels address them
// like any other slot (base + index * stride, 64-bit), and nothing in the megakernel changes.
#pragma once

#include <cstddef>
#include <vector>

#include "gpu/mega_cache.h"

namespace tona::gpu {

struct ExtraLayout {
    std::vector<size_t> slot_off;    // start of the slots of each layer in the allocation
    std::vector<size_t> slot_bytes;  // bytes of one slot of each layer
    size_t arena_start = 0;          // start of the workspace in the allocation
    size_t arena_bytes = 0;
    int base_slots = 0;   // slots every layer has anyway
    int max_slots = 0;    // most slots a layer can use (all its experts)
    size_t read_pad = 0;  // the matvec reads this far behind the end of a slot: the memory behind a block must exist
    int ranks = 2;        // blocks per layer; the prefill takes the memory from the bottom, rank 0 lies on top
};

// Layout of the extra blocks: as many slots per layer as fit, the same for every layer, from the top of the workspace
// downwards, one block per rank and layer in turn so that a prefill that needs only part of the workspace takes the
// same few slots from every layer. Throws if the result does not pass check_extras.
std::vector<std::vector<ExtraBlock>> plan_extras(const ExtraLayout& in);

// Every block lies inside the workspace, matches its layer's slot grid (index * slot bytes reaches exactly the block),
// has an index above the base slots and does not overlap another block. Throws std::runtime_error.
void check_extras(const ExtraLayout& in, const std::vector<std::vector<ExtraBlock>>& blocks);

// Puts the blocks into the caches: n_slots, extras, and the slots at the back of the free lists (used last)
void install_extras(std::vector<LayerCache>& caches, const std::vector<std::vector<ExtraBlock>>& blocks);

// Before a prefill that uses the first `used` bytes of the workspace: the blocks that start below `used` stop being
// slots. Their experts leave the cache. Returns the number of slots taken.
int evict_extras(std::vector<LayerCache>& caches, size_t used);

// After the prefill (the GPU is idle): the memory is slots again, empty
int restore_extras(std::vector<LayerCache>& caches);

// An elastic block (mega_elastic.h) gives its memory back: its experts leave the cache, its slots are no longer free,
// and it never becomes live again (the never-remap rule). Returns the slots taken.
int kill_block(LayerCache& C, int first);

// A block of slots in new memory (an elastic generation): added to the layer's blocks and to the free list, at the back
// (empty slots fill first: a regrow) or at the front (used last: the first generations at start, which then hold the
// least used experts and go first when the pool shrinks)
void add_elastic_block(LayerCache& C, int first, int count, bool front);

// All slots free, no expert in the cache (the base slots 0 ... base - 1 and the live blocks)
void reset_slots(LayerCache& C, int base);

// Slots in live blocks of one layer
int live_extras(const LayerCache& C);

}  // namespace tona::gpu
