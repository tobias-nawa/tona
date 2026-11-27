// The order in which a prefill runs the layers over the chunks of a group, and where a step does not follow its own
// chunk's previous layer. Pure host logic, so a test can check it without a GPU (tests/test_prefill_order.cpp).
#pragma once

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace tona::gpu {

using PfStep = std::pair<uint32_t, int>;  // layer, chunk

// Chunk by chunk: every chunk through all layers. Layer by layer: layers 0 and 1 per chunk (they need the embedding of
// the chunk), from layer 2 each layer for all chunks. With one chunk both are the same.
inline std::vector<PfStep> pf_layer_order(uint32_t n_blocks, int n_chunks, bool layered) {
    std::vector<PfStep> order;
    const uint32_t first_shared = layered ? 2 : n_blocks;
    for (int j = 0; j < n_chunks; j++)
        for (uint32_t l = 0; l < first_shared && l < n_blocks; l++) order.emplace_back(l, j);
    for (uint32_t l = first_shared; l < n_blocks; l++)
        for (int j = 0; j < n_chunks; j++) order.emplace_back(l, j);
    return order;
}

// Step i is layer l > 0 of a chunk whose layer l - 1 did not run just before it: whatever the previous layer left in
// the buffers shared by all chunks (the norm of the next layer's input) belongs to another chunk. Layer 0 starts from
// the embedding.
inline bool pf_needs_resume(const std::vector<PfStep>& order, size_t i) {
    const auto [l, j] = order[i];
    return l > 0 && (i == 0 || order[i - 1] != PfStep{l - 1, j});
}

}  // namespace tona::gpu
