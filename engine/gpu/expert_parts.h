// The tensors of one expert in the order they lie in a cache slot, and where. A gated layer (the Coder, Ornith) has
// gate | up | down; an ungated one (Bolt: up and down only) has up | down, so its up part starts at the slot's start.
// The slot layout itself is set once per layer in ProgramBuilder::init_cache_layout; everything that copies experts
// reads the parts from here instead of assuming three.
#pragma once

#include <cstddef>

#include "gpu/mega_cache.h"
#include "model/blocks.h"

namespace tona::gpu {

struct ExpertParts {
    const TensorInfo* t[3];
    size_t off[3];
    int n;
};

inline ExpertParts expert_parts(const MoeWeights& S, const LayerCache& C) {
    if (S.gate_exps) return {{S.gate_exps, S.up_exps, S.down_exps}, {0, C.off_up, C.off_down}, 3};
    return {{S.up_exps, S.down_exps, nullptr}, {0, C.off_down, 0}, 2};
}

// The slot of one layer: [gate |] up | down, each part padded to 256 bytes. The widths come from the layer's own
// tensors (up_exps [in, ff, E], down_exps [ff, out, E]), so a layer with its own expert width or a latent width needs
// nothing else; an ungated layer (no gate tensor) has its up part at 0.
struct SlotLayout {
    size_t off_up, off_down, slot_bytes;
};

inline SlotLayout slot_layout(const MoeWeights& S) {
    const int64_t in = (int64_t)S.up_exps->dims[0], ff = (int64_t)S.up_exps->dims[1];
    const int64_t out = (int64_t)S.down_exps->dims[1];
    const size_t gate = S.gate_exps ? (size_t)(ff * type_row_bytes(S.gate_exps->type, in)) : 0;
    const size_t up = (size_t)(ff * type_row_bytes(S.up_exps->type, in));
    const size_t down = (size_t)(out * type_row_bytes(S.down_exps->type, ff));
    SlotLayout L{};
    L.off_up = (gate + 255) / 256 * 256;
    L.off_down = L.off_up + (up + 255) / 256 * 256;
    L.slot_bytes = L.off_down + (down + 255) / 256 * 256;
    return L;
}

}  // namespace tona::gpu
