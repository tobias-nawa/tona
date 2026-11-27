// State of the Mamba-2 layers of Bolt in the GPU backend. Per layer: the SSM state [heads][head_dim][128] (the only
// state that needs a commit), the ring of the last conv inputs (no commit: positions past a rollback are overwritten,
// see op_mamba_conv) and the inputs of the last pass for the replay (design: docs/architecture/model-architectures.md
// 3.4). Everything comes from Mega::alloc_state, so reset, snapshots, stash, the prompt cache and the slots of several
// sequences handle it without knowing Mamba; the pointers of PfMamba2 are retargeted per slot in mega_seq.cpp.
#pragma once

#include <cstddef>

#include "gpu/mega_prefill_types.h"
#include "model/blocks.h"

namespace tona::gpu {

class Mega;

class NemotronHState {
public:
    // Floats of one layer's state (what alloc_layer allocates), for the VRAM budget
    static size_t layer_floats(const Mamba2Dims& D);
    // Allocates the state of one Mamba-2 layer and returns the descriptor with the state pointers filled in (weights
    // are filled by the program). Throws for shapes the kernels are not written for.
    static PfMamba2 alloc_layer(Mega& g, const Mamba2Dims& D);
};

}  // namespace tona::gpu
