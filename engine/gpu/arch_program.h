// The part of the program builder that belongs to one model architecture. ProgramBuilder (mega_build.h) is the toolbox
// (ops, matvec phases, allocation, slots, upload, diagnostics) and calls these steps in a fixed order; an architecture
// owns its buffers, its layer pattern and the ops of its blocks. Host and cold path only.
#pragma once

#include <memory>

#include "backend/expert_layout.h"
#include "gpu/kernel_set.h"
#include "model/model_base.h"

namespace tona::gpu {

class ProgramBuilder;

class ArchProgram {
public:
    virtual ~ArchProgram() = default;
    // Per-token buffers of the blocks, the model's KV geometry and the logits. Runs first: the order of the
    // allocations is part of the program (TONA_MEGA_DUMP_PROGRAM compares it).
    virtual void alloc_buffers() = 0;
    // Whether the architecture has prompt kernels of its own (the prefill path); without them every prompt runs through
    // the decode kernel in passes of several tokens (backend::Backend::prefill_chunk() == 0)
    virtual bool has_prefill() const { return true; }
    // Embedding, all layers and the head as the program of the target
    virtual void build_target() = 0;
    // Called only when the model has an MTP head; an architecture without one throws.
    virtual void build_mtp() = 0;
    // Called only when a drafter is loaded; an architecture without one throws.
    virtual void build_drafter() = 0;
};

// The one place that maps the loaded model to its architecture
std::unique_ptr<ArchProgram> make_arch_program(ProgramBuilder& pb);

// The megakernel of the model's architecture, next to the program builder it belongs to. A function of its own, not
// ArchProgram::kernels(): Mega probes the kernels before the builder (and with it the program) exists.
const KernelSet& kernels_of(const ModelBase& m);

// The shapes the architecture's kernels are written for; throws when the model has others. Also before the builder.
void check_arch_shapes(const ModelBase& m, const backend::ExpertLayout& experts);

}  // namespace tona::gpu
