// What the host knows about the megakernel of one architecture: the kernels of its translation unit
// (gpu/arch/<name>/kernel.hip) behind a table of functions. A launch needs the kernel in the launching translation
// unit (no -fgpu-rdc), so the launch function sits in that unit and `Mega` reaches it through this table, one
// indirect call per pass. Host and device code may include this header; it holds no device code.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/rope_yarn.h"
#include "gpu/mega_types.h"

namespace tona::gpu {

// Scalar arguments of k_mega. The pointers stay kernel parameters of their own: only those are known to point into
// global memory, in a struct they would be flat pointers (flat instead of global loads and atomics).
struct MegaScalars {
    int o0, o1;  // steps [o0, o1) of the program
    uint64_t timeout_ticks;
    int wall_khz;
    unsigned test_gen, test_all;  // TONA_TEST_FREEZE_MS: barrier of the simulated eviction in this launch
    uint64_t freeze_ticks, cap_ticks, test_ticks;  // see WaitLimits (freeze_ticks ~0: off)
    unsigned long long* hist;                      // TONA_MEGA_GAP_HIST: histogram of the gaps between polls
};

// Properties of a program that choose among the instantiations of one architecture
struct KernelTraits {
    bool q6_tile = false;  // the program has Q6_K tile tasks (matvec_phase.h, Q6Layout)
};

// Everything a launch hands to the kernel, with the grid size
struct KernelArgs {
    const void *ops, *tasks;
    MegaParams* dparams;
    unsigned* bar;
    unsigned long long* prof;
    MegaScalars scalars;
    unsigned blocks;
};

// Resources of one instantiation (hipFuncGetAttributes and the occupancy calculation of HIP)
struct KernelProbe {
    int vgprs = 0, max_threads = 0, blocks_per_cu = 0;
    size_t lds_bytes = 0, scratch_bytes = 0;
};

// Kernel width class: 0 for one token, 1 for up to 4, 2 for up to 8 (one instantiation each)
struct KernelSet {
    // Constant tables of the kernel's translation unit (quantization grids, YaRN); once, before the first launch
    void (*init_tables)(const rope::YarnTable& yarn);
    KernelProbe (*probe)(int width_class, KernelTraits traits);
    void (*launch)(int width_class, KernelTraits traits, const KernelArgs& args);
};

}  // namespace tona::gpu
