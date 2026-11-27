// The megakernel of Bolt (nemotron_h_moe), as the host sees it: defined in gpu/arch/nemotron_h/kernel.hip.
#pragma once

#include "gpu/kernel_set.h"

namespace tona::gpu {

const KernelSet& nemotron_h_kernels();

// Chunks of the full-attention split (FA_SPL of gpu/mega_op_fattn.h, checked in kernel.hip): sizes the partial results
inline constexpr int kBoltFattnSplits = 64;

}  // namespace tona::gpu
