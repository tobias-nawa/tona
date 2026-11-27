// The megakernel of Ornith (qwen35moe), as the host sees it: defined in gpu/arch/qwen35moe/kernel.hip.
#pragma once

#include "gpu/kernel_set.h"

namespace tona::gpu {

const KernelSet& qwen35moe_kernels();

// Chunks of the full-attention split (FA_SPL of gpu/mega_op_fattn.h, checked in kernel.hip): sizes the partial results
inline constexpr int kFattnSplits = 64;

}  // namespace tona::gpu
