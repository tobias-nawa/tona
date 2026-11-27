// The megakernel of the Coder (qwen4exp), as the host sees it: defined in gpu/arch/qwen4exp/kernel.hip.
#pragma once

#include "gpu/kernel_set.h"

namespace tona::gpu {

const KernelSet& qwen4exp_kernels();

// The same architecture with 512 experts per layer (expert_cap): kernels with the wide router rows
const KernelSet& qwen4exp_512_kernels();

}  // namespace tona::gpu
