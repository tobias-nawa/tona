// The megakernel of the Coder (qwen4exp, 256 experts per layer): the instantiations of k_mega with its profile. A
// launch needs the kernel in the same translation unit (no -fgpu-rdc); the host code in gpu/mega_launch.cpp reaches
// the functions through the KernelSet.

#include "gpu/arch/qwen4exp/kernel_set.h"
#include "gpu/arch/qwen4exp/kernel_unit.h"

namespace tona::gpu {

const KernelSet& qwen4exp_kernels() { return kernel_set<Qwen4ExpProfile>(); }

}  // namespace tona::gpu
