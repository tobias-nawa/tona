// The megakernel of the 512-expert Flash-Next (qwen4exp, 512 experts per layer): the instantiations of k_mega with
// its profile, in a unit of its own (see kernel_unit.h).

#include "gpu/arch/qwen4exp/kernel_set.h"
#include "gpu/arch/qwen4exp/kernel_unit.h"

namespace tona::gpu {

const KernelSet& qwen4exp_512_kernels() { return kernel_set<Qwen4Exp512Profile>(); }

}  // namespace tona::gpu
