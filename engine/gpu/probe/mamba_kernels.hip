// Test-only profile of the Mamba-2 steps (probe_kernels.h).
#include "gpu/mega_op_mamba.h"
#include "gpu/probe/probe_kernels.h"
#include "gpu/probe/probe_set.h"

namespace tona::gpu::probe {
namespace {

// k_mega with the Mamba-2 steps only
struct MambaProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_mamba<NB>(op, c);
    }
};

}  // namespace

const KernelSet& mamba_kernels() { return ProbeKernels<MambaProfile>::set; }

}  // namespace tona::gpu::probe
