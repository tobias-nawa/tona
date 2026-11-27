// Test-only profile of the plain residual step (probe_kernels.h).
#include "gpu/mega_op_resnorm.h"
#include "gpu/probe/probe_kernels.h"
#include "gpu/probe/probe_set.h"

namespace tona::gpu::probe {
namespace {

// k_mega with the plain residual step only
struct ResProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_resnorm(op, c);
    }
};

}  // namespace

const KernelSet& resnorm_kernels() { return ProbeKernels<ResProfile>::set; }

}  // namespace tona::gpu::probe
