// Test-only profile of the latent MoE steps (probe_kernels.h): the steps of mega_op_moe_latent.h and the sigmoid
// routing.
#include "gpu/mega_op_moe_latent.h"
#include "gpu/mega_op_route_sig.h"
#include "gpu/probe/probe_kernels.h"
#include "gpu/probe/probe_set.h"

namespace tona::gpu::probe {
namespace {

struct LatentProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr int max_experts = MEGA_MAX_EXPERTS_WIDE;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_moe_latent(op, c) || dispatch_route_sigmoid(op, c);
    }
};

}  // namespace

const KernelSet& latent_moe_kernels() { return ProbeKernels<LatentProfile>::set; }

}  // namespace tona::gpu::probe
