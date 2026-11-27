// Test-only profile of the matvec phase with the NVFP4 and IQ2_XXS loops (probe_kernels.h): k_mega runs OP_MATVEC steps
// only, as it does for a model whose profile names ExtraQuants::kNvfp4Iq2xxs.
#include "gpu/probe/probe_kernels.h"
#include "gpu/probe/probe_set.h"

namespace tona::gpu::probe {
namespace {

struct QuantProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr ExtraQuants quants = ExtraQuants::kNvfp4Iq2xxs;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp&, const Ctx&) {
        return false;
    }
};

}  // namespace

const KernelSet& quant_matvec_kernels() { return ProbeKernels<QuantProfile>::set; }

}  // namespace tona::gpu::probe
