// Test-only profiles of the full-attention steps and of the QSA attention (probe_kernels.h).
#include "gpu/mega_op_attn.h"
#include "gpu/mega_op_fattn.h"
#include "gpu/probe/probe_kernels.h"
#include "gpu/probe/probe_set.h"

namespace tona::gpu::probe {
namespace {

// k_mega with the full-attention steps only (the profile of an architecture that has no other family)
template <FattnCfg Cfg>
struct FattnProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_fattn<Cfg>(op, c);
    }
};

struct QsaProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_qsa(op, c);
    }
};

static_assert(kFattnSplits == FA_SPL, "the header of the probes states the split count of mega_op_fattn.h");

constexpr FattnCfg cfg_of(const FattnShape& s) { return {s.hd, s.gate, s.qk_norm, s.rope}; }

}  // namespace

const KernelSet& fattn_kernels(const FattnShape& s) {
    if (s == kOrnith) return ProbeKernels<FattnProfile<cfg_of(kOrnith)>>::set;
    if (s == kBolt) return ProbeKernels<FattnProfile<cfg_of(kBolt)>>::set;
    if (s == kBoltAll) return ProbeKernels<FattnProfile<cfg_of(kBoltAll)>>::set;
    throw std::invalid_argument("no probe kernel for this full-attention shape");
}

const KernelSet& qsa_attention_kernels() { return ProbeKernels<QsaProfile>::set; }

}  // namespace tona::gpu::probe
