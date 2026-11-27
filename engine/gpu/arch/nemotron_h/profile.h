// Kernel profile of Bolt (nemotron_h_moe): the op families its program runs, in the order the kernel tries them (see
// KernelProfile in gpu/mega_kernel.h). One plain residual stream, Mamba-2, full-context attention without gate, QK
// norm and RoPE (head size 128, 32 q heads over 2 KV heads), the latent MoE with the CPU's routing (no GPU router), and
// the matvec loops of NVFP4 and IQ2_XXS. Only this architecture's kernel translation unit includes it.
#pragma once

#include "gpu/mega_kernel.h"
#include "gpu/mega_op_fattn.h"
#include "gpu/mega_op_mamba.h"
#include "gpu/mega_op_moe_latent.h"
#include "gpu/mega_op_resnorm.h"

namespace tona::gpu {
namespace {

inline constexpr FattnCfg kBoltAttention{128, false, false, false};

struct NemotronHProfile {
    static constexpr Q6Layout q6 = Q6Layout::kPlain;
    static constexpr ExtraQuants quants = ExtraQuants::kNvfp4Iq2xxs;
    // Up to 496 experts per layer: the router rows of the mailbox are 512 wide (expert_cap)
    static constexpr int max_experts = MEGA_MAX_EXPERTS_WIDE;

    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_resnorm(op, c) || dispatch_mamba<NB>(op, c) || dispatch_fattn<kBoltAttention>(op, c) ||
               dispatch_moe_latent(op, c);
    }
};

}  // namespace
}  // namespace tona::gpu
