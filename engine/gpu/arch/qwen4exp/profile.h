// Kernel profile of the Coder (qwen4exp): the op families its program runs, in the order the kernel tries them, and
// the Q6_K layouts of its kernel instantiation (see KernelProfile in gpu/mega_kernel.h). Only this architecture's
// kernel translation unit includes it, and with it the op headers of these families.
#pragma once

#include "gpu/mega_kernel.h"
#include "gpu/mega_op_attn.h"
#include "gpu/mega_op_dattn.h"
#include "gpu/mega_op_dn.h"
#include "gpu/mega_op_moe.h"
#include "gpu/mega_op_mtp.h"
#include "gpu/mega_op_norm.h"
#include "gpu/mega_op_ple.h"

namespace tona::gpu {
namespace {

template <Q6Layout L>
struct Qwen4ExpProfile {
    static constexpr Q6Layout q6 = L;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;
    static constexpr const char* unit = "gpu/arch/qwen4exp/kernel.hip";

    // Hyper-connection norms, DeltaNet, QSA, drafter, MoE, PLE, MTP head. The order is the machine code's: the
    // first family that knows the step kind runs it.
    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_norm(op, c) || dispatch_dn<NB>(op, c) || dispatch_qsa(op, c) || dispatch_draft(op, c) ||
               dispatch_moe(op, c) || dispatch_ple(op, c) || dispatch_mtp(op, c);
    }
};

// The profile of the 512-expert Flash-Next (same architecture, same op families): only the size class of the router
// rows differs. A struct of its own, so that the Coder's profile keeps its name and its kernel its machine code.
template <Q6Layout L>
struct Qwen4Exp512Profile : Qwen4ExpProfile<L> {
    static constexpr int max_experts = MEGA_MAX_EXPERTS_WIDE;
    static constexpr const char* unit = "gpu/arch/qwen4exp/kernel_512.hip";
};

}  // namespace
}  // namespace tona::gpu
