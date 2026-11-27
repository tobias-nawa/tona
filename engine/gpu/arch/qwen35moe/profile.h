// Kernel profile of Ornith (qwen35moe): the op families its program runs, in the order the kernel tries them, and the
// Q6_K layouts of its kernel instantiation (see KernelProfile in gpu/mega_kernel.h). One plain residual stream (no
// hyper-connections, no PLE, no indexer), DeltaNet with a silu readout gate, full-context gated attention, MoE with a
// shared expert. Only this architecture's kernel translation unit includes it.
#pragma once

#include "gpu/mega_kernel.h"
#include "gpu/mega_op_dn.h"
#include "gpu/mega_op_fattn.h"
#include "gpu/mega_op_moe.h"
#include "gpu/mega_op_mtp.h"
#include "gpu/mega_op_resnorm.h"

namespace tona::gpu {
namespace {

// Ornith's attention: 16 q heads and 2 KV heads of 256, gate, QK norm and RoPE on the first 64 dimensions
inline constexpr FattnCfg kOrnithAttention{256, true, true, true};

template <Q6Layout L>
struct Qwen35MoeProfile {
    static constexpr Q6Layout q6 = L;
    static constexpr int max_experts = MEGA_MAX_EXPERTS;

    template <int NB>
    static __device__ __forceinline__ bool dispatch(const MegaOp& op, const Ctx& c) {
        return dispatch_resnorm(op, c) || dispatch_dn<NB, DnGate::kSilu>(op, c) ||
               dispatch_fattn<kOrnithAttention>(op, c) || dispatch_moe(op, c) || dispatch_mtp(op, c);
    }
};

}  // namespace
}  // namespace tona::gpu
