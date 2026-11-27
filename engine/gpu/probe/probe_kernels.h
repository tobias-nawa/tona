// Kernel sets for tests: the real megakernel (k_mega) with a test-only profile that runs only the steps under test, as
// a KernelSet like an architecture's (gpu/kernel_set.h), so a test launches it through the same table as Mega does and
// the steps run with the barrier machinery of production. They are built into tona_gpu_probes, which only tests link.
// Host-visible declarations; the kernels are in the .hip files of this directory.
#pragma once

#include "gpu/kernel_set.h"

namespace tona::gpu::probe {

// Variant of the full-attention steps (gpu/mega_op_fattn.h): head size, gate, QK norm, RoPE
struct FattnShape {
    int hd;
    bool gate, qk_norm, rope;
    bool operator==(const FattnShape&) const = default;
};
inline constexpr FattnShape kOrnith{256, true, true, true};   // 16 q heads, 2 KV heads
inline constexpr FattnShape kBolt{128, false, false, false};  // 32 q heads, 2 KV heads
inline constexpr FattnShape kBoltAll{128, true, true, true};  // every option at the small head size

// Chunks per (token, head) of the full-attention split (FA_SPL of mega_op_fattn.h; the probe unit checks it): sizes the
// buffer of partial results
inline constexpr int kFattnSplits = 64;

// The kernels of one of the three shapes above; throws for another
const KernelSet& fattn_kernels(const FattnShape& s);

// The steps of the QSA attention of the Coder (the path full attention is compared with in speed)
const KernelSet& qsa_attention_kernels();

// The plain residual step (gpu/mega_op_resnorm.h)
const KernelSet& resnorm_kernels();

// The matvec phase with the NVFP4 and IQ2_XXS loops (ExtraQuants::kNvfp4Iq2xxs in gpu/matvec_phase.h), no other step
const KernelSet& quant_matvec_kernels();

// The Mamba-2 steps (gpu/mega_op_mamba.h)
const KernelSet& mamba_kernels();

// The latent MoE steps (gpu/mega_op_moe_latent.h) and the sigmoid routing step
const KernelSet& latent_moe_kernels();

}  // namespace tona::gpu::probe
