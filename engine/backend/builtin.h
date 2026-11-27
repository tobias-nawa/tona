// The backends linked into a program, registered in the order "auto" tries them (backend/registry.h). Programs call
// register_builtin() once at the start of main; the build defines TONA_HAVE_<NAME> for each backend it links.
#pragma once
#include "backend/registry.h"
#if TONA_HAVE_REFERENCE
#include "gpu/mega.h"
#endif
#if TONA_HAVE_METAL
#include "backend/metal/metal.h"
#endif

namespace tona::backend {

#if TONA_HAVE_REFERENCE
namespace builtin {
inline bool reference_probe(std::string&) { return true; }

// The reference exactly as before the registry: same constructor call, same options. Mega takes the model through
// ModelBase; the architecture's own program and kernels are chosen inside (gpu/mega_build.cpp).
inline std::unique_ptr<Backend> make_reference(const ModelBase& m, const CreateOptions& o) {
    return std::make_unique<gpu::Mega>(m,
                                       gpu::MegaOptions{-1, o.max_ctx, 0, true, o.feat_layers, o.drafter, o.mtp, true});
}

// ggml types with a kernel in the megakernel for the Coder: the cases of dispatch_float in gpu/matvec_phase.h (the int8
// and wide paths serve subsets of it) and the host-repacked Q6_K tile layout (1019, model/quant_tile.cpp). A type
// outside this list is a task that no kernel case would run, so the registry refuses the file at load. Keep it equal
// to dispatch_float when a case is added there.
inline registry::QuantSet reference_coder_types() { return {0, 1, 8, 12, 13, 14, 18, 20, 21, 22, 23, 30, 42, 1019}; }

// Ornith (qwen35moe) as its Q4_K_M file has it: F32 norms, Q4_K and Q6_K weights
inline registry::QuantSet reference_ornith_types() { return {0, 12, 14}; }

// Bolt (nemotron_h_moe): F32 norms and small tensors, Q8_0 (embedding, LM head, attention, latent projections), IQ2_XXS
// (experts) and NVFP4 (Mamba-2 projections); the kernel of Bolt carries the loops of the last two (ExtraQuants)
inline registry::QuantSet reference_bolt_types() { return {0, 8, 16, 40}; }
}  // namespace builtin
#endif

inline void register_builtin() {
#if TONA_HAVE_REFERENCE
    registry::add({"hip-gfx1200",
                   builtin::reference_probe,
                   {{ArchId::Qwen4Exp, builtin::reference_coder_types(), registry::Tier::Reference},
                    {ArchId::Qwen35Moe, builtin::reference_ornith_types(), registry::Tier::Maintained},
                    {ArchId::NemotronHMoe, builtin::reference_bolt_types(), registry::Tier::Maintained}},
                   builtin::make_reference});
#endif
#if TONA_HAVE_METAL
    registry::add(
        {"metal",
         metal::probe,
         {{ArchId::Qwen4Exp, registry::QuantSet(metal::weight_types()), registry::Tier::Experimental}},
         [](const ModelBase& m, const CreateOptions& o) { return metal::create(model_as<Qwen4ExpModel>(m), o); }});
#endif
}

}  // namespace tona::backend
