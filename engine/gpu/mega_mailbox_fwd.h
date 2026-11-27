// The name of the mailbox type and the expert limit it is sized for, without the struct (gpu/mega_types.h), so that
// headers that only pass a pointer need nothing else.
#pragma once

namespace tona::gpu {

// Experts per layer that the kernels, the mailbox and the host buffers of the Coder are sized for.
constexpr int MEGA_MAX_EXPERTS = 256;

// The size class of a model with more experts per layer (512-expert Flash-Next). It is a template parameter of the
// megakernel and of the route kernel of the prefill, so that the Coder's instantiations stay what they were (a
// larger constant in them cost 2 ms per pass in an earlier experiment); host buffers are sized per model.
constexpr int MEGA_MAX_EXPERTS_WIDE = 512;

// The size class a model with n experts per layer runs in
constexpr int expert_cap(unsigned n) {
    return n <= (unsigned)MEGA_MAX_EXPERTS ? MEGA_MAX_EXPERTS : MEGA_MAX_EXPERTS_WIDE;
}

template <int MaxE>
struct MegaMailboxT;
using MegaMailbox = MegaMailboxT<MEGA_MAX_EXPERTS>;

}  // namespace tona::gpu
