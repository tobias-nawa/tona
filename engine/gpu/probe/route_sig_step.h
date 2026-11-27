// The sigmoid routing step of the megakernel (gpu/mega_op_route_sig.h) as a kernel of its own for
// tests/test_gpu_route_sig: one block runs the step on the given op and pass parameters. Built into
// tona_gpu_probe_route, which only tests link.
#pragma once

#include "gpu/mega_types.h"

namespace tona::gpu::probe {

// P: device memory holding the MegaParams of the pass (nb tokens; nseq > 0: route the rows of all tokens)
void launch_route_sig_step(const MegaOp& op, MegaParams* P, int nb);

}  // namespace tona::gpu::probe
