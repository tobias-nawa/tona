// The sigmoid routing step as a kernel of its own (route_sig_step.h).
#include "gpu/mega_op_route_sig.h"
#include "gpu/probe/route_sig_step.h"

namespace tona::gpu::probe {
namespace {

__global__ void __launch_bounds__(THREADS) k_step(MegaOp op, MegaParams* P, int nb) {
    __shared__ __attribute__((aligned(16))) float sm[SM_FLOATS];
    Ctx c{};
    c.gw = (int64_t)blockIdx.x * MWAVES + threadIdx.x / WAVE;
    c.GW = (int64_t)gridDim.x * MWAVES;
    c.lane = threadIdx.x % WAVE;
    c.nb = nb, c.P = P, c.sm = sm;
    op_route_sigmoid(op, c);
}

}  // namespace

void launch_route_sig_step(const MegaOp& op, MegaParams* P, int nb) { k_step<<<1, THREADS>>>(op, P, nb); }

}  // namespace tona::gpu::probe
