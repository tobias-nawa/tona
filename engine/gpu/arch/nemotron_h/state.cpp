#include "gpu/arch/nemotron_h/state.h"

#include "gpu/mega.h"
#include "gpu/prefill_mamba.h"

namespace tona::gpu {

namespace {
constexpr size_t kHalves = 2;  // the inputs of the last pass and of the one before: the halves alternate by parity
}

size_t NemotronHState::layer_floats(const Mamba2Dims& D) {
    const size_t ssm = (size_t)D.inner * D.state;  // heads * head_dim * state
    const size_t ring = (size_t)(D.conv - 1 + MEGA_BMAX) * D.conv_dim();
    const size_t pending = kHalves * MEGA_BMAX * (D.heads + D.inner + D.groups * D.state);
    return ssm + ring + pending;
}

PfMamba2 NemotronHState::alloc_layer(Mega& g, const Mamba2Dims& D) {
    mamba_check_shapes((int)D.heads, (int)(D.inner / D.heads), (int)D.state, (int)D.groups, (int)D.conv);
    PfMamba2 m;
    m.ssm = g.alloc_state((size_t)D.inner * D.state);
    m.ring = g.alloc_state((size_t)(D.conv - 1 + MEGA_BMAX) * D.conv_dim());
    m.pdec = g.alloc_state(kHalves * MEGA_BMAX * D.heads);
    m.pxdt = g.alloc_state(kHalves * MEGA_BMAX * D.inner);
    m.pb = g.alloc_state(kHalves * MEGA_BMAX * D.groups * D.state);
    return m;
}

}  // namespace tona::gpu
