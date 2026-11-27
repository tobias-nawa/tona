// Bolt's prefill sequence (arch/nemotron_h/prefill_*.cpp), chosen by prefill_sequence() in prefill_run.cpp, and the
// phase buffers only it uses.
#pragma once

#include <any>
#include <vector>

#include "gpu/prefill_run.h"

namespace tona::gpu {

// Buffers of the Mamba-2 phase: the in-projection output, the conv output, the scan output, the decay and dt per
// head (da = dt * A and the scores of the chunked scan besides), the normed output as f16 for the out-projection
struct MambaBufs {
    float *zx, *conv, *y, *decay, *dt, *da, *cb;
    _Float16* o16;
};
struct MambaShape {
    int in_dim, conv_dim, inner, heads, groups;
};
MambaBufs carve_mamba(Carve& p, int C, const MambaShape& s);

// Buffers of the attention phase: K and V of the chunk and the attention output as f16 (q is the projection's output)
struct BoltAttnBufs {
    float *q, *tk, *tv;
    _Float16* o16;
};
BoltAttnBufs carve_bolt_attn(Carve& p, const PfDims& D, int C);

// Bolt's prefill records besides the shared descriptors: the gain of the norm in front of every block and of the output
// norm (one plain residual stream). Recorded while the program is built.
struct NemotronPfRes {
    std::vector<const float*> norm;
    std::vector<int> ordinal;  // per block: its ordinal among the MoE blocks, -1 for the others
    const float* head_norm = nullptr;
    static NemotronPfRes& reset(std::any& store, size_t n_block);
    static const NemotronPfRes& of(const std::any& store);
};

// Tag of PrefillRun::Step<NemotronHPrefillTag>
struct NemotronHPrefillTag {};

const PrefillSequence& nemotron_h_prefill_sequence();

}  // namespace tona::gpu
