// Bolt's block code as a specialization of PrefillRun::Step (see prefill_run.h): one plain residual stream, then per
// block a Mamba-2, an attention or a latent MoE. Blocks are indexed 0 ... 87; the MoE blocks are the driver's "layers"
// of the expert caches under their ordinal among the MoE blocks (NemotronPfRes::ordinal).
#pragma once

#include "gpu/arch/nemotron_h/prefill.h"

namespace tona::gpu {

template <>
class PrefillRun::Step<NemotronHPrefillTag> {
public:
    explicit Step(PrefillRun& r);
    void body(uint32_t l, int j);
    // The streams of chunk j through the output norm: B.mixed16
    void head_input();
    // The norm of block l's input over the residual stream of this chunk into B.mixed16 (PrefillSequence::resume)
    void resume(uint32_t l);

private:
    void embed_chunk(int j);
    void mamba_block(const PfMamba2& P, int pending);
    void attention_block(const PfGatedAttn& A);

    PrefillRun& r_;
    Mega& M;
    const PfDims& D;
    PfBufs& B;
    hipStream_t st;
    int C, n;
    int64_t pos0;
    const uint8_t* arena_end;
    uint8_t* kv_now_;
};

}  // namespace tona::gpu
