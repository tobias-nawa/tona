// Ornith's layer code as a specialization of PrefillRun::Step (see prefill_run.h): one plain residual stream, gated
// DeltaNet and full-context gated attention layers, then the shared MoE.
#pragma once

#include "gpu/arch/qwen35moe/prefill.h"

namespace tona::gpu {

template <>
class PrefillRun::Step<Qwen35MoePrefillTag> {
public:
    explicit Step(PrefillRun& r);
    void body(uint32_t l, int j);
    // The streams of chunk j through the output norm: B.mixed16
    void head_input();
    // The norm of layer l's input over the residual stream of this chunk into B.mixed16 (PrefillSequence::resume)
    void resume(uint32_t l);
    // MTP KV entries of the prompt per GEMM (prefill_mtp.cpp); false when the head is off or TONA_MTP_PF_MEGA asks for
    // passes
    bool mtp_entries();

private:
    void embed_chunk(int j);
    bool attn_layer(const PfLayer& F, bool tail);
    void dn_layer(const PfLayer& F, int pending);
    void mtp_host_buffer(int rows);
    void mtp_part(const Qwen35MtpBufs& mb, int j, int t0, int S, const float* hfirst);

    PrefillRun& r_;
    const PfFlags& flags_;
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
