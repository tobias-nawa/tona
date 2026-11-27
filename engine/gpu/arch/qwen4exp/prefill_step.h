// The Coder's layer code as a specialization of PrefillRun::Step (see prefill_run.h): one object per layer call, built
// from the driver's state, plus the head input and the MTP entries of the prompt. Members are spread over
// prefill_layer.cpp (layers), prefill_mtp.cpp (MTP entries) and prefill_residual.cpp (the gated residual).
#pragma once

#include "gpu/arch/qwen4exp/prefill.h"

namespace tona::gpu {

template <>
class PrefillRun::Step<Qwen4ExpPrefillTag> {
public:
    explicit Step(PrefillRun& r);
    void body(uint32_t l, int j);
    // The open combine of the last sublayer goes before the chunk switches
    void end_chunk() { flush(); }
    // The streams of chunk j after the head's gated residual: B.mixed16
    void head_input(int j);
    // MTP KV entries of the prompt per GEMM; false when the head is off or TONA_MTP_PF_MEGA asks for passes
    bool mtp_entries();

private:
    // prefill_residual.cpp: the gated residual with its open combine; pend_inj_ (the combine that waits until it is
    // known whether a norm follows) lives in the run's architecture state and survives from layer to layer
    void gated_residual(const PfGR& G, float* inj);
    void combine(const float* inj);
    void flush();
    // Down and inject (same input) in one GEMM when they fit, else two (TONA_PF_PAIR=0)
    void pair(const PfW& a, const PfW& b, const GemmIn& in, const GemmOut& oa, const GemmOut& ob, int rows_t);
    static Qwen4ExpRun& run_of(PrefillRun& r);

    void embed_chunk(int j);
    void feature_taps(uint32_t l);
    void ple_step();
    bool qsa_layer(const PfLayer& F, bool tail);
    void select_blocks(const PfLayer& F, const QsaBufs& b);
    void dn_layer(const PfLayer& F, int pending);
    void mtp_host_buffer(int rows);
    void mtp_part(const MtpBufs& mb, int j, int t0, int S, const float* hfirst);

    PrefillRun& r_;
    const float*& pend_inj_;
    const Qwen4ExpFixed& X;
    const int hc_, rank_;  // streams and rank of the gated residual
    Mega& M;
    const PfDims& D;
    PfBufs& B;
    hipStream_t st;
    int C, n;
    int64_t pos0;
    const PfFlags& flags_;
    const uint8_t* arena_end;
    uint8_t* kv_now_;
};

}  // namespace tona::gpu
