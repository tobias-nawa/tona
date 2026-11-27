// The Coder's prefill sequence (arch/qwen4exp/prefill_*.cpp), chosen by prefill_sequence() in prefill_run.cpp, and the
// phase buffers only it uses: quantized self-attention, PLE, the MTP entries of the prompt.
#pragma once

#include <any>
#include <vector>

#include "gpu/prefill_run.h"

namespace tona::gpu {

// A gated residual of the hyper-connection stream: down and up projections, the inject weight and the stream norm
struct PfGR {
    PfW down, up, inject;
    const float* norm = nullptr;
};

// MTP entries of the prompt in the prefill: input, eh_proj, gr_attn, idx_k/k/v, qsa.prep
struct PfMtp {
    PfW eh, idx_k, k, v;
    PfGR gr;
    const float *enorm = nullptr, *hnorm = nullptr, *k_norm = nullptr, *idx_k_norm = nullptr;
    float* ring = nullptr;
    float rope_base = 0;
};

// Environment switches of the gated residual, read once
struct Qwen4ExpPfFlags {
    bool pair_off;  // TONA_PF_PAIR=0: down and inject as two GEMMs
    bool unfused;   // TONA_PF_UNFUSED=1: combine and norm as two kernels
    static const Qwen4ExpPfFlags& get();
};

// Up GEMM with the mix in the epilogue (gemm_f16_mix); TONA_PF_MIX=0 turns it off
bool qwen4exp_mix_fused(uint32_t n_hc, uint32_t d);

// Buffers of the quantized self-attention phase
struct QsaBufs {
    float *rk, *qfull, *tk, *tv, *iq, *qn, *g, *score;
    _Float16* o16;
    int32_t *cells, *ncells;
};
// Buffers of the PLE phase
struct PleBufs {
    _Float16* e16;
    float *pk, *pv, *pg, *pn;
};
// Buffers of the MTP entries of the prompt
struct MtpBufs {
    float *e, *Rm, *rk, *tk, *tv;
    _Float16* x16;
};

// The Coder's prefill records besides the shared descriptors: the gated residual (hyper-connection) before the mixer
// and before the MoE of every layer, and before the head, and the MTP head's entries. Recorded while the program is
// built (the target first, then the MTP head), read by the prefill.
struct Qwen4ExpPfRes {
    struct Layer {
        PfGR attn, ffn;
    };
    std::vector<Layer> layers;
    PfGR head;
    PfMtp mtp;
    static Qwen4ExpPfRes& reset(std::any& store, size_t n_layer);
    static const Qwen4ExpPfRes& of(const std::any& store);
};

// Fixed buffers of the gated residual in the prefill workspace (see PrefillSequence::carve_fixed): the down output
// (rank), the up output without the mix, the inject values of the two sublayers, and the f16 inputs of the GEMMs
struct Qwen4ExpFixed {
    float *lo = nullptr, *gate = nullptr, *inj_a = nullptr, *inj_f = nullptr;
    _Float16 *xn16 = nullptr, *lo16 = nullptr;
};

// What the Coder keeps across the layers of one prefill run (PrefillRun::arch_run_)
struct Qwen4ExpRun {
    const float* pend_inj = nullptr;
    Qwen4ExpFixed fx;
};

// The fixed buffers of the gated residual (Qwen4ExpFixed), carved from the workspace; with a store the pointers are
// kept in the run's state
void qwen4exp_carve_fixed(Carve& c, const Hparams& h, int C, std::any* run);
void qwen4exp_dump_mtp(const Qwen4ExpPfRes& res, const PfDumpSink& out);

QsaBufs carve_qsa(Carve& p, const PfDims& D, int C, int64_t sstride);
PleBufs carve_ple(Carve& p, const PfDims& D, int C);

// Tag of PrefillRun::Step<Qwen4ExpPrefillTag>
struct Qwen4ExpPrefillTag {};

const PrefillSequence& qwen4exp_prefill_sequence();

}  // namespace tona::gpu
