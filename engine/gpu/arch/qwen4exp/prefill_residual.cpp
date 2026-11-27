// The Coder's residual in the prefill: four gated streams. Gated residual (norm of the streams, down, up, mix) and the
// combine of the previous sublayer, which waits until it is known whether a norm follows.
#include "backend/env.h"
#include "gpu/arch/qwen4exp/prefill_step.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

Qwen4ExpPfRes& Qwen4ExpPfRes::reset(std::any& store, size_t n_layer) {
    Qwen4ExpPfRes& r = store.emplace<Qwen4ExpPfRes>();
    r.layers.resize(n_layer);
    return r;
}

const Qwen4ExpPfRes& Qwen4ExpPfRes::of(const std::any& store) { return std::any_cast<const Qwen4ExpPfRes&>(store); }

Qwen4ExpRun& PrefillRun::Step<Qwen4ExpPrefillTag>::run_of(PrefillRun& r) {
    if (!r.arch_run_.has_value()) r.arch_run_.emplace<Qwen4ExpRun>();
    return std::any_cast<Qwen4ExpRun&>(r.arch_run_);
}

const Qwen4ExpPfFlags& Qwen4ExpPfFlags::get() {
    static const Qwen4ExpPfFlags f{tona::env::integer("TONA_PF_PAIR", 1) == 0, tona::env::flag("TONA_PF_UNFUSED")};
    return f;
}

bool qwen4exp_mix_fused(uint32_t n_hc, uint32_t d) {
    static const bool off = tona::env::integer("TONA_PF_MIX", 1) == 0;
    return !off && n_hc == 4 && d % 64 == 0;
}

// In the order the buffers had in the shared workspace layout: lo, gate (only with streams and without the mix in the
// up GEMM), the inject values of the two sublayers, then the f16 inputs
void qwen4exp_carve_fixed(Carve& c, const Hparams& h, int C, std::any* run) {
    const size_t hcd = (size_t)h.n_hc * h.d, rank = h.hc_rank;
    Qwen4ExpFixed x;
    x.lo = c.take<float>((size_t)C * rank);
    x.gate = c.take<float>(qwen4exp_mix_fused(h.n_hc, h.d) || h.n_hc == 1 ? 0 : (size_t)C * hcd);
    x.inj_a = c.take<float>((size_t)C * 4);
    x.inj_f = c.take<float>((size_t)C * 4);
    x.xn16 = c.take<_Float16>((size_t)C * hcd);
    x.lo16 = c.take<_Float16>((size_t)C * rank);
    if (!run) return;
    if (!run->has_value()) run->emplace<Qwen4ExpRun>();
    std::any_cast<Qwen4ExpRun&>(*run).fx = x;
}

void qwen4exp_dump_mtp(const Qwen4ExpPfRes& res, const PfDumpSink& out) {
    out.weight("mtp.eh", res.mtp.eh), out.weight("mtp.idx_k", res.mtp.idx_k), out.weight("mtp.k", res.mtp.k);
    out.weight("mtp.v", res.mtp.v);
}

// Two weights with the same input in one GEMM (rows of b behind those of a): the input is read once, every column
// computes as alone (bit-identical); TONA_PF_PAIR=0 turns it off
void PrefillRun::Step<Qwen4ExpPrefillTag>::pair(const PfW& a, const PfW& b, const GemmIn& in, const GemmOut& oa,
                                                const GemmOut& ob, int rows_t) {
    if (Qwen4ExpPfFlags::get().pair_off || a.in != b.in || (a.rows + b.rows) * a.in > (int64_t)B.w16_elems) {
        r_.gemmq(a, in.x, in.ld, oa.y, oa.ld, rows_t);
        r_.gemmq(b, in.x, in.ld, ob.y, ob.ld, rows_t);
        return;
    }
    bool fill = true;
    uint8_t* kept = r_.wc_get(a.w, 0, 2, (size_t)((a.rows + b.rows) * a.in) * sizeof(_Float16), fill);
    _Float16* wp = kept ? reinterpret_cast<_Float16*>(kept) : B.w16;
    if (fill) {
        dequant_f16({a.type, a.rows, a.in, a.in}, a.w, wp, st);
        dequant_f16({b.type, b.rows, b.in, b.in}, b.w, wp + a.rows * a.in, st);
    }
    GemmSeg s{in.x, wp, oa.y, rows_t, (int32_t)(a.rows + b.rows), (int32_t)a.in, in.ld, (int32_t)a.in, oa.ld};
    s.y2 = ob.y, s.n2 = (int32_t)a.rows, s.ldy2 = ob.ld;
    gemm_f16(&s, 1, st);
}

// R += y * 2 sigmoid(inject / hc) of the previous sublayer, if one is waiting
void PrefillRun::Step<Qwen4ExpPrefillTag>::flush() {
    if (!pend_inj_) return;
    launch_combine({B.R, B.y, pend_inj_, r_.n, hc_, D.d}, st);
    pend_inj_ = nullptr;
}

// combine waits until it is known whether a norm follows (then k_combine_norm16); TONA_PF_UNFUSED=1: separate
void PrefillRun::Step<Qwen4ExpPrefillTag>::combine(const float* inj) {
    pend_inj_ = inj;
    if (Qwen4ExpPfFlags::get().unfused) flush();
}

// Gated residual: norm of the streams, down (silu(. / hc)), up, mix; inject (optional) from xn
void PrefillRun::Step<Qwen4ExpPrefillTag>::gated_residual(const PfGR& G, float* inj) {
    const int n = r_.n;  // the MTP entries run on parts of the chunk
    if (pend_inj_) {
        launch_combine_norm16({B.R, B.y, pend_inj_, G.norm, X.xn16, B.sc, hc_, D.d, D.eps}, n, st);
        pend_inj_ = nullptr;
    } else {
        launch_norm16({B.R, G.norm, X.xn16, B.sc, hc_, D.d, D.eps}, n, st);
    }
    if (inj)
        pair(G.down, G.inject, {X.xn16, (int)D.hcd}, {X.lo, rank_}, {inj, 4}, n);
    else
        r_.gemmq(G.down, X.xn16, (int)D.hcd, X.lo, rank_, n);
    launch_act16({X.lo, X.lo16, (int64_t)n * rank_, 1.0f / hc_}, st);
    // up with the mix in the epilogue (gate is not written); TONA_PF_MIX=0: GEMM and k_mix separate
    if (qwen4exp_mix_fused(M.h_.n_hc, M.h_.d)) {
        bool fill = true;
        uint8_t* kept = r_.wc_get(G.up.w, 0, 3, (size_t)(G.up.rows * G.up.in) * sizeof(_Float16), fill);
        _Float16* wp = kept ? reinterpret_cast<_Float16*>(kept) : B.w16;
        if (fill) dequant_f16({G.up.type, G.up.rows, G.up.in, G.up.in}, G.up.w, wp, st);
        const GemmSeg s{X.lo16,        wp, X.gate, n, (int32_t)G.up.rows, (int32_t)G.up.in, rank_, (int32_t)G.up.in,
                        (int32_t)D.hcd};
        gemm_f16_mix(s, GemmMix{B.R, B.sc, G.norm, B.mixed, B.mixed16, (int32_t)D.d}, st);
    } else {
        r_.gemmq(G.up, X.lo16, rank_, X.gate, (int)D.hcd, n);
        launch_mix({B.R, B.sc, G.norm, X.gate, B.mixed, B.mixed16, n, hc_, D.d}, st);
    }
}

}  // namespace tona::gpu
