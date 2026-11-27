// Ornith's MTP entries of the prompt: the KV of the MTP layer per GEMM instead of one megakernel pass per 8 tokens.
// Only K and V of the entries matter for later drafts (the q, the MoE and the head of an entry are never read), so a
// part is: [enorm(e) | hnorm(h)] -> eh_proj -> norm -> K and V -> preparation into the MTP cache.
#include <algorithm>
#include <climits>

#include "backend/omp_guard.h"
#include "gpu/arch/qwen35moe/prefill_step.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_res.h"
#include "model/quant.h"

namespace tona::gpu {

// Pinned host buffer for the embeddings of a part
void PrefillRun::Step<Qwen35MoePrefillTag>::mtp_host_buffer(int rows) {
    if (M.mtp_pf_e_rows_ >= (size_t)rows) return;
    if (M.mtp_pf_e_host_) (void)trace::host_free(M.mtp_pf_e_host_);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&M.mtp_pf_e_host_), (size_t)rows * D.d * sizeof(float),
                                hipHostMallocDefault));
    M.mtp_pf_e_rows_ = rows;
}

// One part of S tokens (from t0 in chunk j). hfirst: h of the position before the first token of the prompt
void PrefillRun::Step<Qwen35MoePrefillTag>::mtp_part(const Qwen35MtpBufs& mb, int j, int t0, int S,
                                                     const float* hfirst) {
    const Qwen35PfMtp& Q = Qwen35PfRes::of(M.pf_arch_).mtp;
    const TensorInfo& et = *M.m_.embedding_table();
    const int64_t erb = row_bytes_of(et.type, D.d);
    const int32_t* ids = r_.job_.tokens + (size_t)j * C + t0;
    HIPCHECK(hipStreamSynchronize(st));  // the pinned buffer of the previous part is transferred
    backend::OmpErrors err;              // dequantize_row throws for a type it does not know
#pragma omp parallel for schedule(static)
    for (int i = 0; i < S; i++)
        err.run([&] {
            dequantize_row(et.type, et.data + (int64_t)ids[i] * erb, M.mtp_pf_e_host_ + (int64_t)i * D.d, D.d);
        });
    err.rethrow();
    HIPCHECK(hipMemcpyAsync(mb.e, M.mtp_pf_e_host_, (size_t)S * D.d * sizeof(float), hipMemcpyHostToDevice, st));
    launch_mtp_in16({mb.e, r_.Rp[j], hfirst, Q.enorm, Q.hnorm, mb.x16, t0, 1, D.d, D.eps}, S, st);
    r_.gemmq_f16(Q.eh, mb.x16, 2 * D.d, mb.Rm, D.d, S);
    launch_res_norm16({mb.Rm, nullptr, Q.attn_norm, B.mixed16, D.d, D.eps}, S, st);
    r_.gemmq_f16(Q.k, B.mixed16, D.d, mb.tk, D.nkv * D.hd, S);
    r_.gemmq_f16(Q.v, B.mixed16, D.d, mb.tv, D.nkv * D.hd, S);
    const int64_t p0 = r_.pos_all + (int64_t)j * C + t0;
    launch_qsa_prep({nullptr, nullptr, nullptr, nullptr, mb.tk,   Q.k_norm, mb.tv,      M.mtp_kv_, M.kv_row_,
                     nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,  S,          p0,        0,
                     D.nkv,   D.hd,    0,       0,       INT_MAX, D.eps,    Q.rope_base},
                    st);
}

// Parts of C/2 tokens (room in the phase area). TONA_MTP_PF_MEGA=1: passes of the megakernel as before
bool PrefillRun::Step<Qwen35MoePrefillTag>::mtp_entries() {
    if (!M.mtp_on_ || flags_.mtp_mega) return false;
    const int Sm = std::max(128, C / 2 / 128 * 128);
    mtp_host_buffer(Sm);
    Carve p{B.phase, 0, arena_end};
    Qwen35MtpBufs mb;
    mb.e = p.take<float>((size_t)Sm * D.d);
    mb.x16 = p.take<_Float16>((size_t)Sm * 2 * D.d);
    mb.Rm = p.take<float>((size_t)Sm * D.d);
    mb.tk = p.take<float>((size_t)Sm * D.nkv * D.hd);
    mb.tv = p.take<float>((size_t)Sm * D.nkv * D.hd);
    const float* h0 = M.mtp_h_of(r_.pos_all - 1);
    if (h0 == M.mtp_bad_) h0 = nullptr, M.stats_.mtp_stale++;
    for (int j = 0; j < r_.nsub; j++) {
        const int nj = std::min(C, r_.job_.n - j * C);
        const float* hfirst = j == 0 ? h0 : r_.Rp[j - 1] + (int64_t)(C - 1) * D.hcd;
        for (int t0 = 0; t0 < nj; t0 += Sm) mtp_part(mb, j, t0, std::min(Sm, nj - t0), hfirst);
    }
    return true;
}

}  // namespace tona::gpu
