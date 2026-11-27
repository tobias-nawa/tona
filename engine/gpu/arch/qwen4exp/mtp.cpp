// The MTP head as a third program for the same kernel: input (embedding of the token, h of the target or of the last
// MTP step) -> eh_proj per stream -> QSA block with its own KV -> MoE with all experts in VRAM and routing on the GPU
// -> its own head mixer -> LM head of the target. Intermediate buffers are those of the target (the programs run one
// after the other), the residual streams are its own.
#include <algorithm>
#include <string_view>

#include "backend/env.h"
#include "backend/mtp_vocab.h"
#include "backend/omp_guard.h"
#include "gpu/arch/qwen4exp/program.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

void Qwen4ExpProgram::build_mtp() {
    const MtpHead& M = *g_.mtp_;
    const Layer& S = M.L;
    PfGR pgr;
    g_.h_trunk_ = b_.R[cur_];  // h_nextn: residual of the target after the last layer, before the head mixer
    mtp_enter();
    mtp_input(S, M, &pgr);
    mtp_attention(S, M);
    combine(b_.y, b_.inject_a, &S.gr_ffn);
    gr(S.gr_ffn, b_.inject_f, &pgr);
    mtp_moe(S);
    combine(b_.y, b_.inject_f, &M.gr_head);
    g_.mtp_hout_ = b_.R[cur_];  // h for the next step
    gr(M.gr_head, nullptr, &pgr);
    mtp_head();
    mtp_pick_ops();
    mtp_leave();
}

// Switch the builder to the MTP program and give it its own residual streams
void Qwen4ExpProgram::mtp_enter() {
    const Dims& s = dm_;
    pb_.select_program({&g_.mops_, &g_.mtasks_, &g_.mlabels_}, false);
    g_.me_ = g_.alloc_b(s.d);
    g_.mtp_hin_ = g_.alloc((size_t)MEGA_BMAX * s.hd_all);
    g_.mtp_hlast_ = g_.alloc_state(s.hd_all);
    g_.mtp_bad_ = g_.mtp_hin_ + 1;  // marker, never read
    mtp_mx_ = g_.alloc_b((size_t)s.hc * 2 * s.d);
    mtp_mq_ = g_.alloc_q8_b((size_t)s.hc * 2 * s.d);
    float* mR0 = g_.alloc_b(s.hd_all);
    float* mR1 = g_.alloc_b(s.hd_all);
    mtp_router_ = g_.alloc_b(g_.mtp_->n_expert);
    g_.mlog_ = g_.alloc_b(h_.vocab);
    saved_R_[0] = b_.R[0], saved_R_[1] = b_.R[1], saved_cur_ = cur_;
    b_.R[0] = mR0, b_.R[1] = mR1, cur_ = 0;
}

void Qwen4ExpProgram::mtp_input(const Layer& S, const MtpHead& M, PfGR* pgr) {
    const Dims& s = dm_;
    res_->mtp.enorm = g_.upload_f32(*M.enorm), res_->mtp.hnorm = g_.upload_f32(*M.hnorm);
    res_->mtp.rope_base = M.rope_base;
    pb_.op(OP_MTP_IN, {g_.me_, (void*)res_->mtp.enorm, (void*)res_->mtp.hnorm, mtp_mx_, mtp_mq_.q, mtp_mq_.meta},
           {s.hc, s.d}, {s.eps});
    const void* weh = g_.upload(*M.eh_proj);
    res_->mtp.eh = {M.eh_proj->type, 2 * (int64_t)s.d, (int64_t)s.d, weh};
    pb_.mv_begin();
    for (int g = 0; g < s.hc; g++) {
        const Q8Vec xq{mtp_mq_.q + (int64_t)g * 2 * s.d, mtp_mq_.meta + (int64_t)g * 2 * s.d / 32};
        pb_.add({M.eh_proj->type, 2 * s.d, s.d, weh, mtp_mx_ + (int64_t)g * 2 * s.d, &xq, b_.R[0] + (int64_t)g * s.d});
    }
    pb_.mv_end("mtp.eh");
    pb_.op(OP_COMBINE_NORM,
           {b_.R[cur_], nullptr, nullptr, nullptr, (void*)g_.upload_f32(*S.gr_attn.norm), b_.xn, b_.qxn.q, b_.qxn.meta},
           {s.hc, s.d}, {s.eps});
    gr(S.gr_attn, b_.inject_a, pgr);
    res_->mtp.gr = *pgr;
}

void Qwen4ExpProgram::mtp_attention(const Layer& S, const MtpHead& M) {
    const Dims& s = dm_;
    const size_t kv_bytes = (size_t)g_.max_ctx_ * g_.kv_row_, pool_floats = (size_t)(g_.max_ctx_ / s.blk + 1) * s.idim;
    uint8_t* kv = pb_.alloc_kv(kv_bytes);
    if (g_.kv_host_) g_.kv_hptr_.push_back(static_cast<uint8_t*>(g_.pin_allocs_.back()));
    float* ring = g_.alloc_state((size_t)RAW_RING * s.idim);
    float* pool = g_.alloc(pool_floats);
    g_.seq_buf(kv, kv_bytes, g_.kv_host_ ? g_.kv_hptr_.back() : nullptr);
    g_.seq_buf(pool, pool_floats * sizeof(float));
    g_.mtp_kv_ = kv, g_.mtp_pool_ = pool, res_->mtp.ring = ring;
    PfMtp& Q = res_->mtp;
    // idx_k, k, v first: the program "KV only" (prompt tokens, g_.mkops_) computes only them
    Q.idx_k = pb_.pw(*S.idx_k, g_.upload(*S.idx_k));
    Q.k = pb_.pw(*S.k, g_.upload(*S.k));
    Q.v = pb_.pw(*S.v, g_.upload(*S.v));
    pb_.mv_begin();
    pb_.addw(*S.idx_k, Q.idx_k.w, b_.mixed, &b_.qmix, b_.t_rk);
    pb_.addw(*S.k, Q.k.w, b_.mixed, &b_.qmix, b_.t_k);
    pb_.addw(*S.v, Q.v.w, b_.mixed, &b_.qmix, b_.t_v);
    pb_.addw(*S.q, g_.upload(*S.q), b_.mixed, &b_.qmix, b_.t_qfull);
    pb_.addw(*S.idx_q, g_.upload(*S.idx_q), b_.mixed, &b_.qmix, b_.t_idxq);
    pb_.mv_end("mtp.qsa.in");
    const size_t kv_mv = g_.mops_.size() - 1;
    Q.k_norm = g_.upload_f32(*S.k_norm), Q.idx_k_norm = g_.upload_f32(*S.idx_k_norm);
    pb_.op(OP_QSA_PREP,
           {b_.t_qfull, b_.t_qn, b_.t_g, (void*)g_.upload_f32(*S.q_norm), b_.t_k, (void*)Q.k_norm, b_.t_rk, pool,
            (void*)Q.idx_k_norm, b_.t_idxq, (void*)g_.upload_f32(*S.idx_q_norm), kv, b_.t_v, ring},
           {s.nh, s.nkv, s.hd, s.idim, (int)h_.idx_heads, s.blk, (int)g_.kv_row_, (int)h_.idx_top_k},
           {s.eps, M.rope_base});
    // KV only: up to here, qsa.in without q and indexer q (prep then computes them from leftovers, unused)
    g_.mkops_.assign(g_.mops_.begin(), g_.mops_.end());
    g_.mkops_[kv_mv].b = g_.mkops_[kv_mv].a + 3;
    g_.mklabels_.assign(g_.mlabels_.begin(), g_.mlabels_.end());
    emit_attention(kv, pool);
    pb_.mv_begin();
    pb_.addw(*S.o, g_.upload(*S.o), b_.t_o, &b_.q_o, b_.y);
    pb_.mv_end("mtp.qsa.out");
}

void Qwen4ExpProgram::mtp_moe(const Layer& S) {
    const int d = dm_.d, ff = dm_.ff;
    const MtpHead& M = *g_.mtp_;
    pb_.mv_begin();
    pb_.addw(*S.router, g_.upload(*S.router), b_.mixed, &b_.qmix, mtp_router_);
    pb_.mv_end("mtp.router");
    if (!g_.mpick_) g_.mpick_ = g_.alloc(4 + MEGA_BMAX);  // pick {row, max, sum, NaN blocks}, then NaN per route wave
    pb_.op(OP_ROUTE, {mtp_router_, g_.mpick_ + 4}, {(int)M.n_expert, (int)M.n_used});
    g_.upload_mtp_experts();
    const int64_t bytes = g_.mtp_ex_bytes_;
    pb_.mv_begin();
    pb_.group({g_.mtp_ex_type_, d, ff, g_.mtp_ex_, b_.mixed, &b_.qmix, b_.eg}, bytes, 0, ff);
    pb_.group({g_.mtp_ex_type_, d, ff, g_.mtp_ex_ + g_.mtp_ex_up_, b_.mixed, &b_.qmix, b_.eu}, bytes, 0, ff);
    pb_.addw(*S.shexp_gate, g_.upload(*S.shexp_gate), b_.mixed, &b_.qmix, b_.eg + UMAX * ff);
    pb_.addw(*S.shexp_up, g_.upload(*S.shexp_up), b_.mixed, &b_.qmix, b_.eu + UMAX * ff);
    pb_.addw(*S.shexp_gate_inp, g_.upload(*S.shexp_gate_inp), b_.mixed, &b_.qmix, b_.sh_inp);
    pb_.mv_end("mtp.gate_up");
    pb_.op(OP_ACT_QUANT, {b_.eg, b_.eu, b_.act, b_.q_act.q, b_.q_act.meta}, {b_.act_n, ff});
    pb_.mv_begin();
    pb_.group({g_.mtp_ex_type_, ff, d, g_.mtp_ex_ + g_.mtp_ex_down_, b_.act, &b_.q_act, b_.ed}, bytes, ff, d);
    const Q8Vec xs{b_.q_act.q + UMAX * ff, b_.q_act.meta + UMAX * ff / 32};
    pb_.addw(*S.shexp_down, g_.upload(*S.shexp_down), b_.act + UMAX * ff, &xs, b_.sh_d);
    pb_.mv_end("mtp.down");
    pb_.op(OP_MOE_SUM, {b_.ed, b_.sh_d, b_.sh_inp, b_.y, nullptr}, {UMAX, d});
}

// LM head of the MTP step: truncated (default) or the full head of the target (TONA_MTP_VOCAB=full)
void Qwen4ExpProgram::mtp_head() {
    const char* vf = env::text("TONA_MTP_VOCAB");
    if (vf && std::string_view(vf) == "full") {
        pb_.mv_begin();
        pb_.addw(*m_.output(), g_.head_w_, b_.mixed, &b_.qmix, g_.mlog_);
        pb_.mv_end("mtp.head");
        return;
    }
    mtp_truncated_head(vf);
    const TensorInfo& ot = *m_.output();
    const bool dyn = mtp_dyn_setup();
    g_.mtp_dyn_.on = dyn;
    if (dyn) mtp_dyn_ops(row_bytes_of(ot.type, dm_.d));
    pb_.mv_begin();
    pb_.add({g_.mtp_head_type_, dm_.d, (int64_t)(g_.mtp_vocab_.n_static() + g_.mtp_vocab_.n_ctx()), g_.mtp_head_,
             b_.mixed, &b_.qmix, g_.mlog_});
    if (dyn)
        pb_.add({ot.type, dm_.d, (int64_t)g_.mtp_dyn_.rows, g_.mtp_dyn_.dynw, b_.mixed, &b_.qmix,
                 g_.mlog_ + g_.mtp_dyn_.first});
    pb_.mv_end("mtp.head");
}

// Only these rows of the LM head: the full one costs 675 MB per step. Behind the list TONA_MTP_DYN (8192) rows for
// tokens of the context that are not in the list (identifiers from the prompt: with the old list 5.9 % of the generated
// tokens were missing, mostly those); free: token -1
void Qwen4ExpProgram::mtp_truncated_head(const char* vf) {
    const int d = dm_.d;
    const TensorInfo& ot = *m_.output();
    // TONA_MTP_HEAD=iq4: copy in IQ4_NL (4.5 instead of 6.6 bit for Q6_K), otherwise in the type of the target
    const char* mode = env::text("TONA_MTP_HEAD");
    g_.mtp_head_q4_ = mode && std::string_view(mode) == "iq4";
    const uint32_t htype = g_.mtp_head_q4_ ? 20u : ot.type;
    const int64_t rb = row_bytes_of(htype, d);
    // Rows for tokens of the context: with the scorer the ones outside the ids it ranks, and prompts of a language the
    // list does not hold
    const int64_t dyn = std::max<int64_t>(0, env::integer("TONA_MTP_DYN", mtp_dyn_rank() > 0 ? 4096 : 8192));
    const backend::MtpList which = mtp_dyn_rank() > 0 ? backend::MtpList::Narrow : backend::MtpList::Wide;
    g_.mtp_vocab_ =
        backend::MtpVocabMap(backend::mtp_vocab_list(h_.vocab, which, vf, "gpu::Mega"), (size_t)dyn, h_.vocab);
    const size_t nstat = g_.mtp_vocab_.n_static();
    std::vector<uint8_t> rows((nstat + dyn) * rb);
    backend::OmpErrors err;  // a failing pread must reach the caller, not end the process
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nstat; i++) err.run([&] { g_.head_row(g_.mtp_vocab_[i], rows.data() + i * rb); });
    err.rethrow();
    g_.mtp_head_type_ = htype;
    void* p;
    HIPCHECK(trace::dev_malloc(&p, rows.size() + kMatvecReadPad));
    HIPCHECK(hipMemcpy(p, rows.data(), rows.size(), hipMemcpyHostToDevice));
    g_.allocs_.push_back(p);
    g_.vram_ += rows.size();
    g_.mtp_head_ = static_cast<uint8_t*>(p), g_.mtp_rb_ = rb;
    if (dyn > 0) {
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.mtp_rows_host_), dyn * rb, hipHostMallocDefault));
        g_.pin_allocs_.push_back(g_.mtp_rows_host_);
    }
}

// Token per row of the head to the device
void Qwen4ExpProgram::mtp_upload_map() {
    if (g_.mtp_vocab_.empty() || g_.mtp_map_dev_) return;
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.mtp_map_dev_), g_.mtp_vocab_.size() * sizeof(int32_t)));
    HIPCHECK(hipMemcpy(g_.mtp_map_dev_, g_.mtp_vocab_.data(), g_.mtp_vocab_.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice));
    g_.allocs_.push_back(g_.mtp_map_dev_);
}

void Qwen4ExpProgram::mtp_pick_ops() {
    if (!g_.mpick_) g_.mpick_ = g_.alloc(4 + MEGA_BMAX);
    mtp_upload_map();
    float* ppart = g_.alloc(3 * 64);  // per block (<= 2 * CUs)
    for (int phase = 0; phase < 2; phase++)
        pb_.op(OP_MTP_PICK, {g_.mlog_, g_.mtp_map_dev_, ppart, g_.mpick_},
               {(int)g_.mtp_rows(), (int)h_.vocab, phase, g_.mtp_dyn_.on ? 1 : 0});
}

// Back to the target program; the parameters and host buffers of the MTP passes
void Qwen4ExpProgram::mtp_leave() {
    b_.R[0] = saved_R_[0], b_.R[1] = saved_R_[1], cur_ = saved_cur_;
    pb_.select_program({&g_.ops_, &g_.tasks_, &g_.labels_}, true);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.mparams_host_), sizeof(MegaParams), hipHostMallocDefault));
    std::memset(g_.mparams_host_, 0, sizeof(MegaParams));
    g_.mparams_host_->budget_ms = g_.params_host_->budget_ms;
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.d_mparams_), sizeof(MegaParams)));
    HIPCHECK(hipMemset(g_.d_mparams_, 0, sizeof(MegaParams)));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.mtp_in_host_),
                                (size_t)MEGA_BMAX * 2 * dm_.d * sizeof(float), hipHostMallocDefault));
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.mpick_host_), (4 + MEGA_BMAX) * sizeof(float),
                                hipHostMallocDefault));
    g_.pin_allocs_.push_back(g_.mpick_host_);
}

}  // namespace tona::gpu
