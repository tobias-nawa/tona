// Ornith's MTP head (blk.<n_layer> of the model file) as a third program for the same kernel: input (embedding of the
// token, h of the target or of the last MTP step) -> eh_proj -> one gated-attention layer with its own KV -> MoE with
// all experts in VRAM and routing on the GPU -> shared_head_norm -> the LM head of the target. One residual stream, so
// h is the stream itself. The intermediate buffers are those of the target (the programs run one after the other),
// the residual stream is its own. The host side (entries, drafting steps, several slots) is the shared one in
// gpu/mega_mtp.cpp.
#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string_view>

#include "backend/env.h"
#include "backend/mtp_vocab.h"
#include "backend/omp_guard.h"
#include "gpu/arch/qwen35moe/prefill.h"
#include "gpu/arch/qwen35moe/program.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "model/progress.h"

namespace tona::gpu {

void Qwen35MoeProgram::build_mtp() {
    if (!h_.n_mtp) throw std::runtime_error("gpu::Mega: the model file has no MTP layer");
    const Qwen35Mtp& M = m_.mtp();
    // The route op keeps its top-k and the expert weights in sm (as the Coder's MTP head checks at start)
    if (64 + 2 * (int)h_.n_expert + 32 + (int)h_.n_expert_used > SM_FLOATS)
        throw std::runtime_error("gpu::Mega: the MTP head has more experts than the route step holds");
    g_.h_trunk_ = b_.R;  // the residual of the target after the last layer, before the output norm
    mtp_enter();
    mtp_input(M);
    mtp_layer(M);
    mtp_moe(M.layer.moe);
    // h for the next draft is the stream after the layer, before the head norm
    residual_norm(b_.y, *M.shared_head_norm);
    g_.mtp_hout_ = mt_.R;
    mtp_head_and_pick();
    mtp_leave();
}

// Switch the builder to the MTP program and give it its own residual stream and input buffers
void Qwen35MoeProgram::mtp_enter() {
    const Dims& s = dm_;
    pb_.select_program({&g_.mops_, &g_.mtasks_, &g_.mlabels_}, false);
    g_.me_ = g_.alloc_b(s.d);
    g_.mtp_hin_ = g_.alloc((size_t)MEGA_BMAX * s.d);
    g_.mtp_hlast_ = g_.alloc_state(s.d);
    g_.mtp_bad_ = g_.mtp_hin_ + 1;  // marker, never read
    mt_.mx = g_.alloc_b((size_t)2 * s.d);
    mt_.mq = g_.alloc_q8_b((size_t)2 * s.d);
    mt_.e = g_.alloc_b(s.d);
    mt_.R = g_.alloc_b(s.d);
    mt_.router = g_.alloc_b(h_.n_expert);
    g_.mlog_ = g_.alloc_b(h_.vocab);
    mt_.saved_R = b_.R;
    b_.R = mt_.R;
}

// x = [enorm(e) | hnorm(h)] per token, eh_proj down to d, then the norm of the layer's first consumer
void Qwen35MoeProgram::mtp_input(const Qwen35Mtp& M) {
    const Dims& s = dm_;
    Qwen35PfMtp& Q = Qwen35PfRes::mut(g_.pf_arch_)
                         .mtp;  // what the prefill needs for the MTP entries of the prompt (prefill_mtp.cpp)
    Q.enorm = g_.upload_f32(*M.enorm), Q.hnorm = g_.upload_f32(*M.hnorm), Q.rope_base = s.base;
    const float* attn_norm = g_.upload_f32(*M.layer.attn_norm);
    Q.attn_norm = attn_norm;
    pb_.op(OP_MTP_IN, {g_.me_, (void*)Q.enorm, (void*)Q.hnorm, mt_.mx, mt_.mq.q, mt_.mq.meta}, {1, s.d}, {s.eps});
    const void* weh = g_.upload(*M.eh_proj);
    Q.eh = {M.eh_proj->type, 2 * (int64_t)s.d, (int64_t)s.d, weh};
    pb_.mv_begin();
    pb_.addw(*M.eh_proj, weh, mt_.mx, &mt_.mq, mt_.e);
    pb_.mv_end("mtp.eh");
    pb_.op(OP_RES_NORM, {mt_.e, nullptr, (void*)attn_norm, b_.R, b_.xn, b_.qxn.q, b_.qxn.meta}, {s.d}, {s.eps});
}

// The attention layer with its own KV; the program "KV only" (prompt tokens, g_.mkops_) ends after the step that writes
// K and V (the q it computes along is unused)
void Qwen35MoeProgram::mtp_layer(const Qwen35Mtp& M) {
    PfLayer F;
    layer_attention(M.layer, F);
    g_.mtp_kv_ = F.attn.kv;
    Qwen35PfMtp& Q = Qwen35PfRes::mut(g_.pf_arch_).mtp;
    Q.k = F.attn.k, Q.v = F.attn.v, Q.k_norm = F.attn.k_norm;
    size_t cut = g_.mops_.size();
    while (cut > 0 && g_.mops_[cut - 1].kind != OP_FATTN_PREP) cut--;
    g_.mkops_.assign(g_.mops_.begin(), g_.mops_.begin() + (std::ptrdiff_t)cut);
    g_.mklabels_.assign(g_.mlabels_.begin(), g_.mlabels_.begin() + (std::ptrdiff_t)cut);
    residual_norm(b_.y, *M.layer.post_attn_norm);
}

// All experts of the MTP layer into one buffer, per expert [gate | up | down] as the file has the matrices, each
// part on a 256-byte boundary like a slot of the cache
void Qwen35MoeProgram::mtp_experts_upload(const MoeWeights& S) {
    const int64_t d = dm_.d, ff = dm_.ff, E = h_.n_expert;
    const TensorInfo* src[3] = {S.gate_exps, S.up_exps, S.down_exps};
    const int64_t rows[3] = {ff, ff, d}, ins[3] = {d, d, ff};
    auto up256 = [](int64_t v) { return (v + 255) / 256 * 256; };
    int64_t per[3];
    for (int k = 0; k < 3; k++) per[k] = rows[k] * row_bytes_of(src[k]->type, ins[k]);
    mt_.up = up256(per[0]), mt_.down = mt_.up + up256(per[1]), mt_.bytes = mt_.down + up256(per[2]);
    const size_t total = (size_t)(E * mt_.bytes);
    void* p;
    HIPCHECK(trace::dev_malloc(&p, total + kMatvecReadPad));
    HIPCHECK(hipMemset(p, 0, total + kMatvecReadPad));
    g_.allocs_.push_back(p);
    g_.vram_ += total;
    mt_.experts = static_cast<uint8_t*>(p);
    const int64_t off[3] = {0, mt_.up, mt_.down};
    for (int k = 0; k < 3; k++)
        for (int64_t e = 0; e < E; e++) {
            HIPCHECK(hipMemcpy(mt_.experts + e * mt_.bytes + off[k], src[k]->data + e * per[k], (size_t)per[k],
                               hipMemcpyHostToDevice));
            progress::add((double)per[k]);
        }
}

// Router, routing on the GPU, the experts of the union, the shared expert and the sum
void Qwen35MoeProgram::mtp_moe(const MoeWeights& S) {
    const int d = dm_.d, ff = dm_.ff;
    pb_.mv_begin();
    pb_.addw(*S.router, g_.upload(*S.router), b_.xn, &b_.qxn, mt_.router);
    pb_.mv_end("mtp.router");
    if (!g_.mpick_) g_.mpick_ = g_.alloc(4 + MEGA_BMAX);  // pick {row, max, sum, NaN blocks}, then NaN per route wave
    pb_.op(OP_ROUTE, {mt_.router, g_.mpick_ + 4}, {(int)h_.n_expert, (int)h_.n_expert_used});
    mtp_experts_upload(S);
    const int64_t bytes = mt_.bytes;
    pb_.mv_begin();
    pb_.group({S.gate_exps->type, d, ff, mt_.experts, b_.xn, &b_.qxn, b_.eg}, bytes, 0, ff);
    pb_.group({S.up_exps->type, d, ff, mt_.experts + mt_.up, b_.xn, &b_.qxn, b_.eu}, bytes, 0, ff);
    pb_.addw(*S.shexp_gate, g_.upload(*S.shexp_gate), b_.xn, &b_.qxn, b_.eg + UMAX * ff);
    pb_.addw(*S.shexp_up, g_.upload(*S.shexp_up), b_.xn, &b_.qxn, b_.eu + UMAX * ff);
    pb_.addw(*S.shexp_gate_inp, g_.upload(*S.shexp_gate_inp), b_.xn, &b_.qxn, b_.sh_inp);
    pb_.mv_end("mtp.gate_up");
    pb_.op(OP_ACT_QUANT, {b_.eg, b_.eu, b_.act, b_.q_act.q, b_.q_act.meta}, {b_.act_n, ff});
    pb_.mv_begin();
    pb_.group({S.down_exps->type, ff, d, mt_.experts + mt_.down, b_.act, &b_.q_act, b_.ed}, bytes, ff, d);
    const Q8Vec xs{b_.q_act.q + UMAX * ff, b_.q_act.meta + UMAX * ff / 32};
    pb_.addw(*S.shexp_down, g_.upload(*S.shexp_down), b_.act + UMAX * ff, &xs, b_.sh_d);
    pb_.program().tasks->back().wide = 0;  // x only as int8 (mtp.act), as in the target's layers
    pb_.mv_end("mtp.down");
    pb_.op(OP_MOE_SUM, {b_.ed, b_.sh_d, b_.sh_inp, b_.y, nullptr}, {UMAX, d});
}

// Only these rows of the LM head (the full one is 417 MB per draft step, 83 % of the step): the token list shared with
// the Coder (same tokenizer) and, behind it, TONA_MTP_DYN rows for tokens of the context that are not in the list; the
// pick maps a row back to its token. TONA_MTP_VOCAB=full: the whole head. The draft only proposes; the target verifies,
// so the output distribution is the same.
void Qwen35MoeProgram::mtp_truncated_head(const char* vf) {
    const int d = dm_.d;
    const TensorInfo& ot = *m_.output();
    const int64_t rb = row_bytes_of(ot.type, d);
    const int64_t dyn = std::max<int64_t>(0, env::integer("TONA_MTP_DYN", 8192));
    g_.mtp_vocab_ = backend::MtpVocabMap(backend::mtp_vocab_list(h_.vocab, backend::MtpList::Wide, vf, "gpu::Mega"),
                                         (size_t)dyn, h_.vocab);
    const size_t nstat = g_.mtp_vocab_.n_static();
    std::vector<uint8_t> rows((nstat + dyn) * rb);
    backend::OmpErrors err;  // a failing pread must reach the caller, not end the process
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < nstat; i++) err.run([&] { g_.head_row(g_.mtp_vocab_[i], rows.data() + i * rb); });
    err.rethrow();
    g_.mtp_head_type_ = ot.type;
    void* p;
    HIPCHECK(trace::dev_malloc(&p, rows.size() + kMatvecReadPad));
    HIPCHECK(hipMemcpy(p, rows.data(), rows.size(), hipMemcpyHostToDevice));
    g_.allocs_.push_back(p);
    g_.vram_ += rows.size();
    g_.mtp_head_ = static_cast<uint8_t*>(p), g_.mtp_rb_ = rb;
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.mtp_map_dev_), g_.mtp_vocab_.size() * sizeof(int32_t)));
    HIPCHECK(hipMemcpy(g_.mtp_map_dev_, g_.mtp_vocab_.data(), g_.mtp_vocab_.size() * sizeof(int32_t),
                       hipMemcpyHostToDevice));
    g_.allocs_.push_back(g_.mtp_map_dev_);
}

// The head over the truncated (or all) rows, then the pick in two steps: the blocks reduce their slices, block 0 merges
void Qwen35MoeProgram::mtp_head_and_pick() {
    const char* vf = env::text("TONA_MTP_VOCAB");
    const bool full = vf && std::string_view(vf) == "full";
    pb_.mv_begin();
    if (full) {
        pb_.addw(*m_.output(), g_.head_w_, b_.xn, &b_.qxn, g_.mlog_);
    } else {
        mtp_truncated_head(vf);
        pb_.add({g_.mtp_head_type_, dm_.d, (int64_t)(g_.mtp_vocab_.n_static() + g_.mtp_vocab_.n_ctx()), g_.mtp_head_,
                 b_.xn, &b_.qxn, g_.mlog_});
    }
    pb_.mv_end("mtp.head");
    float* ppart = g_.alloc(3 * 64);  // per block (<= 2 * CUs)
    for (int phase = 0; phase < 2; phase++)
        pb_.op(OP_MTP_PICK, {g_.mlog_, g_.mtp_map_dev_, ppart, g_.mpick_},
               {(int)g_.mtp_rows(), (int)h_.vocab, phase, 0});
}

// Back to the target program; the parameters and host buffers of the MTP passes
void Qwen35MoeProgram::mtp_leave() {
    b_.R = mt_.saved_R;
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
