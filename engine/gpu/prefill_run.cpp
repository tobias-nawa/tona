// PrefillRun: setup of the workspace and the borrowed slots, the order of the layers over the chunks, and the
// progress report.
#include "gpu/prefill_run.h"

#include <algorithm>
#include <cstdio>
#include <format>
#include <iterator>
#include <span>
#include <stdexcept>
#include <vector>

#include "backend/env.h"
#include "gpu/alloc_trace.h"
#include "gpu/arch/nemotron_h/prefill.h"
#include "gpu/arch/qwen35moe/prefill.h"
#include "gpu/arch/qwen4exp/prefill.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_order.h"
#include "model/progress.h"

namespace tona::gpu {

// The one place that maps the loaded model to its prefill sequence
const PrefillSequence& prefill_sequence(const ModelBase& m) {
    switch (m.arch()) {
        case ArchId::Qwen4Exp:
            return qwen4exp_prefill_sequence();
        case ArchId::Qwen35Moe:
            return qwen35moe_prefill_sequence();
        case ArchId::NemotronHMoe:
            return nemotron_h_prefill_sequence();
        default:
            throw std::runtime_error(std::string("gpu::Mega: no prefill kernels for ") +
                                     std::string(arch_name(m.arch())));
    }
}

PrefillRun::PrefillRun(Mega& m, const PfJob& job)
    : M(m),
      job_(job),
      D(prefill_sequence(m.m_).dims(m.m_, m.feat_layers_.size())),
      flags_(pf_flags()),
      C(job.plan.chunk),
      nsub((job.n + job.plan.chunk - 1) / job.plan.chunk),
      layered(job.plan.layered),
      pos_all(m.pos_),
      n(std::min(job.plan.chunk, job.n)),
      pos0(m.pos_),
      arena(m.pf_arena_),
      Rp(nsub, nullptr),
      Fp(nsub, nullptr) {
    for (const LayerCache& LC : M.cache_) max_sb = std::max(max_sb, LC.slot_bytes);
    arena_end = arena + M.pf_arena_bytes_;
    if (layered) borrow_slots();
    carve_buffers();
}

void PrefillRun::ensure_copy_stream() {
    if (M.pf_cs_) return;
    HIPCHECK(hipStreamCreateWithFlags(&M.pf_cs_, hipStreamNonBlocking));
    for (auto& ev : M.pf_ev_) HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
}

// Number of layers (from the back) whose slots hold `need` bytes; 0: not even the loan limit is enough
uint32_t PrefillRun::loan_fit(size_t need) const {
    const uint32_t L = M.h_.n_layer, kmax = M.pf_loan_layers();
    uint32_t k = 1;
    while (k <= kmax && M.slots_total_ - M.slot_off_[L - k] < need) k++;
    return k > kmax ? 0 : k;
}

uint32_t PrefillRun::loan_count(size_t need) const {
    const uint32_t k = loan_fit(need);
    if (k == 0)
        throw std::runtime_error(std::format(
            "gpu::Mega: {} MB for the layer-by-layer prefill do not fit into the borrowed slots (smaller TONA_PF_GROUP "
            "or TONA_PF_LCHUNK)",
            need >> 20));
    return k;
}

// Staging of the KV cache: with the cache in pinned RAM the attention kernel reads every selected cell over PCIe
// (128K prompt: 24.0 s against 6.2 s with the cache in VRAM, research/measurements/2026-10-30-prefill-attn.md). From
// layer 2 on a layer runs over all chunks and is finished afterwards, so one layer's KV in the borrowed slots is
// enough; the new rows go back to the pinned cache right after the layer. TONA_KV_STAGE=0 turns it off.
bool PrefillRun::staged(uint32_t l) const {
    if (!stage || l < 2 || !M.m_.has_kv(l)) return false;
    if (!M.kv_split_.planned()) return true;  // all in RAM
    size_t idx = 0;                           // allocation index of the layer's KV
    for (uint32_t k = 0; k < l; k++) idx += M.m_.has_kv(k);
    // A cache split by position is read from VRAM as long as the group stays before the split: staging starts after it
    return pos_all + job_.n > M.kv_split_.cells_in_vram(idx, SIZE_MAX, M.max_ctx_);
}

// Rows [c0, c1) of the layer's cache between the staging area (flat) and the cache, which may be split by position
// (kv_place.h: the cells before kv_split_.hot in VRAM, the others in the pinned part)
void PrefillRun::stage_copy(const PfLayer& F, int64_t c0, int64_t c1, bool in) {
    const size_t row = (size_t)M.kv_row_;
    const int64_t hot = F.attn.kv_cold ? M.kv_split_.hot : INT64_MAX;
    auto copy = [&](uint8_t* layer, int64_t from, int64_t to) {
        if (to <= from) return;
        uint8_t* flat = stage + (size_t)from * row;
        const size_t bytes = (size_t)(to - from) * row;
        HIPCHECK(in ? hipMemcpyAsync(flat, layer, bytes, hipMemcpyDefault, st)
                    : hipMemcpyAsync(layer, flat, bytes, hipMemcpyDefault, st));
    };
    copy(F.attn.kv + (size_t)c0 * row, c0, std::min(c1, hot));
    const int64_t s = std::max(c0, hot);
    if (F.attn.kv_cold) copy(F.attn.kv_cold + (size_t)(s - hot) * row, s, c1);
}

// The rows from earlier prefills and groups: the attention reads cells anywhere below pos_all
void PrefillRun::stage_in(const PfLayer& F) { stage_copy(F, 0, pos_all, true); }

// The rows of this group go to the pinned cache, which decode, stash, checkpoints and the other sequence read. Same
// stream as the kernels and as every later read of the cache (stream 0; run() synchronizes before finish()): no
// event needed, and nothing blocks while k_mega waits (k_mega runs only in mtp_pass/inject_draft after the sync).
void PrefillRun::stage_out(const PfLayer& F) { stage_copy(F, pos_all, pos_all + job_.n, false); }

// One line of TONA_ALLOC_TRACE for a range of the borrowed slots, with the group's position so that a fault address can
// be assigned to the exact range
void PrefillRun::trace_loan(const char* name, const void* p, size_t bytes) const {
    if (!trace::on()) return;
    char label[96];
    std::snprintf(label, sizeof label, "%s group_pos=%lld n=%d chunk=%d", name, (long long)pos_all, job_.n, C);
    trace::record("range", p, bytes, label);
}

// The loan must end inside the slots: the sizes are those of pf_layered_need
void PrefillRun::check_loan_end(const uint8_t* q) const {
    const uint8_t* end = M.slots_base_ + M.slots_total_;
    if (q > end)
        throw std::runtime_error(
            std::format("gpu::Mega: the borrowed range reaches {} bytes past the end of the slots", (size_t)(q - end)));
}

// The area of the unpacked dense weights (PrefillRun::wc_get) behind the arena; returns the end of it
uint8_t* PrefillRun::carve_wcache(uint8_t* q) {
    wc_size_ = job_.plan.wcache ? M.pf_wcache_bytes() : 0;
    if (!wc_size_) return q;
    trace_loan("wcache", q, wc_size_);
    wc_base_ = q;
    return q + (wc_size_ + 255) / 256 * 256;
}

// Layer by layer: workspace, residual streams per chunk, features of the chunks inside the drafter's window and two
// layer buffers of E experts each in one piece in the slots of the last layers (borrowed, refilled afterwards)
void PrefillRun::borrow_slots() {
    const size_t ab = (M.prefill_arena_bytes(C) + 511) / 256 * 256;
    const uint32_t L = M.h_.n_layer;
    const bool with_stage = job_.plan.stage;
    const uint32_t k = loan_count(M.pf_layered_need(job_.n, C, pos_all, job_.win0, with_stage, job_.plan.wcache));
    const size_t kv_bytes = with_stage ? ((size_t)(pos_all + job_.n) * M.kv_row_ + 255) / 256 * 256 : 0;
    if (tona::env::flag("TONA_PF_DEBUG") && M.pf_stage_on(pos_all + job_.n))
        std::fprintf(stderr, "Prefill group at %lld: KV staging %.0f MB, %u layers borrowed\n", (long long)pos_all,
                     kv_bytes / 1e6, k);
    // Loans kept from the previous group (finish() returns them only after the last group) are the last `have`
    // layers; only the layers below them are new. The arena then lies inside the loan, or reaches into the new layers.
    const uint32_t have = (uint32_t)M.pf_loans_.size();
    std::vector<PfLoan> added;
    for (uint32_t l = L - std::max(k, have); l < L - have; l++) {
        LayerCache& LC = M.cache_[l];
        added.push_back({l, std::move(LC)});
        const LayerCache& sv = added.back().saved;
        LC = LayerCache{};
        LC.slots = sv.slots, LC.slot_bytes = sv.slot_bytes, LC.off_up = sv.off_up, LC.off_down = sv.off_down;
    }
    M.pf_loans_.insert(M.pf_loans_.begin(), std::make_move_iterator(added.begin()),
                       std::make_move_iterator(added.end()));
    const size_t r_bytes = ((size_t)C * D.hcd * sizeof(float) + 255) / 256 * 256;
    const size_t f_bytes = ((size_t)C * std::max(D.nfeat, 1) * sizeof(float) + 255) / 256 * 256;
    uint8_t* q = M.slots_base_ + M.slot_off_[L - k];
    trace_loan("loan", q, M.slots_total_ - M.slot_off_[L - k]);
    arena = q, q += ab;
    arena_end = arena + ab;
    trace_loan("arena", arena, ab);
    q = carve_wcache(q);
    trace_loan("residual", q, nsub * r_bytes);
    for (int j = 0; j < nsub; j++) Rp[j] = reinterpret_cast<float*>(q), q += r_bytes;
    for (int j = 0; j < nsub; j++)
        if (M.drafter_ && D.nfeat > 0 && pos_all + (int64_t)(j + 1) * C > job_.win0)
            Fp[j] = reinterpret_cast<float*>(q), q += f_bytes;
    trace_loan("kv_stage", q, kv_bytes);
    if (kv_bytes) stage = q, q += kv_bytes;
    const int E = (int)M.experts_.max_experts();  // the buffers fit the widest layer
    trace_loan("layer_buffers", q, 2 * (size_t)E * max_sb);
    for (int b = 0; b < 2; b++) {
        lbuf[b].resize(E), res[b].assign(E, nullptr);
        for (int e = 0; e < E; e++) lbuf[b][e] = q, q += max_sb;
    }
    check_loan_end(q);
    ensure_copy_stream();
    if (!M.pf_lev_[0])
        for (auto& ev : M.pf_lev_) HIPCHECK(hipEventCreateWithFlags(&ev, hipEventDisableTiming));
    HIPCHECK(hipEventRecord(M.pf_lev_[0], st));
    HIPCHECK(hipEventRecord(M.pf_lev_[1], st));
}

// Fixed buffers, in the order of Mega::prefill_arena_bytes
void PrefillRun::carve_buffers() {
    Carve c{arena, 0, arena_end};
    const int d = D.d;
    B.e = c.take<float>((size_t)C * D.in_row);
    B.R = c.take<float>((size_t)C * D.hcd);
    B.sc = c.take<float>((size_t)C * 4);
    prefill_sequence(M.m_).carve_fixed(c, M.m_, D, C, &arch_run_);
    B.mixed = c.take<float>((size_t)C * d);
    B.y = c.take<float>((size_t)C * d);
    B.feat = c.take<float>((size_t)C * std::max(D.nfeat, 1));
    B.mixed16 = c.take<_Float16>((size_t)C * d);
    B.w16_elems = (size_t)3 * EB * D.ff * d;
    B.w16 = c.take<_Float16>(B.w16_elems);
    B.x8 = c.take<uint8_t>((size_t)C * M.pf_max_in());
    B.sx = c.take<float>(C);
    B.sw = c.take<float>(M.pf_max_rows());
    B.phase = arena + (c.off + 255) / 256 * 256;
    if (!layered) Rp[0] = B.R, Fp[0] = B.feat;
}

// Order of the work (prefill_order.h)
std::vector<std::pair<uint32_t, int>> PrefillRun::layer_order() const {
    return pf_layer_order(M.m_.n_blocks(), nsub, layered);
}

// A staged layer: its old rows go in before the first chunk, its new rows out after the last one
void PrefillRun::run_layer(uint32_t l, int j) {
    kv_now_ = staged(l) ? stage : nullptr;
    if (kv_now_ && j == 0) stage_in(M.pf_[l]);
    // From layer 2 all chunks of a layer run in a row (layer_order), so the unpacked weights are shared by them
    wc_on_ = layered && wc_size_ > 0 && l >= 2;
    if (wc_on_ && j == 0) wc_reset();
    prefill_sequence(M.m_).layer(*this, l, j);
    wc_on_ = false;
    if (kv_now_ && j == nsub - 1) stage_out(M.pf_[l]);
}

// Layer by layer the buffer shared by the chunks does not hold this step's input norm: the architecture remakes it
void PrefillRun::resume_input(const std::vector<std::pair<uint32_t, int>>& order, size_t i) {
    if (pf_needs_resume(order, i)) prefill_sequence(M.m_).resume(*this, order[i].first);
}

// TONA_PF_HASH=1 (diagnostic, synchronizes the stream): FNV-1a of the residual stream of chunk j after layer l, to find
// the first layer where two prefill plans (chunk by chunk, layer by layer) or two builds differ
void PrefillRun::hash_residual(uint32_t l) const {
    static const bool on = tona::env::flag("TONA_PF_HASH");
    if (!on) return;
    std::vector<float> h((size_t)n * D.hcd);
    HIPCHECK(hipStreamSynchronize(st));
    HIPCHECK(hipMemcpy(h.data(), B.R, h.size() * sizeof(float), hipMemcpyDeviceToHost));
    uint64_t x = 1469598103934665603ull;
    for (const uint8_t b : std::span<const uint8_t>(reinterpret_cast<const uint8_t*>(h.data()), h.size() * 4))
        x = (x ^ b) * 1099511628211ull;
    std::fprintf(stderr, "pfhash pos %lld layer %u: %016llx\n", (long long)pos0, l, (unsigned long long)x);
}

// Layer l of chunk j is the i-th step
void PrefillRun::step(size_t i, const std::vector<std::pair<uint32_t, int>>& order) {
    const auto [l, j] = order[i];
    n = std::min(C, job_.n - j * C);
    pos0 = pos_all + (int64_t)j * C;
    B.R = Rp[j];
    B.feat = Fp[j];
    // Layer by layer a layer follows another chunk's step: B.mixed16 holds that chunk's norm, not this chunk's input
    resume_input(order, i);
    run_layer(l, j);
    hash_residual(l);
    const uint32_t nblocks = M.m_.n_blocks();
    const bool last = l + 1 == nblocks;
    if (last && job_.nll) head(j);
    // open combine before the switch of the chunk (B.y and the open combine belong to the running one)
    if (i + 1 == order.size() || order[i + 1].second != j) prefill_sequence(M.m_).end_chunk(*this);
    // layer l done for all chunks: its buffer is free for l + 2
    bool done = true;
    for (size_t q = i + 1; q < order.size() && done; q++) done = order[q].first != l;
    // The expert buffers go by the ordinal among the MoE blocks (the block itself for a model with a MoE in every one)
    const int ord = M.m_.moe_ordinal(l);
    if (layered && done && ord >= 0) {
        HIPCHECK(hipEventRecord(M.pf_lev_[ord & 1], st));
        issue_layer((uint32_t)ord + 2);
    }
    // progress for /status, reported when the GPU gets there: per layer of all chunks or per chunk
    if ((layered && done) || (!layered && last)) {
        const int64_t at = M.pf_prog_base_ + (layered ? (int64_t)job_.n * (l + 1) / nblocks
                                                      : std::min<int64_t>(job_.n, (int64_t)(j + 1) * C));
        HIPCHECK(
            hipLaunchHostFunc(st, [](void* v) { progress::prefill_done((int64_t)(intptr_t)v); }, (void*)(intptr_t)at));
    }
}

void PrefillRun::run() {
    if (job_.nll) job_.nll->assign(job_.n, 0.0f);
    const auto order = layer_order();
    issue_layer(0);
    issue_layer(1);
    for (size_t i = 0; i < order.size(); i++) step(i, order);
    t_mtp = std::chrono::steady_clock::now();
    mtp_by_pass_ = M.mtp_on_ && !prefill_sequence(M.m_).mtp_entries(*this);
    if (wc_size_ && tona::env::flag("TONA_PF_DEBUG"))
        std::fprintf(stderr, "Prefill group at %lld: dense weight cache peak %.0f of %.0f MB\n", (long long)pos_all,
                     wc_peak_ / 1e6, wc_size_ / 1e6);
    HIPCHECK(hipStreamSynchronize(st));
    HIPCHECK(hipGetLastError());
    if (flags_.check)
        std::fprintf(stderr, "TONA_PF_CHECK: group at %lld: %lld checks, %lld violations\n", (long long)pos_all,
                     (long long)check_steps, (long long)check_bad);
    finish();
}

}  // namespace tona::gpu
