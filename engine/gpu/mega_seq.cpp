// Several sequences side by side: one copy of the state per slot, switching the active slot, and the program that
// runs the steps with state once per slot.
#include <cstring>
#include <set>

#include "backend/env.h"
#include "gpu/mega.h"
#include "gpu/mega_dump.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_run.h"

namespace tona::gpu {

void* Mega::seq_tr(const void* p, int from, int to) const {
    if (!p || from == to) return const_cast<void*>(p);
    const auto* a = static_cast<const uint8_t*>(p);
    for (const SeqBuf& b : seq_bufs_) {
        if (a >= b.dev[from] && a < b.dev[from] + b.bytes) return b.dev[to] + (a - b.dev[from]);
        if (b.host[from] && a >= b.host[from] && a < b.host[from] + b.bytes) return b.host[to] + (a - b.host[from]);
    }
    return nullptr;  // not a per-slot buffer
}

bool Mega::in_seq(const void* p) const { return p && seq_tr(p, 0, 1) != nullptr; }

// Copy of buffer b for slot s: pinned if the buffer lives in host memory, otherwise in VRAM; returns its size
size_t Mega::alloc_seq_buffer(SeqBuf& b, int s) {
    if (!b.host[0]) {
        b.dev[s] = reinterpret_cast<uint8_t*>(alloc(b.bytes / sizeof(float) + 16));
        return b.bytes;
    }
    void* h;
    HIPCHECK(trace::host_malloc(&h, b.bytes + 64, hipHostMallocMapped | hipHostMallocNonCoherent));
    std::memset(h, 0, b.bytes + 64);
    pin_allocs_.push_back(h);
    void* dptr;
    HIPCHECK(hipHostGetDevicePointer(&dptr, h, 0));
    b.host[s] = static_cast<uint8_t*>(h), b.dev[s] = static_cast<uint8_t*>(dptr);
    return b.bytes;
}

void Mega::alloc_mtp_seq_bufs() {
    mpick_x_ = alloc((size_t)MEGA_SMAX * 4 + MEGA_BMAX);  // per slot a pick, then NaN per route wave
    mtp_hsave_ = alloc((size_t)MEGA_SMAX * h_.n_hc * h_.d);
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&mpickx_host_), (MEGA_SMAX * 4 + MEGA_BMAX) * sizeof(float),
                                hipHostMallocDefault));
    pin_allocs_.push_back(mpickx_host_);
}

// Copies of the buffers for slots 1 ... n_seqs_ - 1 (same size and kind as in slot 0), plus a restore point in pinned
// RAM per slot (slot 0: snap_ in VRAM). The states (alloc_state) register here, KV and pool already during build.
void Mega::alloc_seqs() {
    for (const auto& [p, bytes] : state_bufs_) seq_buf(p, bytes);
    seq_host_.assign(n_seqs_, {});
    if (n_seqs_ <= 1) return;
    size_t sb = 0, vram = 0, pinned = 0;
    for (const auto& [p, bytes] : state_bufs_) sb += bytes;
    for (int s = 1; s < n_seqs_; s++) {
        for (SeqBuf& b : seq_bufs_) (b.host[0] ? pinned : vram) += alloc_seq_buffer(b, s);
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&seq_host_[s].snap), sb, hipHostMallocDefault));
        pin_allocs_.push_back(seq_host_[s].snap);
        pinned += sb;
    }
    if (mtp_on_) alloc_mtp_seq_bufs();
    pin_bytes_ += pinned;
    // scripts/bench-decode.sh reads "pinned" from this line
    if (env::text("TONA_VRAM_LOG"))
        std::fprintf(stderr, "Sequences: %d slots, %.0f MB VRAM and %.0f MB pinned per further slot\n", n_seqs_,
                     vram / 1e6 / (n_seqs_ - 1), pinned / 1e6 / (n_seqs_ - 1));
}

void Mega::swap_seq(SeqHost& h) {
    std::swap(h.pos, pos_), std::swap(h.tokens, tokens_), std::swap(h.batch, batch_);
    std::swap(h.snap_tokens, snap_tokens_), std::swap(h.pending_commit, pending_commit_), std::swap(h.parity, parity_);
    std::swap(h.pass_nb, pass_nb_), std::swap(h.snap_commit, snap_commit_), std::swap(h.snap_parity, snap_parity_);
    std::swap(h.fw_m, fw_m_), std::swap(h.awaiting_commit, awaiting_commit_);
    std::swap(h.snap_valid, snap_valid_), std::swap(h.trunk_ok, trunk_ok_), std::swap(h.snap_pos, snap_pos_);
    std::swap(h.fw_pos, fw_pos_), std::swap(h.fw_row0, fw_row0_), std::swap(h.mtp, mtrk_);
    std::swap(h.snap, snap_), std::swap(h.ops, d_ops_), std::swap(h.mops, d_mops_), std::swap(h.mkops, d_mkops_);
}

// Point every pointer into per-slot state from the buffer of slot `from` to the same place in slot `to`
void Mega::retarget_slot_pointers(int from, int to) {
    auto tr = [&](auto*& p) {
        if (!p) return;
        void* q = seq_tr(p, from, to);
        if (!q) throw std::runtime_error("gpu::Mega: state buffer without a copy per slot");
        p = static_cast<std::remove_reference_t<decltype(p)>>(q);
    };
    for (PfLayer& F : pf_) {
        tr(F.dn.state), tr(F.dn.ring), tr(F.dn.pend_d), tr(F.dn.pend_a), tr(F.dn.pend_k), tr(F.attn.kv),
            tr(F.attn.kv_cold), tr(F.attn.idx.raw_ring), tr(F.attn.idx.pool);
        tr(F.mamba.ssm), tr(F.mamba.ring), tr(F.mamba.pdec), tr(F.mamba.pxdt), tr(F.mamba.pb);
    }
    prefill_sequence(m_).retarget_state(pf_arch_, [&](float*& p) { tr(p); });
    tr(mtp_kv_), tr(mtp_pool_), tr(mtp_hlast_), tr(pf_ple_hist_);
    for (auto& [p, bytes] : state_bufs_) tr(p);
    for (uint8_t*& h : kv_hptr_) tr(h);
}

void Mega::use_slot(int s) {
    if (s < 0 || s >= n_seqs_) throw std::runtime_error(std::format("gpu::Mega: slot {} of {}", s, n_seqs_));
    if (s == cur_slot_) return;
    const int from = cur_slot_;
    retarget_slot_pointers(from, s);
    swap_seq(seq_host_[from]);  // put the running values away
    swap_seq(seq_host_[s]);     // fetch those of s
    cur_slot_ = s;
}

// A copy of a program for slot s: the pointers into per-slot buffers moved to that slot's copies
void* Mega::upload_slot_ops(const char* name, const std::vector<MegaOp>& ops, const std::vector<std::string>& labels,
                            int s) {
    std::vector<MegaOp> c = ops;
    for (MegaOp& o : c)
        for (void*& q : o.p)
            if (in_seq(q)) q = seq_tr(q, 0, s);
    void* d = nullptr;
    HIPCHECK(trace::dev_malloc(&d, c.size() * sizeof(MegaOp)));
    HIPCHECK(hipMemcpy(d, c.data(), c.size() * sizeof(MegaOp), hipMemcpyHostToDevice));
    allocs_.push_back(d);
    if (dump_) dump_->ops(std::format("slot{}.{}", s, name), c, labels);
    return d;
}

// Step j of op `k` points at a per-token buffer: forward_multi moves it to the first row of the slot
void Mega::expand_op(SeqExpand& e, size_t k, int sl) {
    const MegaOp& o = e.ops[k];
    MegaOp c = o;
    c.seq = sl;
    if (c.kind == OP_DN_CONV) c.p[12] = c.p[13] = nullptr, c.i[5] = 0;
    c.nobar = sl + 1 < n_seqs_ ? 1 : o.nobar == 2 ? 0 : o.nobar;
    if (c.kind == OP_MTP_PICK) {
        if (!e.ppart) e.ppart = alloc((size_t)MEGA_SMAX * 3 * 64);
        c.p[2] = e.ppart + sl * 3 * 64, c.p[3] = mpick_x_ + sl * 4;
    }
    if (c.kind == OP_ROUTE) c.p[1] = mpick_x_ + MEGA_SMAX * 4;
    for (int j = 0; j < 14; j++) {
        if (!c.p[j]) continue;
        if (in_seq(c.p[j]))
            c.p[j] = seq_tr(c.p[j], 0, sl);
        else if (const int64_t r = row_bytes_at(c.p[j]))
            e.patches.push_back({(int)e.out.size(), j, sl, static_cast<uint8_t*>(c.p[j]), r});
    }
    e.out.push_back(c), e.out_labels.push_back(std::format("{}#{}", e.labels[k], sl));
}

void Mega::expand_program(SeqExpand& e) {
    for (size_t k = 0; k < e.ops.size(); k++) {
        const MegaOp& o = e.ops[k];
        if (e.per_slot.contains(o.kind)) {
            for (int sl = 0; sl < n_seqs_; sl++) expand_op(e, k, sl);
            continue;
        }
        for (const void* q : o.p)
            if (in_seq(q))
                throw std::runtime_error(
                    std::format("gpu::Mega: step {} reads sequence state but does not run per slot", e.labels[k]));
        MegaOp c = o;
        if (c.nobar == 2) c.nobar = 0;
        e.out.push_back(c), e.out_labels.push_back(e.labels[k]);
    }
}

// Pinned host copy of an expanded program (forward_multi patches the row pointers there) and its device buffer
void Mega::publish_x_program(const std::vector<MegaOp>& ops, MegaOp*& host, void*& dev) {
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&host), ops.size() * sizeof(MegaOp), hipHostMallocDefault));
    pin_allocs_.push_back(host);
    std::memcpy(host, ops.data(), ops.size() * sizeof(MegaOp));
    HIPCHECK(trace::dev_malloc(&dev, ops.size() * sizeof(MegaOp)));
    allocs_.push_back(dev);
}

// Programs per slot: the same steps and tasks with the addresses of the slot (the tasks read no state, they stay
// shared). Program of several sequences: steps with state run per slot one after another (no barrier between them,
// they read and write separate rows and states); dependencies instead of barriers are off (their tables hold for
// the rows of one sequence).
void Mega::build_seq_progs() {
    if (n_seqs_ <= 1) return;
    for (const auto* tasks : {&tasks_, &mtasks_})
        for (const MegaTask& t : *tasks)
            if (in_seq(t.t.x) || in_seq(t.t.y) || in_seq(t.t.xq.q) || in_seq(t.t.xq.meta))
                throw std::runtime_error("gpu::Mega: matvec task reads or writes the state of a sequence");
    for (int s = 1; s < n_seqs_; s++) {
        seq_host_[s].ops = upload_slot_ops("ops", ops_, labels_, s);
        if (mtp_on_) {
            seq_host_[s].mops = upload_slot_ops("mops", mops_, mlabels_, s);
            seq_host_[s].mkops = upload_slot_ops("mkops", mkops_, mklabels_, s);
        }
    }
    static const std::set<int> stateful = {OP_DN_CONV,    OP_DN_RECUR,    OP_QSA_PREP,    OP_IDX_SCORE,  OP_TOPK,
                                           OP_ATTN,       OP_ATTN_SPLIT,  OP_ATTN_WMMA,   OP_ATTN_MERGE, OP_PLE_CONV,
                                           OP_FATTN_PREP, OP_FATTN_SPLIT, OP_FATTN_MERGE, OP_MAMBA_CONV, OP_MAMBA_SCAN};
    SeqExpand target{ops_, labels_, stateful, xops_, xlabels_, xpatch_};
    expand_program(target);
    publish_x_program(xops_, xops_host_, d_xops_);
    if (!mtp_on_) return;
    // MTP head: QSA with its own KV per slot, the pick (two phases) per slot with its own partial and result buffers
    static const std::set<int> mtp_stateful = {OP_QSA_PREP,   OP_IDX_SCORE,   OP_TOPK,       OP_ATTN,
                                               OP_ATTN_SPLIT, OP_ATTN_WMMA,   OP_ATTN_MERGE, OP_MTP_PICK,
                                               OP_FATTN_PREP, OP_FATTN_SPLIT, OP_FATTN_MERGE};
    SeqExpand head{mops_, mlabels_, mtp_stateful, xmops_, xmlabels_, xmpatch_};
    expand_program(head);
    publish_x_program(xmops_, xmops_host_, d_xmops_);
}

}  // namespace tona::gpu
