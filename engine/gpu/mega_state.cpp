// Restore points and stashed state: save and load the state in VRAM, stash it to and from the host, reset.
#include <chrono>
#include <cstring>

#include "backend/snap_layout.h"
#include "backend/stall.h"
#include "gpu/guarded_copy.h"
#include "gpu/mega.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// KV and pool per QSA layer (device), the MTP head behind them; same order as kv_hptr_
std::vector<std::pair<uint8_t*, float*>> Mega::qsa_bufs() const {
    std::vector<std::pair<uint8_t*, float*>> v;
    for (uint32_t l = 0; l < m_.n_blocks(); l++)
        if (m_.has_kv(l)) v.emplace_back(pf_[l].attn.kv, pf_[l].attn.idx.pool);
    if (mtp_on_) v.emplace_back(mtp_kv_, mtp_pool_);
    return v;
}

namespace {
// Bytes of the indexer pool of one layer at pos rows; 0 for full attention
size_t pool_bytes_of(const MoeShape& h, int64_t pos) {
    return h.idx_dim ? backend::snap_pool_bytes(pos, h.idx_block, h.idx_dim) : 0;
}
}  // namespace

// Bytes of the first pos rows of KV buffer i (of n, the MTP layer last) that sit in VRAM although the layer has a
// pinned part: the cells before the split (kv_place.h); 0 for a layer entirely in RAM
size_t Mega::kv_vram_bytes(size_t i, size_t n, int64_t pos) const {
    if (!kv_split_.planned()) return 0;
    const int64_t cells = kv_split_.cells_in_vram(i, n - (mtp_on_ ? 1 : 0), max_ctx_);
    return (size_t)std::min(pos, cells) * (size_t)kv_row_;
}

const std::vector<int32_t>& Mega::snap_tokens_of(int s) const {
    static const std::vector<int32_t> none;
    if (s == cur_slot_) return snap_valid_ ? snap_tokens_ : none;
    return s >= 0 && s < (int)seq_host_.size() && seq_host_[s].snap_valid ? seq_host_[s].snap_tokens : none;
}

std::string Mega::batch_info() const {
    return std::to_string(kblocks_[1]) + " blocks (max " + std::to_string(kmax_[1]) + ", " + std::to_string(kvgpr_[1]) +
           " VGPRs)";
}

// Pinned copy of the drafter ring (allocated on first use)
void Mega::ensure_snap_dring(size_t bytes) {
    if (snap_dring_) return;
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&snap_dring_), bytes, hipHostMallocDefault));
    pin_allocs_.push_back(snap_dring_);
}

void Mega::save_drafter_ring() {
    ensure_snap_dring(dring_bytes());
    size_t db = 0;
    for (const auto& [p, bytes] : dring_bufs_) {
        HIPCHECK(hipMemcpyAsync(snap_dring_ + db, p, bytes, hipMemcpyDeviceToHost, nullptr));
        db += bytes;
    }
}

void Mega::save_state() {
    if (awaiting_commit_) throw std::runtime_error("gpu::Mega: save_state before commit");
    size_t off = 0;
    for (const auto& [p, bytes] : state_bufs_) {
        HIPCHECK(hipMemcpyAsync(snap_ + off, p, bytes, hipMemcpyDefault, nullptr));
        off += bytes;
    }
    // Drafter ring: the generation after the restore point overwrites places that the window there still needs
    if (!dring_bufs_.empty()) save_drafter_ring();
    beat("save-state", -1);
    HIPCHECK(hipStreamSynchronize(nullptr));
    snap_pos_ = pos_, snap_tokens_ = tokens_, snap_commit_ = pending_commit_, snap_parity_ = parity_;
    mtrk_.on_save();
    snap_valid_ = true;
}

void Mega::load_state() {
    if (!snap_valid_ || awaiting_commit_)
        throw std::runtime_error("gpu::Mega: load_state without a restore point or before commit");
    size_t off = 0;
    for (const auto& [p, bytes] : state_bufs_) {
        HIPCHECK(hipMemcpyAsync(p, snap_ + off, bytes, hipMemcpyDefault, nullptr));
        off += bytes;
    }
    if (snap_dring_) {
        size_t db = 0;
        for (const auto& [p, bytes] : dring_bufs_) {
            HIPCHECK(hipMemcpyAsync(p, snap_dring_ + db, bytes, hipMemcpyHostToDevice, nullptr));
            db += bytes;
        }
    }
    beat("load-state", -1);
    if (!GuardedCopy::get().wait_idle({nullptr}, backend::chunk_deadline_s(state_bytes()), nullptr))
        throw backend::CopyStalled("gpu::Mega: load_state stalled");
    pos_ = snap_pos_, tokens_ = snap_tokens_, pending_commit_ = snap_commit_, parity_ = snap_parity_;
    mtrk_.on_load(pos_);
    trunk_ok_ = false;
    batch_.clear();
}

// Stash and unstash run on the engine thread, the one that serves the mailbox of k_mega inside forward() and returns
// only when the kernel has ended (serve_pass, mega_serve.cpp). The kernel therefore never waits for the CPU while we
// are here, and the polling below is not the blocking call that AGENTS.md forbids in that state.
//
// The copies take at most chunk_deadline_s per chunk. On a stall the stash gives its host memory back (the measured
// way out, backend/stall.h), waits for the copy to end and throws CopyStalled: the caller carries on without it.
void Mega::stash(HostSnap& s) const {
    if (!snap_valid_) throw std::runtime_error("gpu::Mega: stash without a restore point");
    GuardedCopy& gc = GuardedCopy::get();
    const auto release = [&s] { s = HostSnap{}; };
    if (!gc.wait_idle({nullptr, pre_st_, pf_cs_}, backend::STREAM_IDLE_S, nullptr))
        throw backend::CopyStalled("gpu::Mega: stash stalled");
    s.tokens = snap_tokens_, s.pos = snap_pos_, s.commit = snap_commit_, s.parity = snap_parity_;
    size_t sb = 0;
    for (const auto& [p, bytes] : state_bufs_) sb += bytes;
    const size_t kvb = (size_t)snap_pos_ * kv_row_, poolb = pool_bytes_of(h_, snap_pos_);
    const auto qsa = qsa_bufs();
    const auto t_alloc = std::chrono::steady_clock::now();
    s.state.resize(sb), s.kv.resize(qsa.size() * kvb), s.pool.resize(qsa.size() * poolb);
    if (gc.prof_on())
        gc.prof_add_setup(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_alloc).count());
    // One pipeline for all regions: state, then KV rows and indexer pool of every QSA layer
    std::vector<GuardedCopy::Region> regions{{s.state.data(), snap_, sb}};
    for (size_t i = 0; i < qsa.size(); i++) {
        if (!kv_hptr_.empty() && kv_hptr_[i]) {
            const size_t hb = kv_vram_bytes(i, qsa.size(), snap_pos_);  // the first cells are in VRAM, the rest pinned
            if (hb) regions.push_back({s.kv.data() + i * kvb, qsa[i].first, hb});
            std::memcpy(s.kv.data() + i * kvb + hb, kv_hptr_[i], kvb - hb);
        } else {
            regions.push_back({s.kv.data() + i * kvb, qsa[i].first, kvb});
        }
        if (poolb) regions.push_back({s.pool.data() + i * poolb, qsa[i].second, poolb});
    }
    // HostSnap::state/kv/pool are not zero-filled (backend.h): they are complete only if this copy succeeded. Any other
    // way out gives them back, so no half-written snapshot can reach a restore or a checkpoint file.
    try {
        if (!gc.to_host(regions, release)) throw backend::CopyStalled("gpu::Mega: stash copy stalled");
    } catch (...) {
        s = HostSnap{};
        throw;
    }
    size_t db = 0;
    for (const auto& [p, bytes] : dring_bufs_) db += bytes;
    if (snap_dring_)
        s.dring.assign(snap_dring_, snap_dring_ + db);
    else
        s.dring.clear();
    gc.prof_report("stash");
}

size_t Mega::stash_bytes() const {
    if (!snap_valid_) return 0;
    size_t b = state_bytes() + (snap_dring_ ? dring_bytes() : 0);
    const size_t kvb = (size_t)snap_pos_ * kv_row_, poolb = pool_bytes_of(h_, snap_pos_);
    return backend::snap_total_bytes(b, qsa_bufs().size() * kvb, qsa_bufs().size() * poolb);
}

bool Mega::snap_fits(const HostSnap& s) const {
    if (!backend::snap_pos_ok(s, max_ctx_) || s.state.size() != state_bytes()) return false;
    if (s.commit < 0 || s.commit > MEGA_BMAX || (s.parity != 0 && s.parity != 1)) return false;
    if (!s.dring.empty() && s.dring.size() != dring_bytes()) return false;
    const size_t nq = qsa_bufs().size();
    const size_t kvb = (size_t)s.pos * kv_row_, poolb = pool_bytes_of(h_, s.pos);
    return backend::snap_kv_pool_ok(s, nq * kvb, nq * poolb);
}

// State, KV rows and indexer pool of every QSA layer from the stash into the device (or the pinned KV); false: stalled
bool Mega::restore_state(const HostSnap& s) {
    const size_t kvb = (size_t)s.pos * kv_row_, poolb = pool_bytes_of(h_, s.pos);
    const auto qsa = qsa_bufs();
    std::vector<GuardedCopy::Region> regions{{const_cast<uint8_t*>(s.state.data()), snap_, s.state.size()}};
    for (size_t i = 0; i < qsa.size(); i++) {
        if (!kv_hptr_.empty() && kv_hptr_[i]) {
            const size_t hb = kv_vram_bytes(i, qsa.size(), s.pos);
            if (hb) regions.push_back({const_cast<uint8_t*>(s.kv.data()) + i * kvb, qsa[i].first, hb});
            std::memcpy(kv_hptr_[i], s.kv.data() + i * kvb + hb, kvb - hb);
        } else {
            regions.push_back({const_cast<uint8_t*>(s.kv.data()) + i * kvb, qsa[i].first, kvb});
        }
        if (poolb) regions.push_back({const_cast<uint8_t*>(s.pool.data()) + i * poolb, qsa[i].second, poolb});
    }
    return GuardedCopy::get().to_device(regions, nullptr);  // the source is only read
}

// The source `s` stays alive until the copy has ended (the caller holds it). On a stall the slot is invalid: the
// restore point is partly overwritten, so it is dropped and CopyStalled tells the caller to compute from scratch.
void Mega::unstash(const HostSnap& s) {
    if (awaiting_commit_) throw std::runtime_error("gpu::Mega: unstash before commit");
    if (!snap_fits(s)) throw std::runtime_error("gpu::Mega: unstash with foreign sizes");
    GuardedCopy& gc = GuardedCopy::get();
    snap_valid_ = false;
    if (!gc.wait_idle({nullptr, pre_st_, pf_cs_}, backend::STREAM_IDLE_S, nullptr) || !restore_state(s))
        throw backend::CopyStalled("gpu::Mega: unstash stalled");
    if (!s.dring.empty()) {
        ensure_snap_dring(s.dring.size());
        std::memcpy(snap_dring_, s.dring.data(), s.dring.size());  // load_state writes it into VRAM
    }
    snap_tokens_ = s.tokens, snap_pos_ = s.pos, snap_commit_ = s.commit, snap_parity_ = s.parity;
    // Restore points are made after run_prompt: MTP caught up to pos - 1, h at pos - 1 in the state
    mtrk_.on_unstash(s.pos);
    snap_valid_ = true;
    load_state();
    gc.prof_report("unstash");
}

void Mega::reset() {
    HIPCHECK(hipDeviceSynchronize());
    for (const auto& [p, bytes] : state_bufs_) HIPCHECK(hipMemset(p, 0, bytes));
    pos_ = 0;
    tokens_.clear();
    batch_.clear();
    pending_commit_ = 0;
    parity_ = 0;
    awaiting_commit_ = false;
    mtrk_.on_reset();
    trunk_ok_ = false;
    stats_.nan_reported = false;  // one warning per request
}

}  // namespace tona::gpu
