// Memory that is sized last: the per-sequence buffers, the prefill workspace, the restore point and the expert slots
// (they get all VRAM that is left), and the optional diagnostics of the programs.
#include <algorithm>
#include <cstdio>
#include <cstring>

#include "backend/budget.h"
#include "backend/env.h"
#include "gpu/mega_build.h"
#include "gpu/mega_dump.h"
#include "gpu/mega_extra.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_check.h"

namespace tona::gpu {

// Buffers per sequence (TONA_SEQS, 1 ... MEGA_SMAX, default 2) go before the expert slots, which get the rest; with
// the DFlash drafter and in the measuring mode TONA_STATE_HOST there is only one
void ProgramBuilder::alloc_slots() {
    g_.n_seqs_ = (int)std::clamp<int64_t>(env::integer("TONA_SEQS", 2), 1, MEGA_SMAX);
    if (g_.drafter_ || env::flag("TONA_STATE_HOST")) g_.n_seqs_ = 1;
    g_.alloc_seqs();
    alloc_prefill_arena();
    alloc_restore_point();
    size_slots();
    plan_elastic();
    install_slots();
}

// Workspace of the prefill before the slots; TONA_PREFILL_CHUNK=0 switches the prefill off, as does an architecture
// without prompt kernels
void ProgramBuilder::alloc_prefill_arena() {
    const int d = d_;
    g_.pf_chunk_ =
        prefill_ ? (int)std::clamp<int64_t>(env::integer("TONA_PREFILL_CHUNK", 2048) / 128 * 128, 0, 8192) : 0;
    // layered prefill: larger chunks in borrowed memory (more tokens per expert in the GEMM)
    g_.pf_lchunk_ = (int)std::clamp<int64_t>(env::integer("TONA_PF_LCHUNK", 6144) / 128 * 128, g_.pf_chunk_, 8192);
    if (g_.pf_chunk_ <= 0) return;
    // Allocated in install_slots, behind the slots (extra slots lie in it); counted here like every other buffer
    g_.pf_arena_bytes_ =
        ((g_.prefill_arena_bytes(g_.pf_chunk_) / sizeof(float) + 64) * sizeof(float) + 64 + 255) / 256 * 256;
    g_.vram_ += g_.pf_arena_bytes_;
    // two buffers: the CPU fills that of the next chunk while the GPU computes
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.pf_host_),
                                (size_t)2 * std::max(g_.pf_chunk_, g_.pf_lchunk_) * 2 * d * sizeof(float),
                                hipHostMallocDefault));
    // counts and offsets of the experts (prefill_check.h), behind them the targets per token (nll)
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&g_.pf_hostc_),
                                (route_count_words(expert_cap(g_.h_.n_expert)) + 8192) * sizeof(int32_t),
                                hipHostMallocDefault));
}

// The restore point before the slots, so they get the remaining VRAM
void ProgramBuilder::alloc_restore_point() {
    vram_before_snap_ = g_.vram_;
    snap_bytes_ = 0;
    for (const auto& [p, bytes] : g_.state_bufs_) snap_bytes_ += bytes;
    HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.snap_), snap_bytes_));
    g_.allocs_.push_back(g_.snap_);
    g_.vram_ += snap_bytes_;
}

// Without a fixed count: as many slots as the free VRAM per sysfs allows minus a reserve (hipMemGetInfo does not see
// the desktop)
void ProgramBuilder::size_slots() {
    per_layer_bytes_ = 0;
    for (const auto& C : g_.cache_) per_layer_bytes_ += C.slot_bytes;
    if (g_.slots_ <= 0) {
        const int64_t reserve = env::integer("TONA_VRAM_RESERVE_MB", 1024) << 20;
        const int64_t pad = (int64_t)h_.n_layer * kMatvecReadPad;
        // The workspace is not allocated yet but belongs to the VRAM that is spoken for
        int64_t avail = vram_free_sysfs() - reserve - pad - (int64_t)g_.pf_arena_bytes_;
        if (backend::vram_limit_gb() >= 0)  // limit TONA_VRAM_GB minus what is already in use
            avail = std::min(avail, (int64_t)(backend::vram_limit_gb() * 1e9) - (int64_t)g_.vram_ - pad);
        g_.slots_ = (int)std::clamp<int64_t>(avail / (int64_t)per_layer_bytes_, 0, g_.experts_.max_experts());
        if (g_.slots_ < (int)h_.n_expert_used)
            throw std::runtime_error(
                std::format("gpu::Mega: VRAM only fits {} expert slots (minimum {}; in use "
                            "{:.2f} GB, limit TONA_VRAM_GB {:.1f})",
                            g_.slots_, h_.n_expert_used, g_.vram_ / 1e9, backend::vram_limit_gb()));
    }
    if (env::text("TONA_VRAM_LOG"))
        std::fprintf(stderr,
                     "VRAM: weights and buffers %.2f GB (of which prefill workspace %.0f MB), restore point %.0f MB, "
                     "slots %d x %.1f MB, free per sysfs %.2f GB\n",
                     vram_before_snap_ / 1e9, g_.pf_arena_bytes_ / 1e6, snap_bytes_ / 1e6, g_.slots_,
                     per_layer_bytes_ / 1e6, vram_free_sysfs() / 1e9);
}

// TONA_VRAM_GOVERNOR=1: the last slots of every layer go into elastic generations (mega_elastic.h). The core keeps at
// least n_expert_used x BMAX slots when the experts run on the GPU (check_slots_fit: a pass of 8 tokens needs that
// many), and 32 (at least twice n_expert_used) in the CPU mode of production, where experts that find no slot run on
// the CPU. It gives up a share for the alignment of the blocks to granules (about one granule per block: both ends
// round out; measured 491 MB for 393 MB of slots) so that the reserve stays whole.
void ProgramBuilder::plan_elastic() {
    g_.elastic_steps_ = 0;
    if (!env::flag("TONA_VRAM_GOVERNOR") || g_.slots_ <= 0) return;
    const int k = (int)std::clamp<int64_t>(env::integer("TONA_VRAM_STEP_SLOTS", 4), 1, 16);
    const int core_min = g_.cpu_mode_ ? std::max(32, 2 * (int)h_.n_expert_used) : (int)h_.n_expert_used * MEGA_BMAX;
    const int64_t want_mb = env::integer("TONA_VRAM_ELASTIC_MB", 1536);
    const int want = (int)((want_mb << 20) / (int64_t)per_layer_bytes_ + 1);
    int steps = std::min(want, g_.slots_ - core_min) / k;
    const auto waste = [&](int n) {
        return (int)((int64_t)n * h_.n_layer * (int64_t)vmm_granule() / (int64_t)per_layer_bytes_ + 1);
    };
    while (steps > 0 && g_.slots_ - steps * k - waste(steps) < core_min) steps--;
    if (steps <= 0) {
        std::fprintf(stderr, "TONA_VRAM_GOVERNOR: %d slots leave no room above the core of %d, governor off\n",
                     g_.slots_, core_min);
        return;
    }
    g_.slots_ -= steps * k + waste(steps);
    g_.elastic_steps_ = steps, g_.elastic_k_ = k;
}

// One chunk of the core: a fixed mechanism granule, not an amount that depends on the user's memory. 32 MiB because
// building the 8.7 GB range from chunks of 2 to 64 MiB costs the same 0.15 s and a kernel over it the same 68 ms
// (research/measurements/2026-11-15-expert-tiers.md, vmm_chunks_build sweep); it is a multiple of the API granule
// (vmm_granule). TONA_SLOTS_CHUNKED=1 turns it on (off by default).
size_t ProgramBuilder::slots_chunk_bytes() const {
    constexpr size_t kChunk = 32ull << 20;
    return env::flag("TONA_SLOTS_CHUNKED") ? kChunk : 0;
}

// The prefill workspace follows the slots in the same allocation: the extra slots in it get indices above the base
// slots (mega_extra.h). With the governor the allocation is a mapping in reserved address space.
void ProgramBuilder::alloc_slots_and_arena() {
    // TONA_SLOTS_CHUNKED (off): the core of the slots and the workspace from chunks, one buffer object each, so
    // that the kernel can take back a chunk instead of 8 GB; needs the same mapping as the governor
    const size_t chunk = slots_chunk_bytes();
    if (g_.elastic_steps_ > 0 || chunk > 0) {
        ElasticConfig cfg;
        cfg.core_bytes = g_.slots_total_ + g_.pf_arena_bytes_;
        cfg.va_bytes = g_.elastic_steps_ > 0 ? (size_t)env::integer("TONA_VRAM_VA_GB", 512) << 30 : 0;
        cfg.chunk_bytes = chunk;
        cfg.step_slots = g_.elastic_k_, cfg.read_pad = kMatvecReadPad;
        g_.slots_base_ = g_.elastic_.open(cfg);
        if (!g_.slots_base_) g_.elastic_steps_ = 0;  // open() said why; the core alone is what is left
    }
    if (!g_.slots_base_) {
        HIPCHECK(trace::dev_malloc(reinterpret_cast<void**>(&g_.slots_base_), g_.slots_total_ + g_.pf_arena_bytes_));
        g_.allocs_.push_back(g_.slots_base_);
    }
    if (g_.pf_arena_bytes_ == 0) return;
    g_.pf_arena_ = g_.slots_base_ + g_.slots_total_;
    HIPCHECK(hipMemset(g_.pf_arena_, 0, g_.pf_arena_bytes_));
}

// One allocation for all layers: the slots of the last layers lie contiguously, the layered prefill borrows them as
// one range. The placeholder addresses in the tasks and in moe.copy become real.
void ProgramBuilder::install_slots() {
    g_.slot_off_.assign(h_.n_layer, 0);
    g_.slots_total_ = 0;
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        g_.slot_off_[l] = g_.slots_total_;
        g_.slots_total_ += (g_.cache_[l].slot_bytes * g_.slots_ + kMatvecReadPad + 255) / 256 * 256;
    }
    alloc_slots_and_arena();
    for (uint32_t l = 0; l < h_.n_layer; l++) {
        LayerCache& C = g_.cache_[l];
        void* p = g_.slots_base_ + g_.slot_off_[l];
        g_.vram_ += C.slot_bytes * g_.slots_;
        const uintptr_t fake = (uintptr_t)(l + 1) << 52;
        for (MegaTask& t : g_.tasks_) {
            const uintptr_t w = reinterpret_cast<uintptr_t>(t.t.w);
            if ((w >> 52) == l + 1) t.t.w = static_cast<uint8_t*>(p) + (w - fake);
        }
        for (MegaOp& o : g_.ops_)  // moe.copy
            for (void*& q : o.p)
                if ((reinterpret_cast<uintptr_t>(q) >> 52) == l + 1)
                    q = static_cast<uint8_t*>(p) + (reinterpret_cast<uintptr_t>(q) - fake);
        C.slots = static_cast<uint8_t*>(p);
        for (int s = g_.slots_ - 1; s >= 0; s--) C.free_slots.push_back(s);
        C.n_slots = g_.slots_;
    }
    install_extra_slots();
    grow_elastic();
}

// The planned generations, then the governor with the marks of the reserve
void ProgramBuilder::grow_elastic() {
    int made = 0;
    while (made < g_.elastic_steps_ && g_.elastic_.grow(g_.cache_, true)) made++;
    if (made == 0) return;
    g_.vram_ += (size_t)made * g_.elastic_.generation_bytes();
    backend::VramMarks marks;
    marks.reserve = env::integer("TONA_VRAM_RESERVE_MB", 1024) << 20;
    g_.elastic_.start_governor(marks, made);
    if (const char* f = env::text("TONA_VRAM_FAKE_FREE"); f && !g_.elastic_.load_fake(f))
        std::fprintf(stderr, "TONA_VRAM_FAKE_FREE: cannot read %s, the card is read instead\n", f);
    if (env::text("TONA_VRAM_LOG"))
        std::fprintf(
            stderr, "VRAM governor: core %d slots, %d elastic generations of %d slots (%.0f MB each), %d more fit\n",
            g_.slots_, made, g_.elastic_k_, g_.elastic_.generation_bytes() / 1e6, g_.elastic_.generations_left());
}

// TONA_PF_EXTRA_SLOTS=0: no extra slots (the workspace stays idle between prefills, as before)
void ProgramBuilder::install_extra_slots() {
    if (g_.pf_arena_bytes_ == 0 || !env::integer("TONA_PF_EXTRA_SLOTS", 1)) return;
    ExtraLayout in;
    in.slot_off = g_.slot_off_;
    for (const LayerCache& C : g_.cache_) in.slot_bytes.push_back(C.slot_bytes);
    in.arena_start = g_.slots_total_, in.arena_bytes = g_.pf_arena_bytes_;
    in.read_pad = kMatvecReadPad;
    in.base_slots = g_.slots_, in.max_slots = (int)g_.experts_.max_experts();
    in.ranks = (int)std::clamp<int64_t>(env::integer("TONA_PF_EXTRA_RANKS", 2), 1, 16);
    const auto blocks = plan_extras(in);
    install_extras(g_.cache_, blocks);
    g_.extras_on_ = !blocks[0].empty();
    if (env::text("TONA_VRAM_LOG") && g_.extras_on_) {
        size_t used = 0;
        for (size_t l = 0; l < blocks.size(); l++)
            for (const ExtraBlock& b : blocks[l]) used += (size_t)b.count * in.slot_bytes[l];
        std::fprintf(stderr,
                     "Extra slots: %d per layer in the prefill workspace (%.0f of %.0f MB, rest is alignment)\n",
                     g_.cache_[0].n_slots - g_.slots_, used / 1e6, g_.pf_arena_bytes_ / 1e6);
    }
}

// Arrival stamps of the blocks at the barriers: TONA_BAR_PROF=<file> writes them per pass, TONA_MEGA_ABORT_TRACE only
// keeps them for the report of an abort (arrival_info)
void ProgramBuilder::setup_arrival_stamps() {
    const size_t n = g_.ops_.size();
    const char* f = env::text("TONA_BAR_PROF");
    if (f && g_.arr_fp_) f = nullptr;
    if (!f && !env::flag("TONA_MEGA_ABORT_TRACE")) return;
    if (!g_.d_arr_) HIPCHECK(trace::dev_malloc(&g_.d_arr_, n * g_.kblocks_[1] * sizeof(unsigned long long)));
    g_.params_host_->arr = g_.d_arr_;
    if (!f) return;
    g_.arr_fp_ = std::fopen(f, "wb");
    if (FILE* lf = std::fopen(std::format("{}.labels", f).c_str(), "w")) {
        for (size_t o = 0; o < n; o++) std::fprintf(lf, "%s %d\n", g_.labels_[o].c_str(), (int)g_.ops_[o].nobar);
        std::fclose(lf);
    }
}

// Timestamps per step are always written by the kernel (one value per step, costs nothing measurable); they are
// evaluated at an abort (abort_info) and per pass with TONA_MEGA_PROF. TONA_MEGA_NOBAR_STEPS=a,b,...: no barrier after
// these steps (estimate for dependencies instead of barriers where the next step needs only local data; the output
// is meaningless). TONA_BAR_PROF=<file>: per pass the arrival of each block at each barrier (header nops, blocks, nb;
// then prof, arr).
void ProgramBuilder::setup_diagnostics() {
    const size_t n = g_.ops_.size();
    HIPCHECK(trace::dev_malloc(&g_.d_prof_, 2 * n * sizeof(unsigned long long)));
    HIPCHECK(hipMemset(g_.d_prof_, 0, 2 * n * sizeof(unsigned long long)));
    if (env::flag("TONA_MEGA_PROF")) g_.prof_us_.assign(n, 0.0);
    g_.params_host_->nobar_all = env::flag("TONA_MEGA_NOBAR");
    g_.params_host_->wide_min = g_.wide_min_;
    if (const char* ns = env::text("TONA_MEGA_NOBAR_STEPS")) {
        const std::string list = std::string(",") + ns + ",";
        int count = 0;
        for (size_t o = 0; o + 1 < n; o++)
            if (list.contains("," + g_.labels_[o] + ",")) g_.ops_[o].nobar = 1, count++;
        std::fprintf(stderr, "TONA_MEGA_NOBAR_STEPS: %d steps without a barrier\n", count);
    }
    setup_arrival_stamps();
}

// The programs of the target model and of the MTP head go to the device (the tasks of the MTP head twice: for the
// full program and for the one that only writes KV)
void ProgramBuilder::upload_programs() {
    g_.upload_prog(g_.ops_, g_.tasks_, g_.d_ops_, g_.d_tasks_);
    if (g_.mtp_on_) {
        g_.upload_prog(g_.mops_, g_.mtasks_, g_.d_mops_, g_.d_mtasks_);
        g_.upload_prog(g_.mkops_, g_.mtasks_, g_.d_mkops_, g_.d_mktasks_);
    }
    g_.build_seq_progs();
    if (g_.dump_) g_.dump_->state();
}

}  // namespace tona::gpu
