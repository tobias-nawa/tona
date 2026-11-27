// Host side of the megakernel launch: the kernel's resources, the barrier buffer, the limits of the waits, and the
// start and restart of a pass. The kernels themselves are in the architecture's translation unit
// (gpu/arch/<name>/kernel.hip) and reached through the KernelSet.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <format>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/evict_counter.h"
#include "gpu/mega.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_util.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

// Time limit of the barriers: 100 ms (CHAIN_TIMEOUT_MS) aborted rarely in ordinary steps (gr.up at position 904,
// qsa.prep), probably when the desktop briefly occupied the GPU. 1 s still detects real hangs.
#ifndef MEGA_TIMEOUT_MS
#define MEGA_TIMEOUT_MS 1000
#endif

// The blocks HIP finds resident for the instantiation (width class, tile), noted for TONA_MEGA_INFO
static int probe_kernel(const KernelSet& set, int ix, bool tile, int cus, KernelProbe& p) {
    p = set.probe(ix, {tile});
    if (env::text("TONA_MEGA_INFO"))
        std::fprintf(stderr, "k_mega<%d>%s: VGPRs %d, LDS %zu, scratch %zu, maxThreads %d, per WGP %d\n",
                     ix ? 4 * ix : 1, tile ? " (tile)" : "", p.vgprs, p.lds_bytes, p.scratch_bytes, p.max_threads,
                     p.blocks_per_cu);
    return p.blocks_per_cu * cus;
}

// Tables of the kernel's translation unit, occupancy of the three kernels and the number of blocks
void Mega::init_kernels() {
    // One table for the megakernel, the layered prefill and the MTP layer (vLLM #37435: an MTP head without the
    // extension lost all acceptance above the native length)
    const rope::YarnTable& yarn = rope::yarn_active(h_.rope_base);
    kernels_->init_tables(yarn);
    set_prefill_yarn(yarn);
    check_prefill_yarn(yarn);
    if (yarn.factor > 1.0)
        std::fprintf(stderr, "RoPE: YaRN factor %g, mscale %g (TONA_ROPE_YARN)\n", yarn.factor, yarn.mscale);
    int cus = 0;
    HIPCHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    // One block per CU (HIP counts WGPs of 2 CUs): 32 x 16 waves, 0 aborts in 101,000 launches. The occupancy
    // calculation of HIP is not reliable at ~180-192 VGPRs (182 VGPRs: 1 block per WGP, 185 and 192: 2), 32 blocks ran
    // with both kernels without an abort; so it is information only. If a block hangs, the barrier aborts.
    static_assert(BMAX == 8, "kernels for 1, up to 4 and up to 8 tokens");
    for (int i = 0; i < 3; i++) {
        kmax_[i] = 1 << 20;
        for (const bool tile : {false, true}) {
            // The numbers hold for both kernels: the smaller occupancy, the more registers
            KernelProbe p;
            kmax_[i] = std::min(kmax_[i], probe_kernel(*kernels_, i, tile, cus, p));
            kvgpr_[i] = std::max(kvgpr_[i], p.vgprs);
            lds_bytes_ = std::max(lds_bytes_, (int)p.lds_bytes);
        }
        kblocks_[i] = blocks_ > 0 ? blocks_ : 2 * cus - free_cus();
    }
    max_blocks_ = kmax_[0], vgprs_ = kvgpr_[0], blocks_ = kblocks_[0];
}

void Mega::init_barrier() {
    HIPCHECK(hipDeviceGetAttribute(&wall_khz_, hipDeviceAttributeWallClockRate, 0));
    HIPCHECK(trace::dev_malloc(&d_bar_, BAR_BYTES + BAR_HIST_BYTES));
    HIPCHECK(hipMemset(d_bar_, 0, BAR_BYTES + BAR_HIST_BYTES));
    // The blocks find the mailbox words of the eviction credit here (null: they abort at the deadline)
    if (evict_credit_enabled()) {
        const void* link = &mbox_dev_->ev_req;
        HIPCHECK(hipMemcpy(static_cast<uint8_t*>(d_bar_) + kEvictLinkOffset, static_cast<const void*>(&link),
                           sizeof link, hipMemcpyHostToDevice));
    }
}

// Limits of the waits for one launch (see WaitLimits). 2 ms between two polls: the largest gap of a wave that ran was
// under 64 us in 1.08e9 polls, none lay between 64 us and 2 ms, and the longer ones (2 ms to 100 ms) match the
// evictions of the same runs (research/measurements/2026-10-31-barrier-aborts.md, TONA_MEGA_GAP_HIST).
static MegaScalars scalars_for(const int2 steps, int timeout_ms, int wall_khz, void* bar, int64_t launches) {
    static const int64_t freeze_us = env::integer("TONA_MEGA_FREEZE_US", 2000);
    static const int64_t cap_ms = env::integer("TONA_MEGA_WAIT_CAP_MS", 5000);
    static const bool hist = env::flag("TONA_MEGA_GAP_HIST");
    // TONA_TEST_FREEZE_MS: in the 20th launch the waits at barrier TONA_TEST_FREEZE_GEN stand still for this long
    // (never longer than 1 s: the driver resets the GPU after 2 s)
    static const int64_t test_ms = std::min<int64_t>(env::integer("TONA_TEST_FREEZE_MS", 0), 1000);
    // TONA_TEST_FREEZE_ALL: all blocks stand still after that barrier (the waits behind it are short): tests the
    // deadline
    static const unsigned test_all = env::flag("TONA_TEST_FREEZE_ALL") ? 1 : 0;
    static const int64_t test_launch = env::integer("TONA_TEST_FREEZE_LAUNCH", 20);
    static const int64_t test_gen = env::integer("TONA_TEST_FREEZE_GEN", 300);
    const uint64_t khz = (uint64_t)wall_khz;
    // The first launch from TONA_TEST_FREEZE_LAUNCH on whose program has the barrier (the drafter's programs are
    // shorter)
    static bool fired = false;
    const bool test = test_ms > 0 && !fired && launches >= test_launch && steps.y > (int)test_gen + 1;
    if (test) {
        fired = true;
        std::fprintf(stderr, "gpu::Mega: test launch %lld: simulated freeze of %lld ms\n", (long long)launches,
                     (long long)test_ms);
    }
    return {steps.x,
            steps.y,
            khz * timeout_ms,
            wall_khz,
            (unsigned)test_gen,
            test_all,
            freeze_us > 0 ? khz * freeze_us / 1000 : ~0ull,
            khz * cap_ms,
            test ? khz * test_ms : 0,
            hist ? reinterpret_cast<unsigned long long*>(static_cast<uint8_t*>(bar) + BAR_BYTES) : nullptr};
}

void Mega::launch(int nb) {
    launch_program({nb, (int)ops_.size(), d_ops_, d_tasks_, d_params_, d_prof_}, 0, params_host_);
}

// Starts the kernel at step o0 with a clean barrier state; a holds the arguments of the program.
void Mega::start_kernel(const LaunchArgs& a, int o0) {
    mbox_host_->yield_seq = 0;  // no kernel reads the field now (yields exist only in programs with a router)
    HIPCHECK(hipMemsetAsync(static_cast<uint8_t*>(d_bar_) + 4 * BAR_YIELD, 0, 4, nullptr));
    HIPCHECK(hipMemsetAsync(d_bar_, 0, 32, nullptr));
    HIPCHECK(
        hipMemsetAsync(static_cast<uint8_t*>(d_bar_) + 64, 0xff, BAR_DONE - 64, nullptr));  // progress: not started
    HIPCHECK(hipMemsetAsync(static_cast<uint8_t*>(d_bar_) + BAR_DONE, 0, BAR_BYTES - BAR_DONE, nullptr));
    auto* bar = static_cast<unsigned*>(d_bar_);
    static const int timeout_ms = (int)env::integer("TONA_MEGA_TIMEOUT_MS", MEGA_TIMEOUT_MS);
    // TONA_MEGA_K8=1 (measuring mode): passes of up to 4 tokens also on k_mega<8>
    static const bool k8 = env::flag("TONA_MEGA_K8");
    const int width_class = a.nb <= 4 && !k8 ? kix(a.nb) : 2;
    const MegaScalars scalars = scalars_for({o0, a.o1}, timeout_ms, wall_khz_, d_bar_, stats_.launches);
    kernels_->launch(width_class, {tile_tasks_.contains(a.tasks)},
                     {a.ops, a.tasks, a.dparams, bar, a.prof, scalars, (unsigned)kblocks_[kix(a.nb)]});
    evict_begin();  // after the launch call: the kernel is already starting, a late baseline can only lower the credit
}

void Mega::launch_program(const LaunchArgs& a, int o0, const MegaParams* hparams) {
    last_drafter_ = a.ops == d_dops_;  // for abort_info: steps of the drafter, not of the target
    last_multi_ = a.ops == d_xops_;
    HIPCHECK(hipMemcpyAsync(a.dparams, hparams, sizeof(MegaParams), hipMemcpyHostToDevice, nullptr));
    last_launch_ = a;
    start_kernel(last_launch_, o0);
    stats_.launches++;
}

bool Mega::resume_if_yielded(uint64_t pass_seq) {
    if (__atomic_load_n(&mbox_host_->yield_seq, __ATOMIC_ACQUIRE) != (uint32_t)(pass_seq + 62)) return false;
    // all blocks end without the CPU (bar[2]); afterwards no kernel is running that waits for it
    HIPCHECK(hipStreamSynchronize(nullptr));
    const LaunchArgs a = last_launch_;
    const int o0 = (int)__atomic_load_n(&mbox_host_->yield_op, __ATOMIC_RELAXED);
    if (o0 < 0 || o0 >= a.o1) throw std::runtime_error(std::format("gpu::Mega: yield at step {} out of range", o0));
    // like launch_program, but without reloading the parameters: the kernel has already written into them (n_cells,
    // selection)
    start_kernel(a, o0);
    stats_.launches++, stats_.yields++;
    beat("resumed", -1);
    if (env::integer("TONA_MEGA_STALL_MS", 0) > 0)
        std::fprintf(stderr, "gpu::Mega: yield at step %d, position %lld, restarted at %lld ms\n", o0, (long long)pos_,
                     (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());
    return true;
}

}  // namespace tona::gpu
