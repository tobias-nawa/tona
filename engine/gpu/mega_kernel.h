// The megakernel: every block runs the ops of the program one after the other, with a grid barrier (or a dependency
// wait) behind each. The architecture is a template parameter, the kernel profile (see KernelProfile): it names the
// op families of the model, so this header includes none of them. Part of the architecture's kernel translation unit
// (gpu/arch/<name>/kernel.hip, same unit, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include <concepts>

#include "gpu/kernel_set.h"
#include "gpu/mega_op_common.h"
#include "gpu/mega_op_matvec.h"
#include "gpu/mega_op_wait.h"

namespace tona::gpu {
namespace {

// What an architecture gives k_mega: the Q6_K layouts its kernel has loops for (matvec_phase.h) and the dispatch of
// its op families, tried in the order of its profile. A family the model does not run is not named by its profile, so
// its code does not exist in the kernel (the unused Q6_K tile cases alone cost every pass 9 %, see Q6Layout).
template <class P>
concept KernelProfile = requires(const MegaOp& op, const Ctx& c) {
    { P::q6 } -> std::convertible_to<Q6Layout>;
    { P::max_experts } -> std::convertible_to<int>;  // experts per layer of the router rows the kernel copies
    { P::template dispatch<1>(op, c) } -> std::same_as<bool>;
};

// The formats beyond the base set a profile's kernel has loops for (ExtraQuants in matvec_phase.h); a profile without
// the member `quants` has none, so the Coder's and Ornith's kernels are the ones they were.
template <class P>
constexpr ExtraQuants profile_quants() {
    if constexpr (requires { P::quants; })
        return P::quants;
    else
        return ExtraQuants::kNone;
}

// Everything the steps need from the kernel arguments
struct MegaRun {
    int o0, o1;
    MegaParams* P;
    unsigned* bar;
    unsigned long long *start, *acc;  // inside the barrier buffer: bytes 16 and 32
    uint64_t timeout_ticks;
    unsigned long long* prof;  // optional: block 0 stamps start and end of every step
    int wall_khz;
    MegaScalars sc;
};

// The limits of a wait of this step: its own time limit (0: the general one) and the deadline of the pass
__device__ __forceinline__ WaitLimits wait_limits(const MegaRun& r, uint64_t lim, uint64_t deadline) {
    return {lim,
            deadline,
            r.sc.freeze_ticks,
            r.sc.cap_ticks,
            r.sc.test_ticks,
            r.sc.test_gen,
            (unsigned)(r.wall_khz / 1000),
            r.sc.test_all,
            reinterpret_cast<unsigned long long*>(r.bar + BAR_DIAG / 4),
            r.sc.hist};
}

// Per-block constants of the pass that the end of a step needs
struct StepFlags {
    unsigned long long* arr;  // optional: arrival time of every block at every step
    bool nobar_all, stamp;
};

__device__ __forceinline__ void init_ctx(Ctx& c, MegaParams* P, float* sm, int wall_khz) {
    c.gw = (int64_t)blockIdx.x * MWAVES + threadIdx.x / WAVE;
    c.GW = (int64_t)gridDim.x * MWAVES;
    c.gt = (int64_t)blockIdx.x * THREADS + threadIdx.x;
    c.GT = (int64_t)gridDim.x * THREADS;
    c.lane = threadIdx.x % WAVE;
    c.P = P;
    c.pos = P->pos;
    c.nb = P->nb;
    c.commit = P->commit, c.parity = P->parity, c.r0 = 0;
    c.sm = sm;
    // Deadline of the whole pass counted from the start of this block (all blocks start within microseconds)
    c.deadline = P->budget_ms > 0 ? wall_clock64() + (uint64_t)P->budget_ms * wall_khz : ~0ull;
}

// Width and position of the step. cond 1: steps over the fed-in tokens (drafter). With several sequences, steps with
// seq >= 0 compute only the rows of their slot, at its position.
__device__ __forceinline__ void step_ctx(Ctx& c, const MegaOp& op, MegaParams* P, bool multi) {
    c.nb = op.cond == 1 ? P->commit : P->nb;
    if (multi) {
        const int s = op.seq;
        c.pos = s >= 0 ? P->spos[s] : P->pos, c.r0 = s >= 0 ? P->sb0[s] : 0;
        c.commit = s >= 0 ? P->scommit[s] : P->commit, c.parity = s >= 0 ? P->sparity[s] : P->parity;
        if (s >= 0) c.nb = P->snb[s];
    }
}

// The step: the first group that knows its kind runs it
template <int NB, KernelProfile Prof>
__device__ __forceinline__ void run_op(const MegaOp& op, const Ctx& c, const StepEnv env) {
    if (op.kind == OP_MATVEC) {
        run_matvec<NB, Prof::q6, profile_quants<Prof>()>(op, c, env);
        return;
    }
    if (Prof::template dispatch<NB>(op, c)) return;
    dispatch_host<Prof::max_experts>(op, c, env);
}

// After an abort of a barrier or dependency wait: record the first aborted step and tell the CPU at once (it would
// otherwise wait at the next router), except after a yield
__device__ __forceinline__ void report_abort(MegaParams* P, unsigned* bar, int o) {
    if (threadIdx.x == 0) {
        atomicCAS(&bar[3], 0u, (unsigned)(o + 1));
        if (P->aborted && !__hip_atomic_load(&bar[BAR_YIELD], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)) {
            __threadfence_system();
            __hip_atomic_store(P->aborted, (uint32_t)(P->seq + 63), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        }
    }
}

// Wait for the blocks the next step depends on instead of a barrier (MegaOp::nobar = 2); false after an abort
__device__ __forceinline__ bool step_dependency(const MegaOp* next, int o, uint64_t deadline, const MegaRun r) {
    const uint64_t* masks = (const uint64_t*)next->p[13];
    const uint64_t mask = masks[(r.P->nb - 1) * 64 + blockIdx.x];
    if (dep_wait(r.bar, r.acc, mask, (unsigned)(o + 1), wait_limits(r, r.timeout_ticks, deadline))) return true;
    report_abort(r.P, r.bar, o);
    return false;
}

// Barrier behind the step with the time limit of the step (f[3] in ms, e.g. behind the wait for the CPU) or the
// general one. False after an abort.
__device__ __forceinline__ bool step_barrier(const MegaOp& op, int o, uint64_t deadline, const MegaRun r) {
    const uint64_t lim = op.f[3] > 0 ? (uint64_t)(op.f[3] * r.wall_khz) : r.timeout_ticks;
    if (!grid_barrier(r.bar, r.acc, gridDim.x, wait_limits(r, lim, deadline))) {
        report_abort(r.P, r.bar, o);
        return false;
    }
    return true;
}

// Spread of the block start times, recorded by block 0 at the first barrier of the pass
__device__ __forceinline__ void record_start_skew(const MegaRun r) {
    if (blockIdx.x == 0 && threadIdx.x == 0)
        atomicMax(&r.acc[1], __hip_atomic_load(&r.start[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT) -
                                 ~__hip_atomic_load(&r.start[1], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT));
}

// Profile stamp (optional): block 0 notes when step o ended
__device__ __forceinline__ void step_stamp(const MegaRun r, const StepFlags f, int o) {
    if (f.stamp) r.prof[2 * o + 1] = wall_clock64();
}

// Synchronization behind step o and profile stamps. Returns false if the pass was aborted.
__device__ __forceinline__ bool step_end(const MegaOp* ops, int o, const Ctx& c, const MegaRun r, const StepFlags f,
                                         bool& first) {
    const MegaOp& op = ops[o];
    if (f.arr) {
        __syncthreads();
        if (threadIdx.x == 0) f.arr[(int64_t)o * gridDim.x + blockIdx.x] = wall_clock64();
    }
    // The next step needs a barrier or a dependency wait unless this is the last one, or the step says it does not
    // depend on this one (host.wait -> shared expert).
    const bool sync = o + 1 < r.o1 && !(op.nobar == 1 || f.nobar_all);
    if (sync && op.nobar == 2) {
        if (!step_dependency(ops + o + 1, o, c.deadline, r)) return false;
    } else if (sync && !step_barrier(op, o, c.deadline, r)) {
        return false;
    }
    step_stamp(r, f, o);
    if (sync && op.nobar != 2) {
        if (first) record_start_skew(r);
        first = false;
    }
    return true;
}

template <int NB, KernelProfile Prof>
__global__ void __launch_bounds__(THREADS) __attribute__((amdgpu_waves_per_eu(CHAIN_WAVES)))
k_mega(const MegaOp* __restrict__ ops, const DevTask* __restrict__ tasks, MegaParams* P, unsigned* bar,
       unsigned long long* prof, const MegaScalars sc) {
    // Code generation of this kernel reacts to small changes of the source shape (a decode pass runs ~1,000 steps
    // through it): measure decode on the server path after every edit (docs/contribute/verification.md).
    // The kernel spills (SGPR spills mostly: the compiler hoists ~47 address sums out of the step loop), but removing
    // spills made decode slower: flags read from P per step, that plus a noinline step_end, or an empty asm per step
    // cost 0.5 to 3.4 % on the server path (research/measurements/2026-10-28-mega-spills.md). Above 192 VGPRs the two
    // blocks of 16 waves per WGP are no longer both resident and the barriers hang, so the wave limit stays at 8.
    const int o0 = sc.o0, o1 = sc.o1, wall_khz = sc.wall_khz;
    const uint64_t timeout_ticks = sc.timeout_ticks;
    unsigned long long* const start = reinterpret_cast<unsigned long long*>(bar + 4);
    unsigned long long* const acc = reinterpret_cast<unsigned long long*>(bar + 8);
    // The IQ2_XXS grid goes into LDS before the barrier inside TABS_DECL; absent for a profile without the format
    if constexpr (profile_quants<Prof>() == ExtraQuants::kNvfp4Iq2xxs) load_iq2xxs_lds();
    TABS_DECL;
    __shared__ __attribute__((aligned(16))) float sm[SM_FLOATS];
    if (threadIdx.x == 0) {
        const unsigned long long t = wall_clock64();
        atomicMax(&start[0], t);
        atomicMax(&start[1], ~t);
    }
    Ctx c;
    init_ctx(c, P, sm, wall_khz);
    const bool multi = P->nseq > 0;
    // Optional profile: block 0 stamps the start and the end of every step
    const bool stamp = prof && blockIdx.x == 0 && threadIdx.x == 0;
    if (stamp) prof[2 * o0] = wall_clock64();
    bool first = true;
    StepEnv env{tasks, tabs, sm, P, bar, wall_khz, 0};
    const MegaRun r{o0, o1, P, bar, start, acc, timeout_ticks, prof, wall_khz, sc};
    const StepFlags flags{P->arr, P->nobar_all != 0, stamp};
    // Progress per block (bar[16 + block]): after an abort the host sees which block stood where
    unsigned* progress = bar + 16;
    for (int o = o0; o < o1; o++) {
        const MegaOp& op = ops[o];
        if (threadIdx.x == 0)
            __hip_atomic_store(&progress[blockIdx.x], (unsigned)o, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        step_ctx(c, op, P, multi);
        env.o = o;
        if (c.nb > 0 || op.cond == 1) run_op<NB, Prof>(op, c, env);
        if (!step_end(ops, o, c, r, flags, first)) return;
    }
}

}  // namespace
}  // namespace tona::gpu
