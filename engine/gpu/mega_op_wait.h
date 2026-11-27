// Waiting in the megakernel: for the CPU (host.wait, host.y), pause and abort of a pass, dependencies between blocks.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Time limit of a wait, capped at the deadline of the pass: several waits just below their own limit must not add
// up beyond the driver's limit (amdgpu: 2 s, then a GPU reset that takes the desktop with it).
__device__ __forceinline__ uint64_t wait_limit(uint64_t lim, uint64_t t0, uint64_t deadline) {
    return deadline <= t0 ? 0 : min(lim, deadline - t0);
}

// Ends the pass and counts it as an abort (acc[0] = bar[8..9], as grid_barrier does). Without the count the GPU
// would stop unnoticed by check_aborts and forward would return the logits of the previous pass.
__device__ void abort_pass(unsigned* bar, uint64_t t0, uint64_t now, uint64_t deadline) {
    if (!__hip_atomic_exchange(&bar[2], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
        atomicAdd(reinterpret_cast<unsigned long long*>(bar + 8), 1ull);
        note_abort(reinterpret_cast<unsigned long long*>(bar + BAR_DIAG / 4), 8u | (now > deadline ? 2u : 1u),
                   {t0, now, deadline, 0, 0, 0});
    }
}

// Pause instead of abort (research/23-waiting-in-kernel.md): block 0 notes the step o, reports it to the CPU and
// stops all blocks at their next barrier (bar[2] as for an abort, but without the abort count). The CPU restarts the
// kernel at o when it is done. All blocks have finished the steps before o; o and the barrier-free step behind it
// (shared expert) only write and may run twice.
__device__ void yield_pass(unsigned* bar, MegaMailbox* mb, int o, uint32_t tag) {
    __hip_atomic_store(&bar[BAR_YIELD], (unsigned)o + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    __hip_atomic_store(&mb->yield_op, (uint32_t)o, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_store(&mb->yield_seq, tag, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    __hip_atomic_exchange(&bar[2], 1u, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
}

// What a wait for the CPU needs besides the context: the counter to watch and the pass state to report a yield.
struct HostWait {
    const uint32_t* cnt;
    uint32_t want;
    unsigned* bar;
    MegaMailbox* mb;
    int o, wall_khz;
};
// Block 0 (thread 0) waits for the CPU until cnt reaches want: true when the CPU is done. Otherwise a yield after
// yield_ms or at the deadline of the pass (safe, the restart gets a new deadline); without yield_ms an abort after
// HOST_WAIT_MS or at the deadline.
__device__ bool wait_cpu(const HostWait hw, const Ctx& c) {
    const uint32_t *cnt = hw.cnt, want = hw.want;
    unsigned* bar = hw.bar;
    MegaMailbox* mb = hw.mb;
    const int o = hw.o, wall_khz = hw.wall_khz;
    const uint64_t t0 = wall_clock64(), lim = wait_limit((uint64_t)wall_khz * HOST_WAIT_MS, t0, c.deadline);
    const uint64_t ylim = c.P->yield_ms > 0 ? (uint64_t)wall_khz * c.P->yield_ms : ~0ull;
    // Yield 50 ms before the deadline: otherwise the barriers of the other blocks (which wait for block 0) expire
    // there at the same time and count an abort although the yield saves the pass. Many short waits in prefill (48
    // layers) reach the deadline without a single one exceeding yield_ms.
    const uint64_t margin = (uint64_t)wall_khz * 50, ydl = c.deadline > margin ? c.deadline - margin : 0;
    while (__hip_atomic_load(cnt, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) != want) {
        const uint64_t now = wall_clock64(), dt = now - t0;
        if (c.P->yield_ms > 0 && (dt > ylim || now > ydl)) {
            yield_pass(bar, mb, o, (uint32_t)(c.P->seq + 62));
            return false;
        }
        if (dt > lim) {
            abort_pass(bar, t0, now, c.deadline);
            return false;
        }
        __builtin_amdgcn_s_sleep(1);
    }
    return true;
}

// nb rows of MaxE logits (E of them real) to the mailbox, clamped (NaN = -FLT_MAX, +-inf =
// +-FLT_MAX: the ranking on the CPU needs a strict weak order). Returns this thread's count of NaN among the real
// logits.
template <int MaxE>
__device__ __forceinline__ float copy_logits_clamped(float* dst, const float* src, int nb, int E) {
    float nan_n = 0;
    for (int e = threadIdx.x; e < nb * MaxE; e += blockDim.x) {
        const float v = src[e];
        nan_n += v != v && e % MaxE < E;
        dst[e] = logit_clamp(v);
    }
    return nan_n;
}

template <int MaxE>
__device__ __noinline__ void op_host_wait(const MegaOp& op, const Ctx& c, unsigned* bar, int wall_khz, int o) {
    if (blockIdx.x != 0) return;
    MegaMailbox* mb = (MegaMailbox*)op.p[0];
    const float* router = (const float*)op.p[1];
    const uint32_t want = (uint32_t)(c.P->seq + op.i[0] + 1);
    __shared__ int ok;
    // One system-wide fence in this block only instead of in all (it writes the L2 back, which is expensive)
    // The logits go to the CPU clamped (NaN as the smallest value): its ranking needs a strict weak order. NaN among
    // the E real logits per token is counted per wave in sm, summed by thread 0 below, and reported by the CPU.
    const int E = op.i[2];
    const float nan_r = copy_logits_clamped<MaxE>(mailbox_router<MaxE>(mb), router, c.nb, E);
    float nan_p = 0;
    if (op.p[2]) {  // CPU mode: x of the experts (int8 + metadata) for the CPU
        const int d = op.i[1];
        const uint4* q = (const uint4*)op.p[2];
        uint4* qd = (uint4*)op.p[4];
        for (int k = threadIdx.x; k < c.nb * d / 16; k += blockDim.x) qd[k] = q[k];
        const uint2* m = (const uint2*)op.p[3];
        uint2* md = (uint2*)op.p[5];
        for (int k = threadIdx.x; k < c.nb * d / 32; k += blockDim.x) md[k] = m[k];
        if (op.p[6]) {
            const float* pr = (const float*)op.p[6];
            nan_p = copy_logits_clamped<MaxE>(mailbox_pred<MaxE>(mb), pr, c.nb, E);
        }
    }
    const float wr = wave_sum(nan_r), wp = wave_sum(nan_p);
    if (threadIdx.x % WAVE == 0)
        ((int*)c.sm)[threadIdx.x / WAVE] = (int)wr, ((int*)c.sm)[MWAVES + threadIdx.x / WAVE] = (int)wp;
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {
        int sr = 0, sp = 0;
        for (int w = 0; w < MWAVES; w++) sr += ((int*)c.sm)[w], sp += ((int*)c.sm)[MWAVES + w];
        mb->nan_router = sr, mb->nan_pred = sp;
        __hip_atomic_store(&mb->gpu_seq, want, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
        ok = wait_cpu({&mb->cpu_seq, want, bar, mb, o, wall_khz}, c);
    }
    __syncthreads();
    if (!ok) return;
    const int t = threadIdx.x;  // the mailbox is uncached; read after thread 0's acquire
    if (t < UMAX) c.P->slot[t] = mb->slot[t], c.P->umask[t] = mb->umask[t], c.P->miss_slot[t] = mb->miss_slot[t];
    for (int i = t; i < BMAX * UMAX; i += blockDim.x) c.P->ew[i] = mb->ew[i];
    if (t == 0) c.P->n_u = mb->n_u, c.P->n_miss = mb->n_miss, c.P->n_cpu = mb->n_cpu, c.P->fill_half = mb->fill_half;
}

// maybe_unused: a unit that includes this header without a kernel profile does not reach it through dispatch_host
[[maybe_unused]] __device__ __noinline__ void op_host_y(const MegaOp& op, const Ctx& c, unsigned* bar, int wall_khz,
                                                        int o) {
    if (blockIdx.x != 0 || threadIdx.x != 0 || c.P->n_cpu == 0) return;
    MegaMailbox* mb = (MegaMailbox*)op.p[0];
    wait_cpu({&mb->cpu_done, (uint32_t)(c.P->seq + op.i[0] + 1), bar, mb, o, wall_khz}, c);
}

[[maybe_unused]] __device__ __noinline__ void op_expert_copy(const MegaOp& op, const Ctx& c) {
    const int n = c.P->n_miss;
    if (n == 0) return;
    const int64_t sb = op.i[0], per = sb / 16;
    const uint4* src = (const uint4*)((const uint8_t*)op.p[0] + (int64_t)c.P->fill_half * op.i[1] * sb);
    uint8_t* slots = (uint8_t*)op.p[1];
    for (int64_t i = c.gt; i < n * per; i += c.GT) {
        const int k = (int)(i / per);
        const int64_t off = i - k * per;
        reinterpret_cast<uint4*>(slots + (int64_t)c.P->miss_slot[k] * sb)[off] = src[k * per + off];
    }
}

// Dependency instead of a barrier after step o: report done (done[block] = o + 1), then wait only for the blocks in
// mask. Time limit and abort as in grid_barrier (bar[2] also ends the blocks at ordinary barriers).
__device__ __forceinline__ bool dep_wait(unsigned* bar, unsigned long long* acc, uint64_t mask, unsigned o1,
                                         const WaitLimits& w) {
    unsigned* done = bar + BAR_DONE / 4;
    __threadfence();
    __syncthreads();
    __shared__ int aborted;
    if (threadIdx.x == 0) {
        __hip_atomic_store(&done[blockIdx.x], o1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
        WaitClock c(w, wall_clock64());
        uint64_t polls = 0;
        for (uint64_t m = mask & ~(1ull << blockIdx.x); m;) {
            const int b = __builtin_ctzll(m);
            if (__hip_atomic_load(&done[b], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) >= o1) {
                m &= m - 1;
                continue;
            }
            if (__hip_atomic_load(&bar[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) break;
            const uint64_t now = wall_clock64();
            const uint64_t gap = c.poll(now, w);
            count_poll(w, gap, polls);
            if (const unsigned kind = c.expired(now, w)) {
                if (credit_deadline(bar, w, c, kind)) continue;
                if (!__hip_atomic_exchange(&bar[2], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
                    atomicAdd(&acc[0], 1ull);
                    note_abort(w.note, 4u | kind, {c.t0, now, c.dl, w.timeout_ticks, m, (unsigned long long)b});
                }
                break;
            }
            __builtin_amdgcn_s_sleep(1);
        }
        if (kWaitDiag && w.hist) atomicAdd(&w.hist[0], polls);
        aborted = __hip_atomic_load(&bar[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return !aborted;
}

// Runs the step if it waits for the CPU or copies experts; false for the other groups.
template <int MaxE>
__device__ __forceinline__ bool dispatch_host(const MegaOp& op, const Ctx& c, const StepEnv env) {
    switch (op.kind) {
        case OP_HOST_WAIT:
            op_host_wait<MaxE>(op, c, env.bar, env.wall_khz, env.o);
            return true;
        case OP_HOST_Y:
            op_host_y(op, c, env.bar, env.wall_khz, env.o);
            return true;
        case OP_EXPERT_COPY:
            op_expert_copy(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
