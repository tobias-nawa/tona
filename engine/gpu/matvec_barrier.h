// Grid barrier of the persistent chain kernels. Part of matvec_dev.h; do not include on its own.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "gpu/wait_clock.h"

namespace tona::gpu {
namespace {

// Own grid barrier instead of grid.sync: arrival counter and generation in device memory, with a time limit.
// grid.sync hung for minutes in 2 of ~12 runs (the GPU is shared with the desktop); this barrier makes the kernel
// abort after CHAIN_TIMEOUT_MS and report it instead of blocking the GPU.
// bar: [0] arrivals, [1] generation, [2] abort; start: [0] latest, [1] ~earliest block start (reset per run).
// acc (over runs, until reset_diag): [0] aborts, [1] largest start skew, [2] longest wait; times in ticks of
// wall_clock64.
#ifndef CHAIN_TIMEOUT_MS
#define CHAIN_TIMEOUT_MS 100
#endif

__device__ __forceinline__ WaitLimits plain_limits(uint64_t timeout_ticks) {
    return {timeout_ticks, ~0ull, ~0ull, ~0ull, 0, 0, 0, 0, nullptr, nullptr};
}

// The wait found only the deadline of the pass passed (kind 2): ask for the eviction credit (evict_extend). Returns the
// new deadline, 0 to abort. The words come from the barrier buffer, so that no field of WaitLimits and no pointer of
// the caller is needed (every field costs registers in k_mega); not inlined, because it is the cold path.
__device__ __noinline__ uint64_t extend_deadline(unsigned* bar, uint64_t deadline, unsigned us_ticks, uint64_t dl) {
    const uint8_t* b = reinterpret_cast<const uint8_t*>(bar);
    const EvictLink link{reinterpret_cast<unsigned long long*>(const_cast<uint8_t*>(b) + kEvictExtOffset),
                         reinterpret_cast<const unsigned long long*>(b + 16),
                         reinterpret_cast<unsigned* const*>(b + kEvictLinkOffset)[0]};
    const bool again = evict_extend(
        link, deadline, us_ticks, dl, [] { return (uint64_t)wall_clock64(); }, [] { __builtin_amdgcn_s_sleep(1); });
    return again ? dl : 0;
}

// Both waits: kind 2 may be rescued by the eviction credit; true = poll again with the moved deadline in c
__device__ __forceinline__ bool credit_deadline(unsigned* bar, const WaitLimits& w, WaitClock& c, unsigned kind) {
    if (kind != 2u) return false;
    const uint64_t dl = extend_deadline(bar, w.deadline, w.us_ticks, c.dl);
    if (!dl) return false;
    c.dl = dl;
    __builtin_amdgcn_s_sleep(1);
    return true;
}

// What the block that counts an abort saw (note, 7 words, written once per abort; the CPU has no other way to tell a
// real stall from a wrong alarm): [0] kind | block << 8, with kind bit 0 = limit of the wait passed, bit 1 = deadline
// of the pass passed, bit 2 = dependency wait, bit 3 = wait for the CPU, bit 4 = cap; [1] start of the wait, [2] now,
// [3] deadline, [4] limit (all in wall_clock64 ticks); [5] barrier: arrivals counted, dependency: blocks still
// missing as a mask; [6] barrier: generation, dependency: first missing block. [7]: test hook (test_freeze).
__device__ __noinline__ void note_abort(unsigned long long* note, unsigned kind, const unsigned long long (&v)[6]) {
    if (!note) return;
    __hip_atomic_store(&note[0], (unsigned long long)kind | ((unsigned long long)blockIdx.x << 8), __ATOMIC_RELAXED,
                       __HIP_MEMORY_SCOPE_AGENT);
    for (int i = 0; i < 6; i++) __hip_atomic_store(&note[1 + i], v[i], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// Bucket of a gap between two polls: 0 total polls, 1: 16-64 us, 2: -256 us, 3: -1 ms, 4: -2 ms, 5: -10 ms, 6: -100 ms,
// 7: longer. Only the gaps from 16 us on are counted one by one.
__device__ __noinline__ void note_gap(unsigned long long* hist, uint64_t gap, unsigned us) {
    const uint64_t u = gap / us;
    const int b = u < 64 ? 1 : u < 256 ? 2 : u < 1000 ? 3 : u < 2000 ? 4 : u < 10000 ? 5 : u < 100000 ? 6 : 7;
    atomicAdd(&hist[b], 1ull);
}

// Diagnostics of the waits (gap histogram, simulated evictions) exist only in a build with -DTONA_WAIT_DIAG: in the
// production kernel they cost registers (k_mega spills more VGPRs with them) and the env switches stay without effect
#ifdef TONA_WAIT_DIAG
constexpr bool kWaitDiag = true;
#else
constexpr bool kWaitDiag = false;
#endif

// Test hook: spins for test_ticks without polling, as an evicted wave does
__device__ __noinline__ void test_freeze(uint64_t ticks) {
    const uint64_t t = wall_clock64();
    while (wall_clock64() - t < ticks) __builtin_amdgcn_s_sleep(2);
}

// Test hook, all blocks: stand still for test_ticks right after the barrier before the test barrier
// (TONA_TEST_FREEZE_ALL), so the waits behind it are short
__device__ __forceinline__ void test_stall_all(const WaitLimits& w, unsigned gen) {
    if (kWaitDiag && w.test_all != 0 && w.test_ticks != 0 && gen + 1 == w.test_gen) test_freeze(w.test_ticks);
}

// Thread 0 of a block waits until the last block arrives (generation gen ends) or a limit passes.
// Gap statistics of one poll (TONA_MEGA_GAP_HIST)
__device__ __forceinline__ void count_poll(const WaitLimits& w, uint64_t gap, uint64_t& polls) {
    if (!kWaitDiag || !w.hist) return;
    polls++;
    if (gap >= 16ull * w.us_ticks) note_gap(w.hist, gap, w.us_ticks);
}

// Test hook: one block leaves the barrier before the test barrier and stays away 5 ms longer than the waiters stand
// still, as if it had been evicted in its step: the waiters see their limit passed before it arrives
__device__ __forceinline__ void test_stall_late(const WaitLimits& w, unsigned gen) {
    if (kWaitDiag && w.test_ticks != 0 && w.test_all == 0 && gen + 1 == w.test_gen && w.note &&
        atomicCAS(&w.note[7], 0ull, 1ull) == 0)
        test_freeze(w.test_ticks + 5000ull * w.us_ticks);
}

__device__ __forceinline__ void barrier_wait(unsigned* bar, unsigned long long* acc, unsigned gen,
                                             const WaitLimits& w) {
    WaitClock c(w, wall_clock64());
    uint64_t polls = 0;
    bool frozen = kWaitDiag && w.test_ticks != 0 && w.test_all == 0 && gen == w.test_gen;
    while (__hip_atomic_load(&bar[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen) {
        if (__hip_atomic_load(&bar[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) break;
        if (frozen) test_freeze(w.test_ticks), frozen = false;
        const uint64_t now = wall_clock64();
        const uint64_t gap = c.poll(now, w);
        count_poll(w, gap, polls);
        if (const unsigned kind = c.expired(now, w)) {
            if (credit_deadline(bar, w, c, kind)) continue;
            // let through exactly at the limit: no abort
            if (__hip_atomic_load(&bar[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) != gen) break;
            if (!__hip_atomic_exchange(&bar[2], 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT)) {
                atomicAdd(&acc[0], 1ull);
                note_abort(w.note, kind,
                           {c.t0, now, c.dl, w.timeout_ticks,
                            __hip_atomic_load(&bar[0], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT), gen});
            }
            break;
        }
        __builtin_amdgcn_s_sleep(1);
    }
    if (kWaitDiag && w.hist) atomicAdd(&w.hist[0], polls);
    atomicMax(&acc[2], (unsigned long long)(wall_clock64() - c.t0));
    test_stall_late(w, gen);
    test_stall_all(w, gen);
}

__device__ __forceinline__ bool grid_barrier(unsigned* bar, unsigned long long* acc, unsigned nblocks,
                                             const WaitLimits& w) {
    // every wave releases its writes (not only those of thread 0)
    __threadfence();
    __syncthreads();
    __shared__ int aborted;
    if (threadIdx.x == 0) {
        const unsigned gen = __hip_atomic_load(&bar[1], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        __threadfence();
        if (__hip_atomic_fetch_add(&bar[0], 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT) == nblocks - 1) {
            __hip_atomic_store(&bar[0], 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&bar[1], gen + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
            test_stall_all(w, gen);
        } else {
            barrier_wait(bar, acc, gen, w);
        }
        aborted = __hip_atomic_load(&bar[2], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    __syncthreads();
    // every wave drops its cache lines: in WGP mode the waves of a block sit on two CUs with their own L0 each, and
    // the acquire of thread 0 reaches only one of them
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "agent");
    return !aborted;
}

}  // namespace
}  // namespace tona::gpu
