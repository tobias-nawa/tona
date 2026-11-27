// The arithmetic of the time limits of a wait in the megakernel (matvec_barrier.h, mega_op_wait.h), free of the GPU so
// a unit test can run it on the CPU (tests/test_wait_clock.cpp).
#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__HIPCC__)
#define TONA_WAIT_FN __host__ __device__ inline __attribute__((always_inline))
#else
#define TONA_WAIT_FN inline
#endif

namespace tona::gpu {

// Limits of one wait. A wave that is evicted (KFD takes all queues of the process away for ms to s while the kernel
// stays resident, research/measurements/2026-10-31-barrier-aborts.md) stops polling and finds its limit passed when it
// runs again, before the blocks that were evicted with it have arrived. So a gap between two polls longer than
// freeze_ticks (the wave did not run) is not counted as waiting: t0 and the deadline move by it. A wave that polls
// without gaps for a block that never arrives still aborts after timeout_ticks. cap_ticks bounds the wall time of one
// wait whatever the shifts add up to.
struct WaitLimits {
    uint64_t timeout_ticks, deadline;  // deadline: wall_clock64 from which every wait aborts (~0 for none)
    uint64_t freeze_ticks, cap_ticks;  // ~0 for both: no shifting
    uint64_t test_ticks;               // TONA_TEST_FREEZE_MS: simulated freeze, see test_freeze (0: off)
    unsigned test_gen, us_ticks;       // barrier it happens at; ticks per us
    unsigned test_all;                 // TONA_TEST_FREEZE_ALL: all blocks stand still after the barrier, no late block
    unsigned long long* note;          // 8 words for note_abort, null: none
    unsigned long long* hist;          // optional: histogram of the gaps between polls, null: none
};

// A freeze while no block waits (all in their step, nobody sees a gap) eats into the deadline of the pass too, and the
// waits behind it are short. So the deadline aborts a wait only after it has lasted kDeadlineWaitMs; a wait that
// is short is let through up to kDeadlineGraceMs behind the deadline: with the default budget of 1,500 ms a pass ends
// by 1,800 ms, under the 2 s of the driver (amdgpu lockup_timeout), and the longest eviction measured (1.35 s) plus a
// pass of 0.1 s still fits (research/measurements/2026-10-31-barrier-aborts.md).
constexpr uint64_t kDeadlineWaitMs = 100, kDeadlineGraceMs = 300;

// Eviction credit. The deadline of the pass counts wall time, and a queue that KFD has evicted does not run: the
// kernel is not on the GPU then, so the desktop is not held up either. The 2 s of the driver are the DRM job timeout of
// the amdgpu rings (amdgpu_device.c, amdgpu_device_get_job_timeout_settings: "By default timeout for all queues is 2
// sec"; amdgpu_job.c, amdgpu_job_timedout), not a KFD timer: the eviction itself waits for the preemption at most
// queue_preemption_timeout_ms (kfd_device_queue_manager.c, unmap_queues_cpsch) and has no timer for its duration.
// A wave that finds the deadline passed without having waited long therefore asks the host how long the process's
// queues were evicted since the pass started (evicted_ms of KFD, which the restore adds before the waves run again) and
// moves the deadline by that. The host answers at most kEvictCapMs, the answer is clamped to the wall time since the
// pass started, and without an answer within kEvictAskMs the wave aborts as before. 30 s: ten times the longest
// eviction measured (4.4 s, research/measurements/2026-11-02-evict-deadline-ab-speed.md); a counter that jumps cannot
// buy more than the wall time of the pass.
constexpr uint64_t kEvictAskMs = 50, kEvictCapMs = 30000;

// The credit in ticks: the host's answer, at most the cap and at most the time since the pass started
TONA_WAIT_FN uint64_t evict_credit_ticks(uint64_t delta_ms, uint64_t elapsed_ticks, uint64_t ticks_per_ms) {
    uint64_t ms = delta_ms < kEvictCapMs ? delta_ms : kEvictCapMs;
    const uint64_t elapsed_ms = elapsed_ticks / ticks_per_ms;
    return (ms < elapsed_ms ? ms : elapsed_ms) * ticks_per_ms;
}

// State of one wait: start, shifted start, previous poll, shifted deadline
struct WaitClock {
    uint64_t cap_end, t0, prev, dl;  // cap_end: wall time of the cap (~0: none)
    TONA_WAIT_FN WaitClock(const WaitLimits& w, uint64_t now)
        : cap_end(w.cap_ticks == ~0ull ? ~0ull : now + w.cap_ticks), t0(now), prev(now), dl(w.deadline) {}
    // Called at every poll: moves the limits by the time the wave did not run; returns the gap since the last poll
    TONA_WAIT_FN uint64_t poll(uint64_t now, const WaitLimits& w) {
        const uint64_t gap = now - prev;
        prev = now;
        if (gap > w.freeze_ticks) {
            t0 += gap;
            if (dl != ~0ull) dl += gap;
        }
        return gap;
    }
    // 0: no limit passed, else the bits of the kind (1 limit, 2 deadline, 16 cap)
    TONA_WAIT_FN unsigned expired(uint64_t now, const WaitLimits& w) const {
        const uint64_t ms = 1000ull * w.us_ticks;
        const bool deadline = now > dl && (now - t0 > kDeadlineWaitMs * ms || now - dl > kDeadlineGraceMs * ms);
        return (now - t0 > w.timeout_ticks ? 1u : 0u) | (deadline ? 2u : 0u) | (now > cap_end ? 16u : 0u);
    }
};

// Atomics on the words shared with the host and between blocks (system scope: the mailbox is host memory)
#if defined(__HIP_DEVICE_COMPILE__)
#define TONA_LD(p) __hip_atomic_load((p), __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)
#define TONA_ST(p, v) __hip_atomic_store((p), (v), __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM)
#define TONA_CAS(p, expected, v)                                                                    \
    __hip_atomic_compare_exchange_strong((p), &(expected), (v), __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE, \
                                         __HIP_MEMORY_SCOPE_SYSTEM)
#else
#define TONA_LD(p) __atomic_load_n((p), __ATOMIC_ACQUIRE)
#define TONA_ST(p, v) __atomic_store_n((p), (v), __ATOMIC_RELEASE)
#define TONA_CAS(p, expected, v) \
    __atomic_compare_exchange_n((p), &(expected), (v), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)
#endif

// Where the protocol keeps its words. ext (device memory, reset by every launch): [0] state (0 idle, 1 asking, 2
// refused), [1] credit in ticks. start: wall_clock64 at which the pass started (latest block start). mb (host
// mailbox): [0] request, [1] answer, [2] evicted ms.
struct EvictLink {
    unsigned long long* ext;
    const unsigned long long* start;
    unsigned* mb;
};

// The wait found only the deadline of the pass passed. True: poll again (dl moved, or another block is asking);
// false: abort. One block asks (state 0 -> 1), the others wait for its answer. Refusal (no answer in kEvictAskMs, or
// no more credit than before) is final for the launch. now(): wall_clock64; sleep(): a pause.
template <class Now, class Sleep>
TONA_WAIT_FN bool evict_extend(const EvictLink& l, uint64_t deadline, unsigned us_ticks, uint64_t& dl, Now now,
                               Sleep sleep) {
    if (!l.mb || !l.ext || !l.start) return false;
    const uint64_t per_ms = 1000ull * us_ticks;
    for (;;) {
        const uint64_t credit = TONA_LD(&l.ext[1]);
        if (deadline + credit > dl) {
            dl = deadline + credit;
            return true;
        }
        const uint64_t state = TONA_LD(&l.ext[0]);
        if (state == 2) return false;
        if (state == 1) return true;
        unsigned long long idle = 0;
        if (TONA_CAS(&l.ext[0], idle, 1ull)) break;
    }
    const unsigned ack = TONA_LD(&l.mb[1]);
    TONA_ST(&l.mb[0], ack + 1);
    const uint64_t t0 = now();
    while (TONA_LD(&l.mb[1]) != ack + 1) {
        if (now() - t0 > kEvictAskMs * per_ms) {
            TONA_ST(&l.ext[0], 2ull);
            return false;
        }
        sleep();
    }
    const uint64_t t = now(), s = TONA_LD(l.start);
    const uint64_t credit = evict_credit_ticks(TONA_LD(&l.mb[2]), t > s ? t - s : 0, per_ms);
    if (credit <= TONA_LD(&l.ext[1])) {
        TONA_ST(&l.ext[0], 2ull);
        return false;
    }
    TONA_ST(&l.ext[1], credit);
    TONA_ST(&l.ext[0], 0ull);
    if (deadline + credit > dl) dl = deadline + credit;
    return true;
}

// Byte offsets in the barrier buffer (mega_ops.h): the credit words behind the abort note, the pointer to the mailbox
// words behind the gap histogram (never reset by a launch, null: no credit)
constexpr size_t kEvictExtOffset = 64 + 8 * 256 + 64, kEvictLinkOffset = kEvictExtOffset + 64 + 64;

}  // namespace tona::gpu
