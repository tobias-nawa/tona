// Time limits of the copies between VRAM and the host RAM of a stashed sequence, and the exceptions that report
// them. Pure arithmetic, so it is tested without a GPU (tests/test_stall.cpp).
#pragma once

#include <cstddef>
#include <stdexcept>

namespace tona::backend {

// A copy of a stash (or its restore) that did not end in time. The stash is gone (D2H) or the slot is invalid (H2D);
// the caller carries on without it. Thrown only after the stalled copy has ended, so no DMA is in flight any more.
struct CopyStalled : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Exit code when the copy does not end even after the host released its memory: the GPU of this process is unusable,
// and the next launch would hang as well. A non-zero code restarts the service (Restart=on-failure,
// scripts/tona-service.sh). 75 is EX_TEMPFAIL.
constexpr int EXIT_GPU_STALLED = 75;

// Time the host gets to end a stalled copy after it released memory. Measured: the copy ended 6.4 s and 11 s after the
// release (research/measurements/2026-11-01-stash-hang.md); without release it stood for more than 13 minutes.
constexpr double STALL_GRACE_S = 60.0;

// Limit for one chunk of a copy (and for the wait on idle streams): a base for latency and driver work plus the time
// at a floor rate. The floor is far below the PCIe rate of the stash copy (2026-11-01-stash-hang.md), so a slow but
// running copy never trips it, and a stall is seen after about 2 s instead of after the whole copy.
constexpr double CHUNK_BASE_S = 2.0, CHUNK_FLOOR_BYTES_PER_S = 250e6;

// Wait for the streams to be idle before a stash or unstash. The previous request has synchronized already, so this
// takes microseconds; 30 s only separates "still finishing" from "stalled".
constexpr double STREAM_IDLE_S = 30.0;

inline double chunk_deadline_s(size_t bytes) { return CHUNK_BASE_S + (double)bytes / CHUNK_FLOOR_BYTES_PER_S; }

}  // namespace tona::backend
