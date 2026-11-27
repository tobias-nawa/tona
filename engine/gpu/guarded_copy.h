// Copies between VRAM and host RAM that never wait without a time limit. A blocking hipMemcpy stood for 48 min
// when the host ran short of memory (research/measurements/2026-11-01-stash-hang.md).
//
// How: one non-blocking stream and a ring of pinned bounce buffers. The DMA touches only the bounce buffer, the
// pageable side (the vectors of a stash) only the CPU, so a vector can be released at any time without the GPU
// writing into freed memory. The bounce buffer, the stream and the events live as long as the process and are never
// freed. The engine thread polls the events (hipEventQuery, never a blocking HIP call) with a limit per chunk.
//
// On a stall: log, the caller's `relief` releases host memory (measured: the copy then ends within seconds), and the
// helper waits for the copy to end (STALL_GRACE_S). Only then does it return false, so nothing on the GPU is still
// running. If the copy does not end, the GPU of this process is unusable and the process exits with
// EXIT_GPU_STALLED. Between the stall and the end of the copy the engine thread does nothing else on the GPU.
#pragma once

#include <hip/hip_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <initializer_list>
#include <vector>

namespace tona::gpu {

class GuardedCopy {
public:
    using Relief = std::function<void()>;
    enum class Dir { ToHost, ToDevice };

    // Created at load time: allocating pinned memory under memory pressure is itself a place to hang
    static GuardedCopy& get();

    // One piece of a copy: `host` (pageable is fine) and `dev` are the two ends, whichever direction the copy goes
    struct Region {
        void* host;
        void* dev;
        size_t bytes;
    };

    // All regions of a stash go through the bounce buffers in one pipeline, so the small ones (indexer pools of a few
    // MB) do not each pay for the latency of a copy of their own. false: stalled and ended after the relief (the
    // destination of ToHost is released, that of ToDevice is partly written); the caller throws CopyStalled.
    bool to_host(const std::vector<Region>& regions, const Relief& relief);
    bool to_device(const std::vector<Region>& regions, const Relief& relief);
    // The streams are idle (replaces hipDeviceSynchronize / hipStreamSynchronize); `limit_s` as for a chunk
    bool wait_idle(std::initializer_list<hipStream_t> streams, double limit_s, const Relief& relief);

    // TONA_COPY_PROF=1: calls, bytes and the time in enqueue, wait for events, CPU copy and idle wait since the last
    // report, one line on stderr; off, it costs one branch per call
    void prof_report(const char* op);
    bool prof_on() const { return prof_on_; }
    void prof_add_setup(double s) { prof_.setup += s; }  // time of the caller outside the copies (host allocation)

    // 4 buffers of 16 MB: the DMA of a chunk (0.3 ms at the PCIe rate) always ends before the CPU is at it. 8 and 32 MB
    // chunks, 8 buffers measured the same (research/measurements/2026-11-01-stash-hang.md).
    static constexpr int NBUF = 4;
    static constexpr size_t CHUNK_BYTES = size_t(16) << 20;

private:
    GuardedCopy();
    bool ready_within(const std::function<bool()>& ready, double limit_s) const;
    bool event_within(hipEvent_t ev, double limit_s) const;
    bool stalled(const char* what, size_t done, size_t total, const std::function<bool()>& ready,
                 const Relief& relief) const;
    void issue(Dir dir, void* dev, size_t len, int buf);
    void arm_test_stall(Dir dir);

    struct Prof {
        size_t calls = 0, chunks = 0, bytes = 0;
        double issue = 0, wait = 0, copy = 0, idle = 0, setup = 0;
    } prof_;
    bool prof_on_ = false;

    hipStream_t st_ = nullptr;
    hipEvent_t ev_[NBUF] = {}, last_ = nullptr;
    uint8_t* bounce_[NBUF] = {};
    // Test hook TONA_TEST_STASH_STALL_MS: a wait on a host word in front of the first copy of one direction
    uint32_t *flag_host_ = nullptr, *flag_dev_ = nullptr;
    double grace_s_ = 60;  // TONA_TEST_COPY_GRACE_S shortens it for the exit test
    long stall_ms_ = -2;   // -2: off, -1: the word is never set
    Dir stall_dir_ = Dir::ToHost;
};

}  // namespace tona::gpu
