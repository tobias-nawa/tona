#include "gpu/guarded_copy.h"

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "backend/env.h"
#include "backend/stall.h"
#include "gpu/alloc_trace.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

using Clock = std::chrono::steady_clock;

double since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

const char* dir_name(GuardedCopy::Dir d) { return d == GuardedCopy::Dir::ToHost ? "device to host" : "host to device"; }

uint32_t* g_release_word = nullptr;  // test (b2): set before the exit, so no queue waits for it any more

// The GPU of this process does not answer: leave without teardown (it would wait for the same GPU). Exit code 75
// restarts the service.
[[noreturn]] void leave_stalled(const char* what, size_t done, size_t total, double waited_s) {
    fprintf(stderr,
            "FATAL: stalled (%s), %zu of %zu bytes, no end after %.0f s; the GPU of this process is unusable. Exiting "
            "with code %d so the service restarts.\n",
            what, done, total, waited_s, backend::EXIT_GPU_STALLED);
    fflush(stderr);
    if (g_release_word) __atomic_store_n(g_release_word, 1u, __ATOMIC_RELEASE);
    _exit(backend::EXIT_GPU_STALLED);
}

// Last resort for a HIP call in the copy path that does not return. The enqueue of a host-to-device copy blocked the
// host for as long as the stream stood (measured, 3.5 s in the test hook), where the polling cannot help. A thread
// outside the call ends the process when `due` has passed. The fields are atomics: arm() writes them while the thread
// may still read the previous values.
class Watchdog {
public:
    static Watchdog& get() {
        static Watchdog w;
        return w;
    }
    void arm(const char* what, size_t bytes, double limit_s) {
        what_.store(what, std::memory_order_relaxed);
        bytes_.store(bytes, std::memory_order_relaxed);
        limit_s_.store(limit_s, std::memory_order_relaxed);
        due_ns_.store(now_ns() + (int64_t)(limit_s * 1e9), std::memory_order_release);
    }
    void disarm() { due_ns_.store(0, std::memory_order_release); }

private:
    Watchdog() {
        std::thread([this] {
            for (;; std::this_thread::sleep_for(std::chrono::milliseconds(100))) {
                const int64_t due = due_ns_.load(std::memory_order_acquire);
                if (due != 0 && now_ns() > due)
                    leave_stalled(what_.load(std::memory_order_relaxed), 0, bytes_.load(std::memory_order_relaxed),
                                  limit_s_.load(std::memory_order_relaxed));
            }
        }).detach();
    }
    static int64_t now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now().time_since_epoch()).count();
    }
    std::atomic<int64_t> due_ns_{0};
    std::atomic<const char*> what_{""};
    std::atomic<size_t> bytes_{0};
    std::atomic<double> limit_s_{0};
};

// Copies on several threads: one thread reaches only part of the DRAM rate, and the copy of a chunk is the part of the
// stash that the DMA cannot overlap. The threads start at load time and are never joined (like the bounce buffers,
// they live as long as the process).
class CopyPool {
public:
    static CopyPool& get() {
        static CopyPool* p = new CopyPool;
        return *p;
    }
    void copy(void* dst, const void* src, size_t n) {
        if (workers_ == 0 || n < MIN_PARALLEL_BYTES) {
            std::memcpy(dst, src, n);
            return;
        }
        const size_t part = (n / (workers_ + 1)) & ~size_t(63);
        {
            const std::lock_guard<std::mutex> lk(m_);
            dst_ = static_cast<uint8_t*>(dst), src_ = static_cast<const uint8_t*>(src), part_ = part;
            pending_ = workers_, gen_++;
        }
        cv_.notify_all();
        std::memcpy(static_cast<uint8_t*>(dst) + workers_ * part, static_cast<const uint8_t*>(src) + workers_ * part,
                    n - workers_ * part);  // the caller takes the last part and the remainder
        std::unique_lock<std::mutex> lk(m_);
        done_.wait(lk, [this] { return pending_ == 0; });
    }

private:
    // 1, 2, 3, 4, 6 threads: stash 107, 76, 64, 59, 58 ms at 30K (research/measurements/2026-11-01-stash-hang.md); more
    // than 4 gains nothing. The copy is bound by page faults of the fresh stash and DRAM bandwidth, not by the DMA.
    static constexpr int THREADS = 4;
    static constexpr size_t MIN_PARALLEL_BYTES = size_t(2) << 20;
    CopyPool() {
        workers_ = THREADS - 1;
        for (int i = 0; i < workers_; i++) std::thread([this, i] { run(i); }).detach();
    }
    void run(int id) {
        uint64_t seen = 0;
        for (;;) {
            std::unique_lock<std::mutex> lk(m_);
            cv_.wait(lk, [&] { return gen_ != seen; });
            seen = gen_;
            uint8_t* d = dst_ + id * part_;
            const uint8_t* s = src_ + id * part_;
            const size_t part = part_;
            lk.unlock();
            std::memcpy(d, s, part);
            lk.lock();
            if (--pending_ == 0) done_.notify_one();
        }
    }
    std::mutex m_;
    std::condition_variable cv_, done_;
    uint8_t* dst_ = nullptr;
    const uint8_t* src_ = nullptr;
    size_t part_ = 0;
    uint64_t gen_ = 0;
    int pending_ = 0, workers_ = 0;
};

struct Armed {
    Armed(const char* what, size_t bytes, double limit_s) { Watchdog::get().arm(what, bytes, limit_s); }
    ~Armed() { Watchdog::get().disarm(); }
    Armed(const Armed&) = delete;
    Armed& operator=(const Armed&) = delete;
};

}  // namespace

GuardedCopy& GuardedCopy::get() {
    static GuardedCopy g;
    return g;
}

GuardedCopy::GuardedCopy() {
    HIPCHECK(hipStreamCreateWithFlags(&st_, hipStreamNonBlocking));
    for (int i = 0; i < NBUF; i++) {
        HIPCHECK(hipEventCreateWithFlags(&ev_[i], hipEventDisableTiming));
        HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&bounce_[i]), CHUNK_BYTES, hipHostMallocDefault));
    }
    CopyPool::get();  // its threads start here, not under memory pressure
    last_ = ev_[0];
    prof_on_ = env::flag("TONA_COPY_PROF");
    grace_s_ = (double)env::integer("TONA_TEST_COPY_GRACE_S", (int64_t)backend::STALL_GRACE_S);
    stall_ms_ = env::integer("TONA_TEST_STASH_STALL_MS", -2);
    if (stall_ms_ == -2) return;
    const char* d = env::text("TONA_TEST_STASH_STALL_DIR");
    stall_dir_ = d && std::strcmp(d, "unstash") == 0 ? Dir::ToDevice : Dir::ToHost;
    HIPCHECK(trace::host_malloc(reinterpret_cast<void**>(&flag_host_), sizeof(uint32_t),
                                hipHostMallocMapped | hipHostMallocCoherent));
    HIPCHECK(hipHostGetDevicePointer(reinterpret_cast<void**>(&flag_dev_), flag_host_, 0));
    if (env::flag("TONA_TEST_STASH_STALL_SAFE")) g_release_word = flag_host_;  // test (b2)
}

// Polls without a blocking HIP call; spins for the first ~2 ms (a healthy chunk ends in that time), then sleeps
bool GuardedCopy::ready_within(const std::function<bool()>& ready, double limit_s) const {
    const auto t0 = Clock::now();
    for (int spin = 0; !ready(); spin++) {
        if (since(t0) >= limit_s) return false;
        if (spin > 2000) usleep(100);
    }
    return true;
}

bool GuardedCopy::event_within(hipEvent_t ev, double limit_s) const {
    return ready_within(
        [ev] {
            const hipError_t e = hipEventQuery(ev);
            if (e != hipSuccess && e != hipErrorNotReady) HIPCHECK(e);
            return e == hipSuccess;
        },
        limit_s);
}

// The test hook: the stream waits for a host word that is set after stall_ms_ (or never). The copy then stands
// without a kernel spinning, as it did in the hang. Once per process.
void GuardedCopy::arm_test_stall(Dir dir) {
    if (stall_ms_ == -2 || dir != stall_dir_) return;
    const long ms = stall_ms_;
    stall_ms_ = -2;
    *flag_host_ = 0;
    HIPCHECK(hipStreamWaitValue32(st_, flag_dev_, 1, hipStreamWaitValueEq, 0xFFFFFFFFu));
    fprintf(stderr, "test hook: the %s copy waits for a host word (%s)\n", dir_name(dir),
            ms < 0 ? "never set" : "set after the given time");
    if (ms >= 0)
        std::thread([w = flag_host_, ms] {
            std::this_thread::sleep_for(std::chrono::milliseconds(ms));
            __atomic_store_n(w, 1u, __ATOMIC_RELEASE);
        }).detach();
}

// The copy did not end in time: release host memory, give it STALL_GRACE_S to end, otherwise leave the process
bool GuardedCopy::stalled(const char* what, size_t done, size_t total, const std::function<bool()>& ready,
                          const Relief& relief) const {
    fprintf(stderr, "Stalled (%s) after %zu of %zu bytes: releasing host memory\n", what, done, total);
    const auto t0 = Clock::now();
    if (relief) relief();
    if (!ready_within(ready, grace_s_)) leave_stalled(what, done, total, since(t0));
    fprintf(stderr, "Stalled (%s) ended %.1f s after the release\n", what, since(t0));
    return false;
}

// The enqueue itself must not block the host; a call that does would hide a stall from the polling, so it is
// logged
void GuardedCopy::issue(Dir dir, void* dev, size_t len, int buf) {
    const auto t0 = Clock::now();
    const Armed armed(dir_name(dir), len, backend::chunk_deadline_s(len) + grace_s_);
    if (dir == Dir::ToHost)
        HIPCHECK(hipMemcpyAsync(bounce_[buf], dev, len, hipMemcpyDeviceToHost, st_));
    else
        HIPCHECK(hipMemcpyAsync(dev, bounce_[buf], len, hipMemcpyHostToDevice, st_));
    HIPCHECK(hipEventRecord(ev_[buf], st_));
    last_ = ev_[buf];
    prof_.issue += since(t0), prof_.chunks++, prof_.bytes += len;
    if (since(t0) > 0.5)
        fprintf(stderr, "Enqueue of a %zu-byte copy (%s) blocked the host for %.1f s\n", len, dir_name(dir), since(t0));
}

namespace {

struct Chunk {
    uint8_t* host;
    uint8_t* dev;
    size_t len;
};

std::vector<Chunk> split(const std::vector<GuardedCopy::Region>& regions, size_t chunk, size_t& total) {
    std::vector<Chunk> out;
    total = 0;
    for (const auto& r : regions)
        for (size_t off = 0; off < r.bytes; off += chunk) {
            const size_t len = std::min(chunk, r.bytes - off);
            out.push_back({static_cast<uint8_t*>(r.host) + off, static_cast<uint8_t*>(r.dev) + off, len});
        }
    for (const auto& r : regions) total += r.bytes;
    return out;
}

}  // namespace

// The first NBUF chunks are enqueued at once; while the CPU copies chunk c out of its buffer (on several threads), the
// DMA fills the buffers of the chunks after it. Stash and unstash run on the engine thread between passes, when k_mega
// is not running (see Mega::stash), so the copy threads compete with nothing.
bool GuardedCopy::to_host(const std::vector<Region>& regions, const Relief& relief) {
    arm_test_stall(Dir::ToHost);
    prof_.calls++;
    size_t total = 0, done = 0;
    const auto ch = split(regions, CHUNK_BYTES, total);
    const size_t n = ch.size();
    const auto lag = [this] { return hipEventQuery(last_) == hipSuccess; };
    for (size_t c = 0; c < std::min<size_t>(n, NBUF); c++) issue(Dir::ToHost, ch[c].dev, ch[c].len, (int)c);
    for (size_t c = 0; c < n; c++) {
        const int b = (int)(c % NBUF);
        const auto tw = Clock::now();
        const bool in_time = event_within(ev_[b], backend::chunk_deadline_s(ch[c].len));
        prof_.wait += since(tw);
        if (!in_time) return stalled(dir_name(Dir::ToHost), done, total, lag, relief);
        const auto tc = Clock::now();
        CopyPool::get().copy(ch[c].host, bounce_[b], ch[c].len);
        prof_.copy += since(tc);
        done += ch[c].len;
        if (c + NBUF < n) issue(Dir::ToHost, ch[c + NBUF].dev, ch[c + NBUF].len, b);
    }
    return true;
}

// The CPU fills a buffer only after the DMA that read it has ended (its event). Events of one stream end in order, so
// waiting for the last one at the end covers all.
//
// If the stream stands, the enqueue of the next chunk blocks the host before any event can be polled (measured, hk2b in
// the measurement file): then only the watchdog (Armed in issue) ends the process with EXIT_GPU_STALLED. The polling
// below is the second line, for a DMA that stands without the enqueue blocking.
bool GuardedCopy::to_device(const std::vector<Region>& regions, const Relief& relief) {
    arm_test_stall(Dir::ToDevice);
    prof_.calls++;
    size_t total = 0, done = 0;
    const auto ch = split(regions, CHUNK_BYTES, total);
    const auto lag = [this] { return hipEventQuery(last_) == hipSuccess; };
    for (size_t c = 0; c < ch.size(); c++) {
        const int b = (int)(c % NBUF);
        const auto tw = Clock::now();
        const bool in_time = c < (size_t)NBUF || event_within(ev_[b], backend::chunk_deadline_s(CHUNK_BYTES));
        prof_.wait += since(tw);
        if (!in_time) return stalled(dir_name(Dir::ToDevice), done, total, lag, relief);
        const auto tc = Clock::now();
        CopyPool::get().copy(bounce_[b], ch[c].host, ch[c].len);
        prof_.copy += since(tc);
        issue(Dir::ToDevice, ch[c].dev, ch[c].len, b);
        done += ch[c].len;
    }
    const auto tw = Clock::now();
    const bool in_time = ch.empty() || event_within(last_, backend::chunk_deadline_s(CHUNK_BYTES));
    prof_.wait += since(tw);
    return in_time || stalled(dir_name(Dir::ToDevice), total, total, lag, relief);
}

void GuardedCopy::prof_report(const char* op) {
    if (prof_on_)
        fprintf(stderr,
                "copy prof %s: %zu calls, %zu chunks, %.1f MB, enqueue %.1f ms, wait %.1f ms, cpu copy %.1f ms, idle "
                "%.2f ms, host allocation %.1f ms\n",
                op, prof_.calls, prof_.chunks, prof_.bytes / 1e6, prof_.issue * 1e3, prof_.wait * 1e3, prof_.copy * 1e3,
                prof_.idle * 1e3, prof_.setup * 1e3);
    prof_ = Prof{};
}

bool GuardedCopy::wait_idle(std::initializer_list<hipStream_t> streams, double limit_s, const Relief& relief) {
    const std::vector<hipStream_t> v(streams);
    const auto idle = [&v] {
        return std::all_of(v.begin(), v.end(), [](hipStream_t s) {
            const hipError_t e = hipStreamQuery(s);
            if (e != hipSuccess && e != hipErrorNotReady) HIPCHECK(e);
            return e == hipSuccess;
        });
    };
    const auto t0 = Clock::now();
    const bool ok = ready_within(idle, limit_s);
    prof_.idle += since(t0);
    if (ok) return true;
    return stalled("wait for idle streams", 0, 0, idle, relief);
}

}  // namespace tona::gpu
