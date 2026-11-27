#include "gpu/clock_hold.h"

#include <hip/hip_runtime.h>

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <thread>

#include "gpu/mega.h"

namespace tona::gpu {

namespace {

// One wave: sleeps between polls of a flag in pinned host memory (the host's stop signal) and of the wall clock (the
// end of the chunk). Lighter than a busy loop and as good for the clock (clock_ramp.hip: 92 against 98 ms).
__global__ void hold_spin(const volatile int* stop, long long ticks) {
    const long long t0 = wall_clock64();
    while (!*stop && wall_clock64() - t0 < ticks) __builtin_amdgcn_s_sleep(20);
}

class Hold {
public:
    bool init(void* stream) {
        std::lock_guard lk(mu_);
        if (th_.joinable() || dead_) return !dead_;
        int wall_khz = 0;
        if (hipHostMalloc(reinterpret_cast<void**>(&flag_), sizeof(int), hipHostMallocMapped) != hipSuccess ||
            hipHostGetDevicePointer(reinterpret_cast<void**>(&dflag_), flag_, 0) != hipSuccess ||
            hipDeviceGetAttribute(&wall_khz, hipDeviceAttributeWallClockRate, 0) != hipSuccess || wall_khz <= 0 ||
            hipEventCreateWithFlags(&ev_, hipEventDisableTiming) != hipSuccess)
            return fail("setup");
        st_ = static_cast<hipStream_t>(stream);
        if (!st_ && hipStreamCreateWithFlags(&st_, hipStreamNonBlocking) != hipSuccess) return fail("stream");
        ticks_ = clock_hold_chunk_ticks(wall_khz);
        put(1);
        th_ = std::thread([this] { run(); });
        return true;
    }

    void set(bool on) {
        std::unique_lock lk(mu_);
        if (!th_.joinable() || dead_ || on == on_) return;
        on_ = on;
        if (on) return cv_.notify_all();
        put(1);  // under mu_: the thread starts a chunk only under mu_ and clears the flag there
        cv_.wait(lk, [&] { return !busy_ || dead_; });
    }

private:
    // The kernel reads the flag through the device mapping: a plain store could stay in a register
    void put(int v) { *static_cast<volatile int*>(flag_) = v; }

    bool fail(const char* what) {
        std::fprintf(stderr, "TONA_CLOCK_HOLD: %s failed, the hold is off\n", what);
        dead_ = true;
        return false;
    }

    // Thread: chunk after chunk while on_; the end of a chunk is polled with hipEventQuery, never a blocking call
    void run() {
        std::unique_lock lk(mu_);
        while (true) {
            cv_.wait(lk, [&] { return on_ || dead_; });
            if (dead_) return;
            busy_ = true;
            while (on_) {
                put(0);
                hold_spin<<<1, 32, 0, st_>>>(dflag_, ticks_);
                if (hipEventRecord(ev_, st_) != hipSuccess) {
                    fail("launch");
                    break;
                }
                lk.unlock();
                while (hipEventQuery(ev_) == hipErrorNotReady)
                    std::this_thread::sleep_for(std::chrono::microseconds(300));
                lk.lock();
            }
            busy_ = false;
            cv_.notify_all();
        }
    }

    std::mutex mu_;
    std::condition_variable cv_;
    std::thread th_;  // never joined: the hold lives as long as the process
    int* flag_ = nullptr;
    int* dflag_ = nullptr;
    hipStream_t st_ = nullptr;
    hipEvent_t ev_ = nullptr;
    long long ticks_ = 0;
    bool on_ = false, busy_ = false, dead_ = false;
};

Hold& hold() {
    static Hold* h = new Hold;  // leaked: the thread outlives static destruction
    return *h;
}

}  // namespace

void clock_hold_init(void* stream) { hold().init(stream); }

void clock_hold_set(bool on) { hold().set(on); }

long long clock_hold_chunk_ticks(int wall_khz) { return (long long)wall_khz * 250; }

void Mega::clock_hold(bool on) { clock_hold_set(on); }

}  // namespace tona::gpu
