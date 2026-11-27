// Small helpers of the Mega host code.
#pragma once

#include <hip/hip_runtime.h>
#include <sys/resource.h>

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <format>
#include <stdexcept>
#include <vector>

#include "backend/backend.h"
#include "backend/env.h"
#include "gpu/alloc_trace.h"
#include "gpu/kv_place.h"
#include "model/gguf.h"
#include "model/quant.h"

#define HIPCHECK(x)                                                                                           \
    do {                                                                                                      \
        hipError_t err_ = (x);                                                                                \
        if (err_ != hipSuccess) throw std::runtime_error(std::format("{}: {}", #x, hipGetErrorString(err_))); \
    } while (0)

namespace tona::gpu {

// TONA_TOKHIST: prints the histogram of tokens per union expert (mega_init.cpp)
void report_tokhist(const backend::Stats& stats);

// TONA_PROF_LAYERS (with TONA_MEGA_PROF): GPU-computed union experts per layer, summed over the recorded passes
std::vector<double>& prof_gpu_experts();

// Free VRAM of the dGPU according to amdgpu (total - used, desktop included); hipMemGetInfo does not see the desktop
int64_t vram_free_sysfs();

// Where the KV cache of the n_main attention layers (and the MTP layer, if there is one) lives: the first cells of
// every layer in VRAM within a budget, the rest in pinned RAM (kv_place.h). The budget: TONA_KV_VRAM_GB, else a share
// of the VRAM that is free now minus TONA_VRAM_RESERVE_MB.
KvSplit plan_kv_split(size_t n_main, size_t n_mtp, int64_t max_ctx, int64_t row);

// Number of CUs the megakernel leaves to the desktop (TONA_FREE_CUS)
int free_cus();

// Measuring mode TONA_FAKE_EXPERTS: experts are not loaded or computed correctly
inline bool fake_experts() {
    static const bool on = env::flag("TONA_FAKE_EXPERTS");
    return on;
}

// Seconds between two steady-clock points
template <class Clock>
double seconds_since(typename Clock::time_point t0) {
    return std::chrono::duration<double>(Clock::now() - t0).count();
}

inline std::vector<float> dequant_host(const TensorInfo& t, int64_t row = 0, int64_t n = -1) {
    if (n < 0) n = (int64_t)t.elements();
    std::vector<float> out(n);
    dequantize_row(t.type, t.data + row * type_row_bytes(t.type, n), out.data(), n);
    return out;
}

// TONA_MEGA_STALL_MS=n: a stretch of CPU work the kernel waits for that takes longer than n ms, with the page faults of
// the process in it (the kernel yields after 100 ms; the CPU side decides how long the pass takes)
class CpuStall {
public:
    CpuStall() : on_(limit_ms() > 0) {
        if (!on_) return;  // no syscall and no clock read on the path the kernel waits for
        t0_ = std::chrono::steady_clock::now();
        getrusage(RUSAGE_SELF, &r0_);
    }
    void report(const char* what, int layer, int64_t pos) const {
        if (!on_) return;
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0_).count();
        if (ms <= limit_ms()) return;
        rusage r1;
        getrusage(RUSAGE_SELF, &r1);
        std::fprintf(stderr,
                     "gpu::Mega: CPU %s %d took %.1f ms at position %lld (minor faults +%ld, major +%ld, %lld ms)\n",
                     what, layer, ms, (long long)pos, r1.ru_minflt - r0_.ru_minflt, r1.ru_majflt - r0_.ru_majflt,
                     (long long)std::chrono::duration_cast<std::chrono::milliseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count());
    }

private:
    static double limit_ms() {
        static const double v = (double)env::integer("TONA_MEGA_STALL_MS", 0);
        return v;
    }
    bool on_;
    std::chrono::steady_clock::time_point t0_{};
    rusage r0_{};
};

}  // namespace tona::gpu
