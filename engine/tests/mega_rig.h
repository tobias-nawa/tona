// Test rig for steps of the megakernel: a program of steps on the device, launched through a KernelSet of the probe
// kernels (gpu/probe/probe_kernels.h: the real k_mega with a test-only profile) with the barrier machinery as in
// production (grid barrier with time limit, abort word). The steps are tested as the kernel runs them, not as
// stand-alone functions. Host code only.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <vector>

#include "backend/rope_yarn.h"
#include "gpu/kernel_set.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_types.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;

namespace rig {

namespace gpu = tona::gpu;

// Width class of the kernel for a pass of nb tokens: 0 for one, 1 for up to 4, 2 for up to 8
inline int width_class(int nb) { return nb == 1 ? 0 : nb <= 4 ? 1 : 2; }

// Device state of a launch. One block per CU as the production launch (HIP counts WGPs of 2 CUs).
class Rig {
public:
    explicit Rig(const gpu::KernelSet& ks, double rope_base = 1e7)
        : ks_(ks),
          bar_(gpu::BAR_BYTES + gpu::BAR_HIST_BYTES),
          params_(sizeof(gpu::MegaParams)),
          ops_(16 * sizeof(gpu::MegaOp)) {
        int cus = 0;
        CHECK(hipDeviceGetAttribute(&wall_khz_, hipDeviceAttributeWallClockRate, 0));
        CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
        blocks_ = 2 * cus;
        ks_.init_tables(tona::rope::yarn_active(rope_base));
    }
    int blocks() const { return blocks_; }
    // The matvec tasks (DevTask array in device memory) that OP_MATVEC steps of the program refer to
    void set_tasks(const void* tasks) { tasks_ = tasks; }

    // Runs the program for the pass p and waits; false if a barrier aborted (the step is reported)
    bool run(const std::vector<gpu::MegaOp>& prog, const gpu::MegaParams& p, int blocks = 0) {
        start(prog, p);
        launch(p.nb, (int)prog.size(), blocks);
        CHECK(hipDeviceSynchronize());
        return aborted_step() == 0;
    }

    // Time of `reps` back-to-back launches of the same program, in microseconds per launch
    double time_us(const std::vector<gpu::MegaOp>& prog, const gpu::MegaParams& p, int reps) {
        hipEvent_t e0, e1;
        CHECK(hipEventCreate(&e0));
        CHECK(hipEventCreate(&e1));
        start(prog, p);
        launch(p.nb, (int)prog.size(), 0);  // warm-up: clocks and caches
        CHECK(hipDeviceSynchronize());
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < reps; i++) {
            reset_barrier();
            launch(p.nb, (int)prog.size(), 0);
        }
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        (void)hipEventDestroy(e0);
        (void)hipEventDestroy(e1);
        return 1000.0 * ms / reps;
    }

private:
    void start(const std::vector<gpu::MegaOp>& prog, const gpu::MegaParams& p) {
        CHECK(hipMemcpy(ops_.get(), prog.data(), prog.size() * sizeof(gpu::MegaOp), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(params_.get(), &p, sizeof p, hipMemcpyHostToDevice));
        reset_barrier();
    }
    // As Mega::start_kernel: counters zero, progress "not started", the abort diagnostics clear
    void reset_barrier() {
        auto* b = bar_.get<uint8_t>();
        CHECK(hipMemset(b + 4 * (size_t)gpu::BAR_YIELD, 0, 4));
        CHECK(hipMemset(b, 0, 32));
        CHECK(hipMemset(b + 64, 0xff, gpu::BAR_DONE - 64));
        CHECK(hipMemset(b + gpu::BAR_DONE, 0, gpu::BAR_BYTES - gpu::BAR_DONE));
    }
    void launch(int nb, int steps, int blocks) {
        const uint64_t khz = (uint64_t)wall_khz_;
        // 500 ms: well under the 2 s after which the driver resets the GPU; the passes here take microseconds
        const gpu::MegaScalars sc{0, steps, khz * 500, wall_khz_, 300, 0, ~0ull, khz * 5000, 0, nullptr};
        const gpu::KernelArgs a{ops_.get(), tasks_, params_.get<gpu::MegaParams>(),           bar_.get<unsigned>(),
                                nullptr,    sc,     (unsigned)(blocks > 0 ? blocks : blocks_)};
        ks_.launch(width_class(nb), {}, a);
    }
    // First aborted step + 1 (bar[3]), 0: none
    unsigned aborted_step() const {
        unsigned w = 0;
        CHECK(hipMemcpy(&w, bar_.get<uint8_t>() + 12, 4, hipMemcpyDeviceToHost));
        return w;
    }
    const gpu::KernelSet& ks_;
    DeviceBuffer bar_, params_, ops_;
    int wall_khz_ = 0, blocks_ = 0;
    const void* tasks_ = nullptr;
};

}  // namespace rig
