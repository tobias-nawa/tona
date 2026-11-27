// What the planner assumes a CPU expert costs. Normally the fixed base (60 us, measured at 6 threads on the 9600X,
// research/measurements/2026-10-25-cpu-experts). When other processes take the CPU the experts run 10x slower
// (research/measurements/2026-11-01-slow-decode.md) and the GPU waits for them in the kernel, so the estimate follows
// the measured time and the planner sends the misses to the GPU instead (pinned experts can be read directly; the
// result is bit-identical, checked there).
#pragma once

#include <algorithm>

namespace tona::backend {

// Host side only, no HIP; one instance per Mega, no locking: split_misses (reads, via select) and compute_cpu_experts
// (writes, via cpu_experts) run one after the other in the router loop of Mega::serve_pass, on the thread that called
// it.
class CpuCost {
public:
    // Measured cost per expert in the normal state is 76-80 us against the base of 60: below kDead times the base
    // the estimate stays at the base, so that the plan without load is the one measured before.
    static constexpr double kDead = 2.0;
    static constexpr double kMax = 32;          // ratio cap: all pinned misses are on the GPU long before
    static constexpr double kRise = 1.0 / 16;   // weight of a sample, about one pass of layers
    static constexpr double kFall = 1.0 / 160;  // decay per skipped layer: the CPU is probed again after ~5 passes

    CpuCost() = default;
    CpuCost(double base_us, bool adaptive) : base_(base_us), adaptive_(adaptive) {}

    // Cost of one CPU expert for the planner (us)
    double us() const { return base_ * std::max(1.0, ratio_ / kDead); }

    // A layer's CPU experts took elapsed_us for n experts
    void observe(int n, double elapsed_us) {
        if (!adaptive_ || n <= 0 || base_ <= 0) return;
        const double sample = std::min(kMax * kDead, elapsed_us / n / base_);
        ratio_ += kRise * (sample - ratio_);
        ratio_ = std::max(ratio_, 1.0);
    }

    // The planner had pinned misses but sent none to the CPU: without samples the estimate would never come back
    void skipped() {
        if (!adaptive_) return;
        ratio_ = std::max(1.0, ratio_ - kFall * (ratio_ - 1.0));
    }

    double ratio() const { return ratio_; }

private:
    double base_ = 60;
    bool adaptive_ = false;
    double ratio_ = 1;  // measured / base, as a moving average
};

// Round robin of the planner: the next pinned miss goes to the GPU if that side would finish no later than the CPU
// with one more expert
inline bool to_gpu(double t_gpu, double t_cpu, double gd_us, double cpu_us) { return t_gpu + gd_us <= t_cpu + cpu_us; }

}  // namespace tona::backend
