// Environment of the process that has to be set before the libraries load, so main() restarts itself once with it.
//
// OMP_WAIT_POLICY: libgomp reads it when the library loads. With the default policy, threads waiting at a barrier spin
// and use up the time slices that a pre-empted team member needs. With 8 busy competing threads the decode at 130K fell
// from 67 to 38 tok/s (normal mode; 26 with TONA_DETERMINISTIC=1); with `passive` it stays at 62 and costs nothing
// without load (research/measurements/2026-11-01-slow-decode.md).
//
// HSA_USERPTR_FOR_PAGED_MEM=0: libhsakmt reads it when the HSA runtime opens. By default hipHostMalloc memory is a KFD
// userptr; page migration (compaction, page cache reclaim) invalidates it and the driver evicts all queues of the
// process: slow decodes and barrier aborts. With 0 the memory is a kernel GTT buffer that the kernel does
// not move (research/measurements/2026-11-01-slow-decode.md). The GTT is limited (gpu/gtt_budget.h).
//
// OMP_NUM_THREADS: the physical cores (the 9600X: 6, the 9950X: 16), the one place that decides the thread count of
// every OpenMP team. libgomp reads it when the library loads, and a thread that main() starts later (the generator, the
// look-ahead of the embeddings) takes its team size from that value, so omp_set_num_threads() in main() is not enough.
// On the 9950X 16 threads against the 6 that the launchers set for the 9600X: decode +1.3 to +4.3 %, prefill equal
// (research/measurements/2026-11-17-omp-threads.md). Linux only; macOS keeps what its launcher sets. A value the user
// set (also from a launcher) stays.
#pragma once

#ifndef _WIN32
#include <unistd.h>
#endif

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>

namespace tona::runtime_env {

// Physical cores from the logical CPUs and the SMT state (/sys/devices/system/cpu/smt/active): with SMT the logical
// CPUs are twice the cores. 0 logical CPUs (unknown) counts as 1.
inline int physical_cores(unsigned logical, int smt_active) {
    if (logical == 0) return 1;
    return smt_active == 1 ? static_cast<int>(logical > 1 ? logical / 2 : 1) : static_cast<int>(logical);
}

inline int detect_physical_cores() {
    int smt = 0;
    std::ifstream f("/sys/devices/system/cpu/smt/active");
    if (!(f >> smt)) smt = 0;
    return physical_cores(std::thread::hardware_concurrency(), smt);
}

struct Plan {
    bool omp_policy = false;  // set OMP_WAIT_POLICY=passive
    bool userptr = false;     // set HSA_USERPTR_FOR_PAGED_MEM=0
    int omp_threads = 0;      // set OMP_NUM_THREADS to this; 0: leave it
    bool any() const { return omp_policy || userptr || omp_threads > 0; }
};

// A variable the user set stays as it is (also HSA_USERPTR_FOR_PAGED_MEM=1; an empty value counts as unset, like for
// TONA_*). The OMP policy is also left alone if the user chose a spin count. The restart sets what is missing, so the
// second start sees every value and cannot loop.
inline Plan plan(const char* policy, const char* spincount, const char* userptr, const char* threads = "",
                 int cores = 0) {
    auto unset = [](const char* v) { return !v || !*v; };
    return {unset(policy) && unset(spincount), unset(userptr), unset(threads) && cores > 0 ? cores : 0};
}

// First statement of main(), before any thread, HIP call, file or socket. On failure the program goes on with the
// defaults.
inline void set_defaults(char** argv) {
#ifdef __linux__
    const int cores = detect_physical_cores();
    const Plan p = plan(std::getenv("OMP_WAIT_POLICY"), std::getenv("GOMP_SPINCOUNT"),
                        std::getenv("HSA_USERPTR_FOR_PAGED_MEM"), std::getenv("OMP_NUM_THREADS"), cores);
    if (!p.any()) return;
    if (p.omp_policy) setenv("OMP_WAIT_POLICY", "passive", 1);
    if (p.userptr) setenv("HSA_USERPTR_FOR_PAGED_MEM", "0", 1);
    if (p.omp_threads > 0) setenv("OMP_NUM_THREADS", std::to_string(p.omp_threads).c_str(), 1);
    // Through the real path: exec of /proc/self/exe names the process "exe" (ss, ps, top), and an agent took the
    // server on its port for an unnamed process and killed it (2026-11-03). /proc/self/exe stays the fallback for a
    // binary that was replaced on disk.
    char self[4096];
    const ssize_t n = readlink("/proc/self/exe", self, sizeof self - 1);
    if (n > 0) {
        self[n] = 0;
        // argv is our own command line, handed on unchanged to the same binary: nothing to sanitize
        execv(self, argv);  // NOLINT(clang-analyzer-optin.taint.GenericTaint)
    }
    execv("/proc/self/exe", argv);  // NOLINT(clang-analyzer-optin.taint.GenericTaint)
    std::fprintf(stderr, "warning: restart with the runtime environment failed (%s), using the defaults\n",
                 std::strerror(errno));
    if (p.omp_policy) unsetenv("OMP_WAIT_POLICY");
    if (p.userptr) unsetenv("HSA_USERPTR_FOR_PAGED_MEM");
    if (p.omp_threads > 0) unsetenv("OMP_NUM_THREADS");
#else
    // The restart goes through /proc/self/exe, which macOS has not: it always failed there, warned on every start and
    // went on with libomp's defaults, and every Mac measurement ran that way. Not trying it keeps that behaviour;
    // whether OMP_WAIT_POLICY=passive helps libomp on Apple silicon is unmeasured
    (void)argv;
#endif
}

}  // namespace tona::runtime_env
