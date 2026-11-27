// Persistent kernel: synchronization without a relaunch (part of gpubench.hip, included there: a kernel launch must
// sit in the same translation unit as the kernel).
//
// Every wait loop is bounded: the desktop runs on the same card and a hung kernel could trigger a GPU reset.
// When the bound is exceeded the kernel sets *err and aborts.
#pragma once
#include <hip/hip_cooperative_groups.h>
#include <hip/hip_runtime.h>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>

#include "gpubench_common.h"

namespace cg = cooperative_groups;
constexpr long SPIN_MAX = 50'000'000;  // roughly 1-2 s with s_sleep(1)

#define AGENT __HIP_MEMORY_SCOPE_AGENT
#define SYSTEM __HIP_MEMORY_SCOPE_SYSTEM

__global__ void k_grid_sync(int n) {
    cg::grid_group g = cg::this_grid();
    for (int i = 0; i < n; i++) g.sync();
}

// own barrier: counter plus generation, one thread per block
__device__ bool barrier(unsigned* count, unsigned* gen, unsigned nblocks, int* err) {
    __syncthreads();
    __shared__ bool ok;
    if (threadIdx.x == 0) {
        ok = true;
        unsigned g0 = __hip_atomic_load(gen, __ATOMIC_RELAXED, AGENT);
        if (__hip_atomic_fetch_add(count, 1u, __ATOMIC_ACQ_REL, AGENT) == nblocks - 1) {
            __hip_atomic_store(count, 0u, __ATOMIC_RELAXED, AGENT);
            __hip_atomic_fetch_add(gen, 1u, __ATOMIC_RELEASE, AGENT);
        } else {
            long spin = 0;
            while (__hip_atomic_load(gen, __ATOMIC_ACQUIRE, AGENT) == g0) {
                __builtin_amdgcn_s_sleep(1);
                if (++spin > SPIN_MAX || __hip_atomic_load(err, __ATOMIC_RELAXED, AGENT)) {
                    __hip_atomic_store(err, 1, __ATOMIC_RELAXED, AGENT);
                    ok = false;
                    break;
                }
            }
        }
    }
    __syncthreads();
    return ok;
}

__global__ void k_atomic_barrier(int n, unsigned* count, unsigned* gen, int* err) {
    for (int i = 0; i < n; i++)
        if (!barrier(count, gen, gridDim.x, err)) return;
}

// Point to point: block 0 and block 1 throw a number back and forth
__global__ void k_pingpong(int n, unsigned* flag, int* err) {
    if (threadIdx.x != 0) return;
    unsigned me = blockIdx.x;
    for (int i = 0; i < n; i++) {
        unsigned want = 2u * i + me;  // block 0 waits for even, block 1 for odd values
        long spin = 0;
        while (__hip_atomic_load(flag, __ATOMIC_ACQUIRE, AGENT) != want) {
            if (++spin > SPIN_MAX) {
                __hip_atomic_store(err, 1, __ATOMIC_RELAXED, AGENT);
                return;
            }
        }
        __hip_atomic_store(flag, want + 1, __ATOMIC_RELEASE, AGENT);
    }
}

// GPU and CPU through host memory (fine-grained): this is how the CPU could report expert results
// to a running kernel without a kernel launch
__global__ void k_host_pingpong(int n, unsigned* h2d, unsigned* d2h, int* err) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int i = 1; i <= n; i++) {
        long spin = 0;
        while (__hip_atomic_load(h2d, __ATOMIC_ACQUIRE, SYSTEM) != (unsigned)i) {
            if (++spin > SPIN_MAX) {
                __hip_atomic_store(err, 1, __ATOMIC_RELAXED, SYSTEM);
                return;
            }
        }
        __hip_atomic_store(d2h, (unsigned)i, __ATOMIC_RELEASE, SYSTEM);
    }
}

struct PersistBufs {
    int* err;
    unsigned* count;
    unsigned* gen;
    unsigned* flag;
};

static void check_err(const PersistBufs& b, const char* what) {
    int e = 0;
    CHECK(hipMemcpy(&e, b.err, 4, hipMemcpyDeviceToHost));
    if (e) {
        fprintf(stderr, "%s: wait loop aborted\n", what);
        exit(4);
    }
}

// grid.sync() against the own atomic barrier, for several grid sizes
static void persist_barriers(const PersistBufs& b) {
    const int threads = 256;
    int per_unit = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_unit, k_grid_sync, threads, 0));
    const int max_blocks = g_cus * per_unit;
    for (int blocks : {g_cus, 2 * g_cus, 4 * g_cus, max_blocks}) {
        if (blocks > max_blocks) continue;
        int n = 2000;
        void* args[] = {&n};
        CHECK(hipLaunchCooperativeKernel((void*)k_grid_sync, dim3(blocks), dim3(threads), args, 0, 0));
        CHECK(hipDeviceSynchronize());
        for (int r = 0; r < g_reps; r++) {
            float ms = timed([&] {
                CHECK(hipLaunchCooperativeKernel((void*)k_grid_sync, dim3(blocks), dim3(threads), args, 0, 0));
            });
            emit("persist", "grid.sync() (cooperative groups)", blocks, "us/barrier", ms * 1e3 / n);
        }
        for (int r = 0; r < g_reps; r++) {
            CHECK(hipMemset(b.err, 0, 4));
            CHECK(hipMemset(b.count, 0, 4));
            CHECK(hipMemset(b.gen, 0, 4));
            float ms = timed([&] { k_atomic_barrier<<<blocks, threads>>>(n, b.count, b.gen, b.err); });
            check_err(b, "atomic barrier");
            emit("persist", "own barrier (atomics)", blocks, "us/barrier", ms * 1e3 / n);
        }
    }
}

static void persist_pingpong(const PersistBufs& b) {
    const int n = 20000;
    for (int r = 0; r < g_reps; r++) {
        CHECK(hipMemset(b.err, 0, 4));
        CHECK(hipMemset(b.flag, 0, 4));
        float ms = timed([&] { k_pingpong<<<2, 64>>>(n, b.flag, b.err); });
        check_err(b, "pingpong");
        emit("persist", "block to block (half round trip)", 2, "us", ms * 1e3 / (2.0 * n));
    }
}

// One round of n exchanges with the running kernel; false if the GPU stops answering
static bool host_rounds(unsigned* h2d, unsigned* d2h, int n) {
    std::atomic_ref<unsigned> a_h2d(*h2d), a_d2h(*d2h);
    for (int i = 1; i <= n; i++) {
        a_h2d.store((unsigned)i, std::memory_order_release);
        long spin = 0;
        while (a_d2h.load(std::memory_order_acquire) != (unsigned)i)
            if (++spin > 200'000'000) return false;
    }
    return true;
}

static void persist_host(const PersistBufs& b) {
    unsigned *h2d, *d2h;
    CHECK(hipHostMalloc(&h2d, 64, hipHostMallocCoherent));
    CHECK(hipHostMalloc(&d2h, 64, hipHostMallocCoherent));
    const int hn = 20000;
    for (int r = 0; r < g_reps; r++) {
        *h2d = 0;
        *d2h = 0;
        CHECK(hipMemset(b.err, 0, 4));
        k_host_pingpong<<<1, 64>>>(hn, h2d, d2h, b.err);
        auto t0 = std::chrono::steady_clock::now();
        const bool ok = host_rounds(h2d, d2h, hn);
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count();
        CHECK(hipDeviceSynchronize());
        check_err(b, "host pingpong");
        if (!ok) {
            fprintf(stderr, "host pingpong: GPU does not respond\n");
            exit(4);
        }
        emit("persist", "CPU <-> GPU through host memory (half round trip)", 1, "us", us / (2.0 * hn));
    }
    CHECK(hipHostFree(h2d));
    CHECK(hipHostFree(d2h));
}

static void bench_persist() {
    PersistBufs b{};
    CHECK(hipMalloc(&b.err, 4));
    CHECK(hipMalloc(&b.count, 4));
    CHECK(hipMalloc(&b.gen, 4));
    CHECK(hipMalloc(&b.flag, 4));
    persist_barriers(b);
    persist_pingpong(b);
    persist_host(b);
    CHECK(hipFree(b.err));
    CHECK(hipFree(b.count));
    CHECK(hipFree(b.gen));
    CHECK(hipFree(b.flag));
}
