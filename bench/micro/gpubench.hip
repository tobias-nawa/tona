// GPU micro-benchmarks: VRAM and cache bandwidth, PCIe, kernel launch cost, WMMA and FMA peak.
// Output: one JSON line per single measurement. Evaluation: bench/micro/run.py
// Usage: gpubench [vram|pcie|launch|persist|wmma|fma|all] [repetitions]
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "gpubench_common.h"
#include "gpubench_persist.h"

// ---------------------------------------------------------------- VRAM and caches

__device__ __host__ inline unsigned hash32(unsigned x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

// Random data: zeros could be shrunk by RDNA4 memory compression and fake bandwidth
__global__ void k_fill(uint4* __restrict__ p, size_t n, unsigned seed) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned h = hash32((unsigned)i * 4 + seed);
        p[i] = make_uint4(h, hash32(h), hash32(h + 1), hash32(h + 2));
    }
}

// passes: sweeps over the buffer inside one kernel, so small working sets are not dominated by launch cost
__global__ void k_read(const uint4* __restrict__ p, size_t n, int passes, unsigned* out) {
    uint4 acc = {0, 0, 0, 0};
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (int k = 0; k < passes; k++)
        for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
            uint4 v = p[i];
            acc.x ^= v.x;
            acc.y += v.y;
            acc.z ^= v.z;
            acc.w += v.w;
        }
    if ((acc.x ^ acc.y ^ acc.z ^ acc.w) == 0x12345678u)
        out[0] = acc.x;  // keeps the compiler from optimizing the loop away
}

__global__ void k_write(uint4* __restrict__ p, size_t n, unsigned seed) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned h = (unsigned)i * 0x9e3779b9u + seed;  // cheap, but not regular enough to be compressible
        p[i] = make_uint4(h, h ^ 0x5bd1e995u, h * 3u, ~h);
    }
}

__global__ void k_copy(const uint4* __restrict__ s, uint4* __restrict__ d, size_t n) {
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) d[i] = s[i];
}

static void bench_vram() {
    const size_t max_bytes = 4ull << 30;
    uint4 *a, *b;
    unsigned* out;
    CHECK(hipMalloc(&a, max_bytes));
    CHECK(hipMalloc(&b, max_bytes / 2));
    CHECK(hipMalloc(&out, 4));
    // Exactly as many blocks as are resident at once (here: multiProcessorCount = WGPs, 8 blocks of 8 waves each).
    // More blocks would run one after another, each over its part of the buffer, and small parts would stay in cache.
    int per_unit = 0;
    CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_unit, k_read, 256, 0));
    const int threads = 256, blocks = g_cus * per_unit;
    emit("vram", "resident blocks per unit", per_unit, "blocks", blocks);
    const size_t n1g = (1ull << 30) / sizeof(uint4);
    // Control for compression: 1 GiB of zeros against 1 GiB of random data
    CHECK(hipMemset(a, 0, 1ull << 30));
    k_read<<<blocks, threads>>>(a, n1g, 1, out);
    for (int r = 0; r < g_reps; r++) {
        float ms = timed([&] { k_read<<<blocks, threads>>>(a, n1g, 4, out); });
        emit("vram", "read zeros (control)", 1024, "GB/s", 4.0 * (1 << 30) / (ms * 1e6));
    }
    k_fill<<<blocks, threads>>>(a, max_bytes / sizeof(uint4), 1);
    k_fill<<<blocks, threads>>>(b, max_bytes / 2 / sizeof(uint4), 2);
    CHECK(hipDeviceSynchronize());
    // Read over the working set: small sets stay in L2 or the Infinity Cache
    for (size_t kib = 256; kib <= (4u << 20); kib *= 2) {
        size_t bytes = kib << 10, n = bytes / sizeof(uint4);
        int passes = (int)std::max<size_t>(1, (4ull << 30) / bytes);  // at least 4 GiB per measurement
        k_read<<<blocks, threads>>>(a, n, 1, out);
        CHECK(hipDeviceSynchronize());
        for (int r = 0; r < g_reps; r++) {
            float ms = timed([&] { k_read<<<blocks, threads>>>(a, n, passes, out); });
            emit("vram", "read random", (double)kib / 1024, "GB/s", (double)bytes * passes / (ms * 1e6));
        }
    }
    size_t n = n1g;  // 1 GiB, far above every cache
    for (int r = 0; r < g_reps; r++) {
        float ms = timed([&] {
            for (int i = 0; i < 4; i++) k_write<<<blocks, threads>>>(a, n, r * 4 + i);
        });
        emit("vram", "write", 1024, "GB/s", 4.0 * (1 << 30) / (ms * 1e6));
        ms = timed([&] {
            for (int i = 0; i < 4; i++) k_copy<<<blocks, threads>>>(a, b, n);
        });
        emit("vram", "copy (read+write)", 1024, "GB/s", 8.0 * (1 << 30) / (ms * 1e6));
    }
    CHECK(hipFree(a));
    CHECK(hipFree(b));
    CHECK(hipFree(out));
}

// ---------------------------------------------------------------- PCIe

static void bench_pcie() {
    const size_t max_bytes = 1ull << 30;
    void *pinned, *dev, *dev2, *pinned2;
    CHECK(hipHostMalloc(&pinned, max_bytes, hipHostMallocDefault));
    CHECK(hipHostMalloc(&pinned2, max_bytes, hipHostMallocDefault));
    void* pageable = malloc(max_bytes);
    memset(pageable, 1, max_bytes);
    memset(pinned, 1, max_bytes);
    CHECK(hipMalloc(&dev, max_bytes));
    CHECK(hipMalloc(&dev2, max_bytes));
    for (size_t kib = 64; kib <= (1u << 20); kib *= 4) {
        size_t bytes = kib << 10;
        int inner = (int)std::max<size_t>(1, (1ull << 30) / bytes);
        for (int r = 0; r < g_reps; r++) {
            float ms = timed([&] {
                for (int i = 0; i < inner; i++) CHECK(hipMemcpyAsync(dev, pinned, bytes, hipMemcpyHostToDevice, 0));
            });
            emit("pcie", "H2D pinned", (double)kib, "GB/s", (double)bytes * inner / (ms * 1e6));
            ms = timed([&] {
                for (int i = 0; i < inner; i++) CHECK(hipMemcpyAsync(pinned, dev, bytes, hipMemcpyDeviceToHost, 0));
            });
            emit("pcie", "D2H pinned", (double)kib, "GB/s", (double)bytes * inner / (ms * 1e6));
        }
    }
    for (int r = 0; r < g_reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        CHECK(hipMemcpy(dev, pageable, max_bytes, hipMemcpyHostToDevice));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        emit("pcie", "H2D pageable", 1 << 20, "GB/s", max_bytes / s / 1e9);
    }
    // both directions at once, 256 MiB each
    hipStream_t s1, s2;
    CHECK(hipStreamCreateWithFlags(&s1, hipStreamNonBlocking));
    CHECK(hipStreamCreateWithFlags(&s2, hipStreamNonBlocking));
    const size_t bytes = 256u << 20;
    for (int r = 0; r < g_reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 4; i++) {
            CHECK(hipMemcpyAsync(dev, pinned, bytes, hipMemcpyHostToDevice, s1));
            CHECK(hipMemcpyAsync(pinned2, dev2, bytes, hipMemcpyDeviceToHost, s2));
        }
        CHECK(hipStreamSynchronize(s1));
        CHECK(hipStreamSynchronize(s2));
        double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        emit("pcie", "bidirectional pinned (sum)", 256 * 1024, "GB/s", 8.0 * bytes / s / 1e9);
    }
    CHECK(hipStreamDestroy(s1));
    CHECK(hipStreamDestroy(s2));
    CHECK(hipHostFree(pinned));
    CHECK(hipHostFree(pinned2));
    CHECK(hipFree(dev));
    CHECK(hipFree(dev2));
    free(pageable);
}

// ---------------------------------------------------------------- Kernel launch cost

__global__ void k_empty() {}

static void bench_launch() {
    const int n = 5000;
    k_empty<<<1, 32>>>();
    CHECK(hipDeviceSynchronize());
    hipStream_t s;
    CHECK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    for (int r = 0; r < g_reps; r++) {
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) {
            k_empty<<<1, 32, 0, s>>>();
            CHECK(hipStreamSynchronize(s));
        }
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        emit("launch", "launch+sync (round trip)", 1, "us", us);

        t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < n; i++) k_empty<<<1, 32, 0, s>>>();
        CHECK(hipStreamSynchronize(s));
        us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / n;
        emit("launch", "launches back to back", 1, "us/kernel", us);
    }
    // HIP graph with 200 kernels, like a decode pass
    const int k = 200;
    hipGraph_t g;
    hipGraphExec_t ge;
    CHECK(hipStreamBeginCapture(s, hipStreamCaptureModeGlobal));
    for (int i = 0; i < k; i++) k_empty<<<1, 32, 0, s>>>();
    CHECK(hipStreamEndCapture(s, &g));
    CHECK(hipGraphInstantiate(&ge, g, nullptr, nullptr, 0));
    CHECK(hipGraphLaunch(ge, s));
    CHECK(hipStreamSynchronize(s));
    for (int r = 0; r < g_reps; r++) {
        const int m = 50;
        auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < m; i++) CHECK(hipGraphLaunch(ge, s));
        CHECK(hipStreamSynchronize(s));
        double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() / (m * k);
        emit("launch", "HIP graph, 200 kernels", k, "us/kernel", us);
    }
    CHECK(hipGraphExecDestroy(ge));
    CHECK(hipGraphDestroy(g));
    CHECK(hipStreamDestroy(s));
}

// ---------------------------------------------------------------- WMMA and FMA

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef short short8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int8v __attribute__((ext_vector_type(8)));
typedef int int2v __attribute__((ext_vector_type(2)));

constexpr int ACC = 8;  // independent accumulators per wave, to hide the WMMA latency
constexpr int ITER = 4096;

#define WMMA_KERNEL(NAME, AT, CT, INIT_A, CALL)                          \
    __global__ void NAME(float* out, int seed) {                         \
        AT a, b;                                                         \
        for (int i = 0; i < (int)(sizeof(AT) / sizeof(a[0])); i++) {     \
            INIT_A;                                                      \
        }                                                                \
        CT c[ACC] = {};                                                  \
        for (int it = 0; it < ITER; it++) {                              \
            _Pragma("unroll") for (int j = 0; j < ACC; j++) c[j] = CALL; \
        }                                                                \
        float s = 0;                                                     \
        for (int j = 0; j < ACC; j++)                                    \
            for (int i = 0; i < 8; i++) s += (float)c[j][i];             \
        if (s == -1.0f) out[0] = s;                                      \
    }

WMMA_KERNEL(k_wmma_f16_f32, half8, float8,
            (a[i] = (_Float16)((threadIdx.x + i + seed) * 1e-3f), b[i] = (_Float16)((i - seed) * 1e-3f)),
            __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c[j]))
WMMA_KERNEL(k_wmma_f16_f16, half8, half8,
            (a[i] = (_Float16)((threadIdx.x + i + seed) * 1e-3f), b[i] = (_Float16)((i - seed) * 1e-3f)),
            __builtin_amdgcn_wmma_f16_16x16x16_f16_w32_gfx12(a, b, c[j]))
WMMA_KERNEL(k_wmma_bf16_f32, short8, float8, (a[i] = (short)(threadIdx.x + i + seed), b[i] = (short)(i - seed)),
            __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c[j]))
WMMA_KERNEL(k_wmma_fp8_f32, int2v, float8,
            (a[i] = (int)(threadIdx.x * 0x01010101u + seed), b[i] = (int)(i * 0x01010101u - seed)),
            __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c[j]))
WMMA_KERNEL(k_wmma_iu8_i32, int2v, int8v,
            (a[i] = (int)(threadIdx.x * 0x01010101u + seed), b[i] = (int)(i * 0x01010101u - seed)),
            __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32_gfx12(true, a, true, b, c[j], false))
WMMA_KERNEL(k_wmma_iu4_k32_i32, int2v, int8v,
            (a[i] = (int)(threadIdx.x * 0x11111111u + seed), b[i] = (int)(i * 0x11111111u - seed)),
            __builtin_amdgcn_wmma_i32_16x16x32_iu4_w32_gfx12(true, a, true, b, c[j], false))

__global__ void k_fma(float* out, int seed) {
    float x[ACC];
    for (int j = 0; j < ACC; j++) x[j] = (threadIdx.x + j + seed) * 1e-3f;
    const float m = 0.999f, k = 1e-4f;
    for (int it = 0; it < ITER * 8; it++) {
        _Pragma("unroll") for (int j = 0; j < ACC; j++) x[j] = __builtin_fmaf(x[j], m, k);
    }
    float s = 0;
    for (int j = 0; j < ACC; j++) s += x[j];
    if (s == -1.0f) out[0] = s;
}

static void run_peak(const char* var, void (*kern)(float*, int), double flop_per_wave_op, double ops_per_wave,
                     const char* unit) {
    float* out;
    CHECK(hipMalloc(&out, 4));
    const int threads = 256, blocks = g_cus * 16, launches = 8;
    // warm-up of ~0.5 s so clock and power have settled
    auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.5) {
        kern<<<blocks, threads>>>(out, 1);
        CHECK(hipDeviceSynchronize());
    }
    double waves = (double)blocks * threads / 32;
    for (int r = 0; r < g_reps; r++) {
        float ms = timed([&] {
            for (int i = 0; i < launches; i++) kern<<<blocks, threads>>>(out, r + i);
        });
        emit("peak", var, 0, unit, launches * waves * ops_per_wave * flop_per_wave_op / (ms * 1e9));
    }
    CHECK(hipFree(out));
}

static void bench_wmma() {
    const double wmma_ops = (double)ITER * ACC;  // WMMA instructions per wave
    const double f16x16 = 2.0 * 16 * 16 * 16;
    run_peak("WMMA f16 -> f32", k_wmma_f16_f32, f16x16, wmma_ops, "TFLOPS");
    run_peak("WMMA f16 -> f16", k_wmma_f16_f16, f16x16, wmma_ops, "TFLOPS");
    run_peak("WMMA bf16 -> f32", k_wmma_bf16_f32, f16x16, wmma_ops, "TFLOPS");
    run_peak("WMMA fp8 -> f32", k_wmma_fp8_f32, f16x16, wmma_ops, "TFLOPS");
    run_peak("WMMA int8 -> i32", k_wmma_iu8_i32, f16x16, wmma_ops, "TOPS");
    run_peak("WMMA int4 (K=32) -> i32", k_wmma_iu4_k32_i32, 2.0 * 16 * 16 * 32, wmma_ops, "TOPS");
}

static void bench_fma() {
    // per lane and iteration ACC FMAs of 2 FLOP each, 32 lanes per wave
    run_peak("FMA f32 (vector)", k_fma, 2.0 * 32, (double)ITER * 8 * ACC, "TFLOPS");
}

int main(int argc, char** argv) {
    std::string what = argc > 1 ? argv[1] : "all";
    if (argc > 2) g_reps = atoi(argv[2]);
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    g_cus = p.multiProcessorCount;
    int rt = 0;
    CHECK(hipRuntimeGetVersion(&rt));
    printf(
        "{\"device\":\"%s\",\"arch\":\"%s\",\"cus\":%d,\"clock_khz\":%d,\"mem_clock_khz\":%d,\"bus_bits\":%d,"
        "\"l2_bytes\":%d,\"hip_runtime\":%d,\"hip_headers\":%d}\n",
        p.name, p.gcnArchName, p.multiProcessorCount, p.clockRate, p.memoryClockRate, p.memoryBusWidth, p.l2CacheSize,
        rt, HIP_VERSION);
    fflush(stdout);
    if (what == "vram" || what == "all") bench_vram();
    if (what == "pcie" || what == "all") bench_pcie();
    if (what == "launch" || what == "all") bench_launch();
    if (what == "persist" || what == "all") bench_persist();
    if (what == "wmma" || what == "all") bench_wmma();
    if (what == "fma" || what == "all") bench_fma();
    return 0;
}
