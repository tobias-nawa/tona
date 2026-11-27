// The whole slot range from 32 MB chunks instead of one hipMalloc (docs/architecture/vram-giveback.md): is every
// hipMemCreate its own allocation (VRAM used grows by one chunk per create and falls by one chunk when a handle is
// released), what do 260 of them cost at build time against one hipMalloc of the same size, and does one kernel read
// and write the contiguous range unchanged. No VRAM pressure: 8.3 GB on a card that nothing else uses (run through the
// GPU queue, never beside a server). Build: hipcc -O2 --offload-arch=gfx1200 -std=c++20 vmm_chunks_build.hip -o
// vmm_chunks_build
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

#define CK(x)                                                                                \
    do {                                                                                     \
        const hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                              \
            std::printf("FAILED %s -> %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static long long vram_used_mib() {
    for (int c = 0; c < 8; c++) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_total", c);
        long long total = 0, used = -1;
        std::ifstream(path) >> total;
        if (total < (4LL << 30)) continue;
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_used", c);
        std::ifstream(path) >> used;
        return used >> 20;
    }
    return -1;
}

__global__ void fill(unsigned* p, size_t n, unsigned v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = v ^ (unsigned)i;
}
__global__ void check(const unsigned* p, size_t n, unsigned v, unsigned long long* bad) {
    unsigned long long b = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b += p[i] != (v ^ (unsigned)i);
    if (b) atomicAdd(bad, b);
}

constexpr size_t TOTAL = 260ull * (32ull << 20);  // 8.7 GB whatever the chunk size
static size_t CHUNK = 32ull << 20;
static int N = 260;

static void one_malloc() {
    const long long v0 = vram_used_mib();
    void* p = nullptr;
    auto t = Clock::now();
    CK(hipMalloc(&p, N * CHUNK));
    std::printf("one hipMalloc of %.2f GB: %.2f ms, VRAM used %+lld MiB\n", N * CHUNK / 1e9, ms(t),
                vram_used_mib() - v0);
    CK(hipFree(p));
}

static void chunked() {
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    hipMemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    size_t gran = 0;
    CK(hipMemGetAllocationGranularity(&gran, &prop, hipMemAllocationGranularityMinimum));
    void* base = nullptr;
    CK(hipMemAddressReserve(&base, N * CHUNK, gran, nullptr, 0));
    std::vector<hipMemGenericAllocationHandle_t> h(N);
    std::vector<long long> step_mib;
    const long long v0 = vram_used_mib();
    auto t = Clock::now();
    for (int i = 0; i < N; i++) {
        const bool probe = i < 4 || i == N - 1;
        const long long before = probe ? vram_used_mib() : 0;
        CK(hipMemCreate(&h[i], CHUNK, &prop, 0));
        CK(hipMemMap(static_cast<char*>(base) + i * CHUNK, CHUNK, 0, h[i], 0));
        if (probe) step_mib.push_back(vram_used_mib() - before);
    }
    CK(hipMemSetAccess(base, N * CHUNK, &acc, 1));
    std::printf(
        "%d chunks of %zu MiB (granularity %zu): create+map+access %.1f ms (%.3f ms per chunk, includes 5 sysfs "
        "reads in total), "
        "VRAM used %+lld MiB\n",
        N, CHUNK >> 20, gran, ms(t), ms(t) / N, vram_used_mib() - v0);
    std::printf("VRAM used per create+map step (first four and last): %lld %lld %lld %lld ... %lld MiB\n", step_mib[0],
                step_mib[1], step_mib[2], step_mib[3], step_mib[4]);
    unsigned long long* bad;
    CK(hipMalloc(&bad, 8));
    CK(hipMemset(bad, 0, 8));
    const size_t words = N * CHUNK / 4;
    t = Clock::now();
    fill<<<4096, 256>>>(static_cast<unsigned*>(base), words, 0x5a5a1234u);
    check<<<4096, 256>>>(static_cast<unsigned*>(base), words, 0x5a5a1234u, bad);
    CK(hipDeviceSynchronize());
    unsigned long long b = 0;
    CK(hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost));
    std::printf("one kernel fills and checks the contiguous %.2f GB: %.1f ms, wrong words %llu\n", N * CHUNK / 1e9,
                ms(t), b);
    // Give one chunk back: unmap and release its handle; VRAM used must fall by one chunk
    const long long v1 = vram_used_mib();
    t = Clock::now();
    CK(hipMemUnmap(static_cast<char*>(base) + 100 * CHUNK, CHUNK));
    CK(hipMemRelease(h[100]));
    const double give = ms(t);
    std::printf("one chunk unmapped and released: %.3f ms, VRAM used %+lld MiB\n", give, vram_used_mib() - v1);
    t = Clock::now();
    for (int i = 0; i < N; i++)
        if (i != 100) CK(hipMemUnmap(static_cast<char*>(base) + i * CHUNK, CHUNK));
    for (int i = 0; i < N; i++)
        if (i != 100) CK(hipMemRelease(h[i]));
    std::printf("remaining %d chunks unmapped and released: %.1f ms, VRAM used now %+lld MiB against the start\n",
                N - 1, ms(t), vram_used_mib() - v0);
    CK(hipFree(bad));
}

int main(int argc, char** argv) {
    if (argc > 1) CHUNK = (size_t)std::atoi(argv[1]) << 20;  // chunk size in MiB: the sweep of the build cost
    N = (int)(TOTAL / CHUNK);
    CK(hipSetDevice(0));
    std::printf("VRAM used at start: %lld MiB\n", vram_used_mib());
    one_malloc();
    chunked();
    return 0;
}
