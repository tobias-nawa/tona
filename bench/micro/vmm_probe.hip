// Does the virtual-memory API (hipMemAddressReserve / hipMemCreate / hipMemMap / hipMemUnmap) work on this stack, what
// does a map and an unmap of a chunk cost, and does an unmap give VRAM back to the card (sysfs mem_info_vram_used)?
// Needed to decide how the expert slot pool can return VRAM (docs/architecture/expert-tiers.md). Standalone: allocates
// 1 GB. Build: hipcc -O2 --offload-arch=gfx1200 -std=c++20 vmm_probe.hip -o vmm_probe
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#define CK(x)                                                                                \
    do {                                                                                     \
        const hipError_t e_ = (x);                                                           \
        if (e_ != hipSuccess) {                                                              \
            std::printf("FAILED %s -> %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); \
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

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
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) p[i] = v;
}
__global__ void sum(const unsigned* p, size_t n, unsigned long long* out) {
    unsigned long long s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        s += p[i];
    atomicAdd(out, s);
}

struct Pool {
    void* base = nullptr;
    size_t chunk = 0, n_chunks = 32, total = 0, words = 0;
    hipMemAllocationProp prop = {};
    hipMemAccessDesc acc = {};
    std::vector<hipMemGenericAllocationHandle_t> h;
    unsigned long long* out = nullptr;
};

static void rep(const char* what, const std::vector<double>& v) {
    double sum = 0, mx = 0;
    for (double x : v) sum += x, mx = std::max(mx, x);
    std::printf("%-22s n=%zu mean %.3f ms  max %.3f ms\n", what, v.size(), sum / v.size(), mx);
}

static char* at(const Pool& P, size_t i) { return static_cast<char*>(P.base) + i * P.chunk; }

// Reserve, create and map all chunks, give the device access, fill and sum the whole range
static int map_all(Pool& P, size_t gran) {
    const long long v0 = vram_used_mib();
    auto t = Clock::now();
    CK(hipMemAddressReserve(&P.base, P.total, gran, nullptr, 0));
    std::printf("reserve %zu MiB of address space: %.3f ms\n", P.total >> 20, ms_since(t));
    std::vector<double> t_create, t_map;
    for (size_t i = 0; i < P.n_chunks; i++) {
        t = Clock::now();
        CK(hipMemCreate(&P.h[i], P.chunk, &P.prop, 0));
        t_create.push_back(ms_since(t));
        t = Clock::now();
        CK(hipMemMap(at(P, i), P.chunk, 0, P.h[i], 0));
        t_map.push_back(ms_since(t));
    }
    t = Clock::now();
    CK(hipMemSetAccess(P.base, P.total, &P.acc, 1));
    std::printf("set access on all: %.3f ms\n", ms_since(t));
    std::printf("VRAM used by the card: before %lld MiB, after mapping %lld MiB (+%lld)\n", v0, vram_used_mib(),
                vram_used_mib() - v0);
    rep("hipMemCreate", t_create);
    rep("hipMemMap", t_map);
    fill<<<256, 256>>>(static_cast<unsigned*>(P.base), P.words, 1u);
    CK(hipDeviceSynchronize());
    CK(hipMalloc(&P.out, 8));
    CK(hipMemset(P.out, 0, 8));
    sum<<<256, 256>>>(static_cast<unsigned*>(P.base), P.words, P.out);
    unsigned long long host = 0;
    CK(hipMemcpy(&host, P.out, 8, hipMemcpyDeviceToHost));
    std::printf("kernel over the whole mapping: sum %llu (expected %zu) %s\n", host, P.words,
                host == P.words ? "ok" : "WRONG");
    return 0;
}

// Unmap and release the upper half chunk by chunk while a kernel keeps reading the lower half on another stream
static int unmap_half(Pool& P) {
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    CK(hipMemset(P.out, 0, 8));
    const long long before = vram_used_mib();
    std::vector<double> t_unmap, t_release;
    for (size_t i = P.n_chunks / 2; i < P.n_chunks; i++) {
        sum<<<64, 256, 0, s>>>(static_cast<unsigned*>(P.base), P.words / 2, P.out);
        auto t = Clock::now();
        CK(hipMemUnmap(at(P, i), P.chunk));
        t_unmap.push_back(ms_since(t));
        t = Clock::now();
        CK(hipMemRelease(P.h[i]));
        t_release.push_back(ms_since(t));
    }
    CK(hipStreamSynchronize(s));
    std::printf("VRAM used by the card after unmapping and releasing half: %lld MiB (%+lld)\n", vram_used_mib(),
                vram_used_mib() - before);
    rep("hipMemUnmap (GPU busy)", t_unmap);
    rep("hipMemRelease", t_release);
    CK(hipMemset(P.out, 0, 8));
    sum<<<256, 256>>>(static_cast<unsigned*>(P.base), P.words / 2, P.out);
    unsigned long long host = 0;
    CK(hipMemcpy(&host, P.out, 8, hipMemcpyDeviceToHost));
    std::printf("lower half after the unmap: sum %llu (expected %zu) %s\n", host, P.words / 2,
                host == P.words / 2 ? "ok" : "WRONG");
    const size_t last = P.n_chunks - 1;
    auto t = Clock::now();
    CK(hipMemCreate(&P.h[last], P.chunk, &P.prop, 0));
    CK(hipMemMap(at(P, last), P.chunk, 0, P.h[last], 0));
    CK(hipMemSetAccess(at(P, last), P.chunk, &P.acc, 1));
    std::printf("create + map + set access again of one chunk: %.3f ms\n", ms_since(t));
    return 0;
}

int main() {
    int dev = 0, vmm = -1;
    CK(hipSetDevice(dev));
    CK(hipDeviceGetAttribute(&vmm, hipDeviceAttributeVirtualMemoryManagementSupported, dev));
    std::printf("virtual memory management supported: %d\n", vmm);
    Pool P;
    P.prop.type = hipMemAllocationTypePinned;
    P.prop.location.type = hipMemLocationTypeDevice;
    P.prop.location.id = dev;
    P.acc.location = P.prop.location;
    P.acc.flags = hipMemAccessFlagsProtReadWrite;
    size_t gran = 0, gran_rec = 0;
    CK(hipMemGetAllocationGranularity(&gran, &P.prop, hipMemAllocationGranularityMinimum));
    CK(hipMemGetAllocationGranularity(&gran_rec, &P.prop, hipMemAllocationGranularityRecommended));
    std::printf("granularity: minimum %zu KiB, recommended %zu KiB\n", gran >> 10, gran_rec >> 10);
    P.chunk = (32u << 20) < gran ? gran : ((32u << 20) / gran) * gran;  // about 32 MB
    P.total = P.chunk * P.n_chunks, P.words = P.total / 4;
    P.h.resize(P.n_chunks);
    if (map_all(P, gran)) return 1;
    return unmap_half(P);
}
