// E5 of research/58 (docs/architecture/vram-giveback.md): is every hipMemCreate chunk its own kernel BO, and what do
// ~260 chunks of 32 MB mapped into one contiguous range cost? Run under `strace -f -e trace=ioctl` and count the KFD
// ALLOC_MEMORY_OF_GPU / MAP_MEMORY_TO_GPU ioctls with and without chunks (bench/micro/vmm_chunks.sh). Needs the card
// quiet and idle: it refuses to run unless the free VRAM exceeds the chunks plus 3 GB, so it never creates pressure.
// Usage: vmm_chunks CHUNKS [CHUNK_MB, default 32]. Build: see vmm_chunks.sh.
#include <hip/hip_runtime.h>

#include <algorithm>
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
            return 1;                                                                        \
        }                                                                                    \
    } while (0)

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

static long long sysfs_bytes(const char* name) {
    for (int c = 0; c < 8; c++) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_total", c);
        long long total = 0, v = -1;
        std::ifstream(path) >> total;
        if (total < (4LL << 30)) continue;
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/%s", c, name);
        std::ifstream(path) >> v;
        return v;
    }
    return -1;
}

__global__ void fill(unsigned* p, size_t n) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)(i * 2654435761u);
}

__global__ void check(const unsigned* p, size_t n, unsigned long long* bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != (unsigned)(i * 2654435761u)) atomicAdd(bad, 1ull);
}

static double pct(std::vector<double> v, double q) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0 : v[std::min(v.size() - 1, (size_t)(q * v.size()))];
}

int main(int argc, char** argv) {
    const int chunks = argc > 1 ? std::atoi(argv[1]) : 0;
    const size_t chunk = (argc > 2 ? std::atoll(argv[2]) : 32) << 20;
    CK(hipSetDevice(0));
    hipMemAllocationProp prop = {};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = 0;
    size_t gmin = 0, grec = 0;
    CK(hipMemGetAllocationGranularity(&gmin, &prop, hipMemAllocationGranularityMinimum));
    CK(hipMemGetAllocationGranularity(&grec, &prop, hipMemAllocationGranularityRecommended));
    std::printf("granularity minimum %zu, recommended %zu bytes\n", gmin, grec);
    if (chunks == 0) {
        std::printf("no chunks (ioctl baseline)\n");
        return 0;
    }
    const long long total = (long long)chunks * (long long)chunk,
                    freeb = sysfs_bytes("mem_info_vram_total") - sysfs_bytes("mem_info_vram_used");
    if (freeb < total + (3LL << 30)) {
        std::printf("REFUSED: %.2f GB free, need %.2f GB + 3 GB\n", freeb / 1e9, total / 1e9);
        return 2;
    }
    const long long used0 = sysfs_bytes("mem_info_vram_used");
    void* base = nullptr;
    CK(hipMemAddressReserve(&base, total, grec, nullptr, 0));
    hipMemAccessDesc acc = {};
    acc.location = prop.location;
    acc.flags = hipMemAccessFlagsProtReadWrite;
    std::vector<hipMemGenericAllocationHandle_t> h(chunks);
    std::vector<double> t_create, t_map, t_access;
    for (int i = 0; i < chunks; i++) {
        void* at = static_cast<char*>(base) + (size_t)i * chunk;
        auto t = Clock::now();
        CK(hipMemCreate(&h[i], chunk, &prop, 0));
        t_create.push_back(ms_since(t));
        t = Clock::now();
        CK(hipMemMap(at, chunk, 0, h[i], 0));
        t_map.push_back(ms_since(t));
        t = Clock::now();
        CK(hipMemSetAccess(at, chunk, &acc, 1));
        t_access.push_back(ms_since(t));
    }
    const long long used1 = sysfs_bytes("mem_info_vram_used");
    std::printf("%d chunks of %zu MB: VRAM used grew by %.3f GB (expected %.3f GB)\n", chunks, chunk >> 20,
                (used1 - used0) / 1e9, total / 1e9);
    std::printf("per chunk ms (median / p99): create %.3f / %.3f, map %.3f / %.3f, setaccess %.3f / %.3f\n",
                pct(t_create, 0.5), pct(t_create, 0.99), pct(t_map, 0.5), pct(t_map, 0.99), pct(t_access, 0.5),
                pct(t_access, 0.99));
    // One kernel over the whole range: the chunks behave as one contiguous buffer
    unsigned long long* bad = nullptr;
    CK(hipMalloc(&bad, sizeof *bad));
    CK(hipMemset(bad, 0, sizeof *bad));
    const size_t n = (size_t)total / sizeof(unsigned);
    auto t = Clock::now();
    fill<<<1024, 256>>>(static_cast<unsigned*>(base), n);
    check<<<1024, 256>>>(static_cast<unsigned*>(base), n, bad);
    CK(hipDeviceSynchronize());
    unsigned long long nbad = 1;
    CK(hipMemcpy(&nbad, bad, sizeof nbad, hipMemcpyDeviceToHost));
    std::printf("fill + check of %.2f GB across all chunk borders: %llu wrong words, %.1f ms\n", total / 1e9, nbad,
                ms_since(t));
    t = Clock::now();
    CK(hipMemUnmap(base, total));
    for (auto& x : h) CK(hipMemRelease(x));
    CK(hipMemAddressFree(base, total));
    std::printf("unmap + release of all chunks: %.1f ms; VRAM used after: %.3f GB over the start\n", ms_since(t),
                (sysfs_bytes("mem_info_vram_used") - used0) / 1e9);
    CK(hipFree(bad));
    return nbad == 0 ? 0 : 1;
}
