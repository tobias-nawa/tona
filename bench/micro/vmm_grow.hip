// Does a pool that grows by mapping more handles behind the ones already mapped (never remapping an address) work on
// this stack, also at the smallest granule? Reproduces the sequence of llama.cpp #32350 (RX 9060 XT, ROCm 7.1.1: the
// first mapping is one granule, the next hipMemSetAccess on a sub-range at a nonzero offset fails with
// hipErrorInvalidValue, or a whole-range call succeeds and the next kernel faults), with the granule of that report
// (the minimum) and with the recommended one that Tona uses (mega_elastic.cpp). After every growth step a kernel writes
// a pattern over the whole mapped range and a second kernel checks it. Standalone, up to 110 MB of VRAM (at the 2 MiB
// granule). Build: hipcc -O2 --offload-arch=gfx1200 -std=c++20 vmm_grow.hip -o vmm_grow Usage: vmm_grow [min|rec]
// [sub|whole]   (granule, and the range of hipMemSetAccess after a growth step)
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstring>
#include <vector>

__global__ void fill(unsigned* p, size_t n, unsigned v) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = v + (unsigned)i;
}
__global__ void check(const unsigned* p, size_t n, unsigned v, unsigned* bad) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        if (p[i] != v + (unsigned)i) atomicAdd(bad, 1u);
}

static bool ok(hipError_t e, const char* what, int step) {
    if (e == hipSuccess) return true;
    std::printf("step %d: %s -> %s\n", step, what, hipGetErrorString(e));
    return false;
}

struct Pool {
    void* base = nullptr;
    size_t gran = 0, total = 128u << 20, mapped = 0;
    bool whole = false;
    hipMemAllocationProp prop = {};
    hipMemAccessDesc acc = {};
    std::vector<hipMemGenericAllocationHandle_t> handles;
    unsigned* bad = nullptr;
};

// Writes a pattern over the mapped range and checks it with a second kernel
static bool touch(const Pool& P, int step) {
    unsigned zero = 0, n_bad = 0;
    (void)hipMemcpy(P.bad, &zero, 4, hipMemcpyHostToDevice);
    const size_t words = P.mapped / 4;
    fill<<<64, 256>>>(static_cast<unsigned*>(P.base), words, 17u * step);
    check<<<64, 256>>>(static_cast<unsigned*>(P.base), words, 17u * step, P.bad);
    const bool k = ok(hipDeviceSynchronize(), "kernel over the mapped range", step);
    (void)hipMemcpy(&n_bad, P.bad, 4, hipMemcpyDeviceToHost);
    if (n_bad) std::printf("step %d: %u words wrong\n", step, n_bad);
    return k && n_bad == 0;
}

// One growth step: a new handle behind the mapped range, access, then the kernel over everything mapped
static bool grow(Pool& P, size_t n, int step) {
    const size_t len = n * P.gran;
    if (P.mapped + len > P.total) return false;
    hipMemGenericAllocationHandle_t h = {};
    char* at = static_cast<char*>(P.base) + P.mapped;
    if (!ok(hipMemCreate(&h, len, &P.prop, 0), "create", step)) return false;
    if (!ok(hipMemMap(at, len, 0, h, 0), "map", step)) return false;
    P.handles.push_back(h);
    const hipError_t e =
        P.whole ? hipMemSetAccess(P.base, P.mapped + len, &P.acc, 1) : hipMemSetAccess(at, len, &P.acc, 1);
    if (!ok(e, P.whole ? "set access (whole range)" : "set access (new sub-range)", step)) return false;
    P.mapped += len;
    if (!touch(P, step)) return false;
    std::printf("step %d: %zu bytes added at offset %zu, mapped %zu, kernel ok\n", step, len, P.mapped - len, P.mapped);
    return true;
}

int main(int argc, char** argv) {
    const bool minimum = argc > 1 && !std::strcmp(argv[1], "min");
    Pool P;
    P.whole = argc > 2 && !std::strcmp(argv[2], "whole");
    P.prop.type = hipMemAllocationTypePinned;
    P.prop.location.type = hipMemLocationTypeDevice;
    P.prop.location.id = 0;
    const auto kind = minimum ? hipMemAllocationGranularityMinimum : hipMemAllocationGranularityRecommended;
    if (!ok(hipMemGetAllocationGranularity(&P.gran, &P.prop, kind), "granularity", 0)) return 1;
    P.acc.location = P.prop.location;
    P.acc.flags = hipMemAccessFlagsProtReadWrite;
    if (!ok(hipMemAddressReserve(&P.base, P.total, P.gran, nullptr, 0), "reserve", 0)) return 1;
    if (!ok(hipMalloc(&P.bad, 4), "malloc", 0)) return 1;
    // 1 granule first, then 11, 3, 5 ...: the pattern of the ADD cases of the report (4 KiB, then 44 KiB, ...)
    const size_t steps[] = {1, 11, 3, 5, 1, 7, 2, 13, 4, 9};
    int step = 0;
    bool all = true;
    for (size_t n : steps) all = all && grow(P, n, ++step);
    std::printf("granule %zu (%s), set access on %s: %s\n", P.gran, minimum ? "minimum" : "recommended",
                P.whole ? "whole range" : "new sub-range", all ? "all steps ok" : "FAILED");
    (void)hipDeviceSynchronize();
    (void)hipMemUnmap(P.base, P.mapped);
    for (auto h : P.handles) (void)hipMemRelease(h);
    (void)hipMemAddressFree(P.base, P.total);
    (void)hipFree(P.bad);
    return all ? 0 : 1;
}
