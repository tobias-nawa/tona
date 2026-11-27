// Prefetch into the Infinity Cache: read 10 MB cold (after 256 MB of other data), warm (twice), after s_prefetch_data
// and after a read kernel used as prefetch. Build: hipcc -O3 --offload-arch=gfx1200 prefetch.hip -o prefetch
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <vector>

__global__ void rd(const uint4* p, size_t n, unsigned* out) {
    unsigned s = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        s ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (s == 0x12345678u) out[0] = s;
}
// one 4 KB span per wave and step (s_prefetch_data: length 31 = 32 x 128 bytes)
__global__ void pf(const uint8_t* p, size_t bytes) {
    const size_t wave = (blockIdx.x * (size_t)blockDim.x + threadIdx.x) / 32,
                 waves = (size_t)gridDim.x * blockDim.x / 32;
    for (size_t o = wave * 4096; o < bytes; o += waves * 4096) __builtin_amdgcn_s_prefetch_data(p + o, 31);
}

int main() {
    const size_t B = 10u << 20, F = 256u << 20;
    uint8_t *a, *f;
    unsigned* out;
    hipMalloc(&a, B), hipMalloc(&f, F), hipMalloc(&out, 4);
    hipMemset(a, 1, B), hipMemset(f, 2, F);
    hipEvent_t e0, e1;
    hipEventCreate(&e0), hipEventCreate(&e1);
    auto t = [&](auto k) {
        hipEventRecord(e0);
        k();
        hipEventRecord(e1);
        hipEventSynchronize(e1);
        float ms;
        hipEventElapsedTime(&ms, e0, e1);
        return ms * 1000;
    };
    auto read = [&] { rd<<<64, 512>>>(reinterpret_cast<const uint4*>(a), B / 16, out); };
    auto flush = [&] { rd<<<64, 512>>>(reinterpret_cast<const uint4*>(f), F / 16, out); };
    std::vector<double> cold, warm, pre, pre_t, ld;
    for (int r = 0; r < 15; r++) {
        flush(), hipDeviceSynchronize();
        cold.push_back(t(read));
        warm.push_back(t(read));
        flush(), hipDeviceSynchronize();
        pre_t.push_back(t([&] { pf<<<64, 512>>>(a, B); }));
        pre.push_back(t(read));
        flush(), hipDeviceSynchronize();
        t(read);
        ld.push_back(t(read));
    }
    auto med = [](std::vector<double> v) {
        std::sort(v.begin(), v.end());
        return v[v.size() / 2];
    };
    printf("read 10 MB: cold %.1f us (%.0f GB/s), warm %.1f µs (%.0f GB/s)\n", med(cold), B / med(cold) / 1e3,
           med(warm), B / med(warm) / 1e3);
    printf("s_prefetch_data: prefetch %.1f us, then read %.1f us (%.0f GB/s)\n", med(pre_t), med(pre),
           B / med(pre) / 1e3);
    printf("read kernel as prefetch: then %.1f us\n", med(ld));
}
