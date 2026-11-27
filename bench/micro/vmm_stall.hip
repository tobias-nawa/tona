// What does hipMemUnmap cost when the GPU is idle and when a persistent kernel is running, does one call over many
// chunks cost less than one call per chunk, how long does a kernel that reads the kept chunks stall while others are
// unmapped, and does remapping the same virtual address with a new handle corrupt data (ROCm/TheRock #8744)? Needed for
// the "give VRAM back at run time" design in docs/architecture/expert-tiers.md (section 6). Standalone, 1 GiB of VRAM,
// no server. The spinning kernel stops by itself after 60 ms (own clock), it never waits for the CPU, so a call that
// blocks until the device is idle cannot hang the GPU. Build: hipcc -O2 --offload-arch=gfx1200 -std=c++20 vmm_stall.hip
// -o vmm_stall
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

constexpr size_t kChunks = 32, kChunk = 32u << 20, kWords = kChunk / 4;

struct Pool {
    char* base = nullptr;
    std::vector<hipMemGenericAllocationHandle_t> h = std::vector<hipMemGenericAllocationHandle_t>(kChunks);
    hipMemAllocationProp prop = {};
    hipMemAccessDesc acc = {};
};

__global__ void fill(unsigned* p, size_t n, unsigned gen) {
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        p[i] = (unsigned)i * 2654435761u ^ gen;
}
__global__ void check(const unsigned* p, size_t n, unsigned gen, unsigned long long* bad) {
    unsigned long long b = 0;
    for (size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        b += p[i] != ((unsigned)i * 2654435761u ^ gen);
    if (b) atomicAdd(bad, b);
}

// Persistent stand-in: every block reads one word per kept chunk per iteration and verifies it, thread 0 of block 0
// keeps the longest gap between two iterations (wall clock ticks) and the number of gaps above 0.5 ms; ends after
// run_ticks
__global__ void spin(const char* base, size_t chunks, unsigned long long run_ticks, unsigned long long gap_ticks,
                     unsigned long long* out) {
    const unsigned long long t0 = wall_clock64();
    unsigned long long last = t0, max_gap = 0, n_gaps = 0, iters = 0, bad = 0;
    while (true) {
        const unsigned long long now = wall_clock64();
        if (now - t0 > run_ticks) break;
        for (size_t c = 0; c < chunks; c++) {
            const size_t i = (iters * 4099 + threadIdx.x * 31) % kWords;
            const unsigned v = reinterpret_cast<const volatile unsigned*>(base + c * kChunk)[i];
            bad += v != ((unsigned)i * 2654435761u ^ 1u);
        }
        if (now - last > max_gap) max_gap = now - last;
        if (now - last > gap_ticks) n_gaps++;
        last = now;
        iters++;
    }
    if (threadIdx.x == 0) {
        atomicMax(&out[0], max_gap);
        atomicAdd(&out[1], n_gaps);
        atomicAdd(&out[2], iters);
    }
    if (bad) atomicAdd(&out[3], bad);
}

static double mean(const std::vector<double>& v) {
    double s = 0;
    for (double x : v) s += x;
    return v.empty() ? 0 : s / v.size();
}
static double maxof(const std::vector<double>& v) { return v.empty() ? 0 : *std::max_element(v.begin(), v.end()); }

static void make_chunk(Pool& P, size_t i) {
    CK(hipMemCreate(&P.h[i], kChunk, &P.prop, 0));
    CK(hipMemMap(P.base + i * kChunk, kChunk, 0, P.h[i], 0));
    CK(hipMemSetAccess(P.base + i * kChunk, kChunk, &P.acc, 1));
}
static void drop_chunk(Pool& P, size_t i, std::vector<double>* t_unmap) {
    auto t = Clock::now();
    CK(hipMemUnmap(P.base + i * kChunk, kChunk));
    if (t_unmap) t_unmap->push_back(ms_since(t));
    CK(hipMemRelease(P.h[i]));
}
static void fill_all(Pool& P, size_t from, size_t to, unsigned gen) {
    for (size_t i = from; i < to; i++)  // the index in the pattern counts from the start of the chunk
        fill<<<64, 256>>>(reinterpret_cast<unsigned*>(P.base + i * kChunk), kWords, gen);
    CK(hipDeviceSynchronize());
}

// E1: unmap one chunk per call, GPU idle; E1b: one call over the whole upper half
static void e1_idle(Pool& P) {
    std::vector<double> one;
    for (size_t i = kChunks / 2; i < kChunks; i++) drop_chunk(P, i, &one);
    std::printf("E1  idle, one unmap per chunk (16 x 32 MiB): mean %.3f ms, max %.3f ms, total %.2f ms\n", mean(one),
                maxof(one), mean(one) * one.size());
    for (size_t i = kChunks / 2; i < kChunks; i++) make_chunk(P, i);
    fill_all(P, kChunks / 2, kChunks, 1u);
    auto t = Clock::now();
    const hipError_t e = hipMemUnmap(P.base + kChunks / 2 * kChunk, kChunk * kChunks / 2);
    const double ms = ms_since(t);
    std::printf("E1b idle, one unmap over 16 chunks mapped by 16 calls: %s, %.3f ms\n", hipGetErrorString(e), ms);
    if (e == hipSuccess) {
        for (size_t i = kChunks / 2; i < kChunks; i++) CK(hipMemRelease(P.h[i]));
    } else {
        for (size_t i = kChunks / 2; i < kChunks; i++) drop_chunk(P, i, nullptr);
    }
    for (size_t i = kChunks / 2; i < kChunks; i++) make_chunk(P, i);
    fill_all(P, kChunks / 2, kChunks, 1u);
}

// E2: the spinning kernel reads the kept lower half; unmaps of the upper half start after 10 ms (or none: baseline)
static void e2_spin(Pool& P, bool unmap, unsigned long long* out, double ticks_per_ms) {
    CK(hipMemset(out, 0, 32));
    hipStream_t s;
    CK(hipStreamCreateWithFlags(&s, hipStreamNonBlocking));
    spin<<<4, 64, 0, s>>>(P.base, kChunks / 2, (unsigned long long)(60 * ticks_per_ms),
                          (unsigned long long)(0.5 * ticks_per_ms), out);
    std::vector<double> t_un;
    if (unmap) {
        auto t0 = Clock::now();
        while (ms_since(t0) < 10.0) {
        }
        for (size_t i = kChunks / 2; i < kChunks; i++) drop_chunk(P, i, &t_un);
    }
    CK(hipStreamSynchronize(s));
    unsigned long long h[4];
    CK(hipMemcpy(h, out, 32, hipMemcpyDeviceToHost));
    std::printf("E2  %-9s kernel: longest gap %.3f ms, gaps >0.5 ms: %llu, iterations %llu, wrong reads %llu",
                unmap ? "with unmap" : "baseline", h[0] / ticks_per_ms, h[1], h[2], h[3]);
    if (unmap) std::printf("; unmap per chunk mean %.3f ms, max %.3f ms", mean(t_un), maxof(t_un));
    std::printf("\n");
    CK(hipStreamDestroy(s));
    if (unmap) {
        for (size_t i = kChunks / 2; i < kChunks; i++) make_chunk(P, i);
        fill_all(P, kChunks / 2, kChunks, 1u);
    }
}

// Writes a generation into a chunk at `at`, reads it back and returns whether it is intact
static bool write_read(char* at, unsigned long long* bad, unsigned gen) {
    fill<<<64, 256>>>(reinterpret_cast<unsigned*>(at), kWords, gen);
    CK(hipDeviceSynchronize());
    CK(hipMemset(bad, 0, 8));
    check<<<64, 256>>>(reinterpret_cast<const unsigned*>(at), kWords, gen, bad);
    CK(hipDeviceSynchronize());
    unsigned long long b = 0;
    CK(hipMemcpy(&b, bad, 8, hipMemcpyDeviceToHost));
    return b == 0;
}

// E3 (variant 0): release one chunk, create a new handle, map at the same address, write a new generation, read back.
// Same shape as ROCm/TheRock #8744 (stale translation after unmap at the same address). Variants that are safe for
// other programs (a stale translation writes into pages that were given back and may belong to someone else): 1 = a
// fresh address every time (never reused), 2 = the same handle mapped again at the same address
static void e3_remap(Pool& P, unsigned long long* bad, int iters, int variant) {
    static const char* const kName[] = {"new handle, same address", "new handle, fresh address",
                                        "same handle, same address"};
    long long wrong = 0, reused = 0;
    std::vector<char*> seen;
    for (int it = 0; it < iters; it++) {
        const size_t i = kChunks / 2 + it % (kChunks / 2);
        char* at = P.base + i * kChunk;
        if (variant == 2) {
            CK(hipMemUnmap(at, kChunk));
            CK(hipMemMap(at, kChunk, 0, P.h[i], 0));
            CK(hipMemSetAccess(at, kChunk, &P.acc, 1));
        } else if (variant == 1) {
            if (it < (int)(kChunks / 2)) drop_chunk(P, i, nullptr);  // the old address is never mapped again
            void* fresh = nullptr;  // never freed, so the runtime cannot hand the same address out again
            CK(hipMemAddressReserve(&fresh, kChunk, 0, nullptr, 0));
            at = static_cast<char*>(fresh);
            reused += std::find(seen.begin(), seen.end(), at) != seen.end();
            seen.push_back(at);
            hipMemGenericAllocationHandle_t h;
            CK(hipMemCreate(&h, kChunk, &P.prop, 0));
            CK(hipMemMap(at, kChunk, 0, h, 0));
            CK(hipMemSetAccess(at, kChunk, &P.acc, 1));
            wrong += !write_read(at, bad, 100u + it);
            CK(hipMemUnmap(at, kChunk));
            CK(hipMemRelease(h));
            continue;
        } else {
            drop_chunk(P, i, nullptr);
            make_chunk(P, i);
        }
        wrong += !write_read(at, bad, 100u + it);
    }
    std::printf("E3  remap, %s, %d iterations: wrong in %lld%s\n", kName[variant], iters, wrong,
                variant == 1 ? (reused ? " (an address came back!)" : ", all addresses distinct") : "");
}

// E4: how much address space can be reserved (a pool with many fresh ranges needs it) and does a chunk mapped far out
// work
static void e4_big_va(Pool& P, unsigned long long* bad) {
    for (size_t gib : {64u, 512u, 4096u}) {
        void* va = nullptr;
        auto t = Clock::now();
        const hipError_t e = hipMemAddressReserve(&va, gib << 30, 0, nullptr, 0);
        const double ms = ms_since(t);
        std::printf("E4  reserve %zu GiB of address space: %s, %.3f ms", gib, hipGetErrorString(e), ms);
        if (e != hipSuccess) {
            std::printf("\n");
            continue;
        }
        char* far_end = static_cast<char*>(va) + (gib << 30) - kChunk;
        hipMemGenericAllocationHandle_t h;
        CK(hipMemCreate(&h, kChunk, &P.prop, 0));
        CK(hipMemMap(far_end, kChunk, 0, h, 0));
        CK(hipMemSetAccess(far_end, kChunk, &P.acc, 1));
        std::printf(", chunk at the far end %s\n", write_read(far_end, bad, 7u) ? "intact" : "WRONG");
        CK(hipMemUnmap(far_end, kChunk));  // never mapped again at this address
        CK(hipMemRelease(h));
        CK(hipMemAddressFree(va, gib << 30));
    }
}

int main(int argc, char** argv) {
    int dev = 0, rate_khz = 0;
    CK(hipSetDevice(dev));
    CK(hipDeviceGetAttribute(&rate_khz, hipDeviceAttributeWallClockRate, dev));
    Pool P;
    P.prop.type = hipMemAllocationTypePinned;
    P.prop.location.type = hipMemLocationTypeDevice;
    P.prop.location.id = dev;
    P.acc.location = P.prop.location;
    P.acc.flags = hipMemAccessFlagsProtReadWrite;
    CK(hipMemAddressReserve(reinterpret_cast<void**>(&P.base), kChunks * kChunk, 0, nullptr, 0));
    for (size_t i = 0; i < kChunks; i++) make_chunk(P, i);
    fill_all(P, 0, kChunks, 1u);
    unsigned long long* out = nullptr;
    CK(hipMalloc(&out, 32));
    const double ticks_per_ms = rate_khz;  // wall_clock64 runs at the rate in kHz, i.e. ticks per ms
    std::printf("wall clock %d kHz\n", rate_khz);
    bool unsafe = false;
    for (int a = 1; a < argc; a++) {
        if (std::strcmp(argv[a], "--unsafe-same-address") == 0) unsafe = true;
    }
    if (unsafe) {  // see SAFETY at the top: E1 and E2 refill remapped chunks at the same address
        e1_idle(P);
        e2_spin(P, false, out, ticks_per_ms);
        e2_spin(P, true, out, ticks_per_ms);
        e2_spin(P, false, out, ticks_per_ms);
        e2_spin(P, true, out, ticks_per_ms);
    }
    if (unsafe) e3_remap(P, out, 400, 0);
    if (!unsafe) e3_remap(P, out, 400, 2);  // same handle: the pool stays intact
    if (!unsafe) e3_remap(P, out, 400, 1);  // fresh addresses: the upper half of the pool is gone afterwards

    e4_big_va(P, out);
    std::printf("done\n");
    return 0;
}
