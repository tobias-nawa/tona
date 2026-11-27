// Probe: volatile MTLBuffers under memory pressure. N blocks of 32 MB, the first half in a committed residency set,
// the second half not; all volatile. Prints the cost of the state calls, then every 0.5 s for T s how many blocks of
// each half are empty and the VM counters (purgeable, purgeable wired, compressor, purges, compressions).
// (research/measurements/2026-11-16-mac-purgeable-probe.md). Usage: metal_purge_probe [BLOCKS [SECONDS [out]]]
// Build: clang++ -std=c++20 -O2 -fobjc-arc -framework Metal -framework Foundation metal_purge_probe.mm -o probe
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <mach/mach.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

static double now_s() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static vm_statistics64_data_t vm() {
    vm_statistics64_data_t v{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&v, &n);
    return v;
}

static void cost(id<MTLBuffer> t, id<MTLResidencySet> set) {
    // Cost of the calls, on the last block (not in the set)
    const int reps = 200;
    double t0 = now_s();
    for (int r = 0; r < reps; r++) {
        [t setPurgeableState:MTLPurgeableStateVolatile];
        [t setPurgeableState:MTLPurgeableStateNonVolatile];
    }
    const double toggle = (now_s() - t0) / (2 * reps) * 1e6;
    t0 = now_s();
    for (int r = 0; r < reps; r++) [t setPurgeableState:MTLPurgeableStateKeepCurrent];
    const double query = (now_s() - t0) / reps * 1e6;
    t0 = now_s();
    for (int r = 0; r < reps; r++) {
        [set addAllocation:t];
        [set commit];
        [set removeAllocation:t];
        [set commit];
    }
    const double rset = (now_s() - t0) / (2 * reps) * 1e6;
    std::printf("calls: volatile/nonvolatile %.1f us, query %.1f us, residency add/remove + commit %.1f us\n", toggle,
                query, rset);
}

static void watch(const std::vector<id<MTLBuffer>>& b, int half, double secs, double gb, size_t bytes) {
    const int nb = (int)b.size();
    for (int i = 0; i < nb; i++) [b[i] setPurgeableState:MTLPurgeableStateVolatile];
    const auto v0 = vm();
    std::printf("blocks %d (%.2f GB), in the set %d; volatile now: purgeable %.2f GB, purgeable wired %.2f GB\n", nb,
                nb * (double)bytes / 1e9, half, v0.purgeable_count * gb, v0.purgeable_wired_count * gb);
    const double start = now_s();
    int last_in = -1, last_out = -1;
    while (now_s() - start < secs) {
        int in = 0, out = 0;
        for (int i = 0; i < nb; i++)
            if ([b[i] setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateEmpty)
                (i < half ? in : out)++;
        const auto v = vm();
        if (in != last_in || out != last_out || (int)((now_s() - start) * 2) % 10 == 0)
            std::printf("%6.1f s  empty: in set %d/%d, out %d/%d; free %.2f, purgeable %.2f (wired %.2f), "
                        "compressor %.2f GB; purges +%llu, compressions +%.2f GB\n",
                        now_s() - start, in, half, out, nb - half, v.free_count * gb, v.purgeable_count * gb,
                        v.purgeable_wired_count * gb, v.compressor_page_count * gb,
                        (unsigned long long)(v.purges - v0.purges), (v.compressions - v0.compressions) * gb);
        std::fflush(stdout);
        last_in = in, last_out = out;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
}

static void verify(const std::vector<id<MTLBuffer>>& b) {
    const int nb = (int)b.size();
    // Contents after nonvolatile: an empty block reads zero, a kept one its pattern
    int kept = 0, zero = 0, other = 0;
    for (int i = 0; i < nb; i++) {
        const MTLPurgeableState was = [b[i] setPurgeableState:MTLPurgeableStateNonVolatile];
        const auto* p = (const uint8_t*)b[i].contents;
        const bool ok = p[0] == (uint8_t)(i + 1) && p[1] == 0x5a;
        if (was == MTLPurgeableStateEmpty)
            (p[0] == 0 ? zero : other)++;
        else if (ok)
            kept++;
        else
            other++;
    }
    std::printf("end: kept with contents %d, empty and zero %d, inconsistent %d\n", kept, zero, other);
}

int main(int argc, char** argv) {
    const int nb = argc > 1 ? atoi(argv[1]) : 64;
    const double secs = argc > 2 ? atof(argv[2]) : 60;
    const bool all_out = argc > 3 && strcmp(argv[3], "out") == 0;  // no block in the residency set
    const size_t bytes = 32u << 20, page = (size_t)getpagesize();
    const double gb = page / 1e9;
    @autoreleasepool {
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        std::vector<id<MTLBuffer>> b(nb);
        for (int i = 0; i < nb; i++) {
            b[i] = [dev newBufferWithLength:bytes options:MTLResourceStorageModeShared];
            auto* p = (uint8_t*)b[i].contents;
            for (size_t o = 0; o < bytes; o += page) p[o] = (uint8_t)(i + 1), p[o + 1] = 0x5a;  // touched, resident
        }
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.initialCapacity = nb;
        id<MTLResidencySet> set = [dev newResidencySetWithDescriptor:d error:nil];
        const int half = all_out ? 0 : nb / 2;
        for (int i = 0; i < half; i++) [set addAllocation:b[i]];
        [set commit];
        [set requestResidency];

        cost(b[nb - 1], set);
        watch(b, half, secs, gb, bytes);
        verify(b);
    }
    return 0;
}
