// Cost of giving back and taking a piece of pinned host memory (expert tiers, docs/architecture/expert-tiers.md):
// hipHostMalloc (GTT) + hipHostFree, and mmap(huge pages) + hipHostRegister + hipHostUnregister + munmap, of 32 MB
// pieces, while a second thread runs short kernels on its own stream and records the longest gap between them (a stall
// of the queues shows there). Run with HSA_USERPTR_FOR_PAGED_MEM=0 like the engine. Build: hipcc -O3
// --offload-arch=gfx1200 piece_cost.hip -o piece_cost -lpthread
#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <thread>
#include <vector>

using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

__global__ void tiny(int* p) { p[0] += 1; }

static std::atomic<bool> stop{false};
static std::vector<double> gaps;  // ms between consecutive kernel completions

static void ticker() {
    hipStream_t s;
    hipStreamCreateWithFlags(&s, hipStreamNonBlocking);
    int* d;
    hipMalloc(&d, 4);
    hipMemset(d, 0, 4);
    auto last = Clock::now();
    while (!stop) {
        hipLaunchKernelGGL(tiny, dim3(1), dim3(1), 0, s, d);
        hipStreamSynchronize(s);
        gaps.push_back(ms_since(last));
        last = Clock::now();
        std::this_thread::sleep_for(std::chrono::microseconds(200));
    }
    hipFree(d);
}

static void report(const char* what, std::vector<double>& t) {
    std::sort(t.begin(), t.end());
    std::printf("%-34s n=%zu median %.3f ms  p90 %.3f  max %.3f\n", what, t.size(), t[t.size() / 2],
                t[t.size() * 9 / 10], t.back());
}

int main(int argc, char** argv) {
    const size_t piece = (argc > 1 ? std::atoi(argv[1]) : 32) << 20;
    const int reps = 40;
    (void)hipFree(nullptr);
    std::thread th(ticker);
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    const size_t base_n = gaps.size();
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    std::vector<double> idle(gaps.begin() + base_n, gaps.end());
    report("kernel gap, idle (reference)", idle);

    std::vector<double> ta, tf, tt;
    const size_t g0 = gaps.size();
    for (int i = 0; i < reps; i++) {
        void* p;
        auto t = Clock::now();
        if (hipHostMalloc(&p, piece, hipHostMallocMapped | hipHostMallocNonCoherent) != hipSuccess) return 1;
        ta.push_back(ms_since(t));
        t = Clock::now();
        std::memset(p, i, piece);  // reading the file into it costs this much memory traffic at least
        tt.push_back(ms_since(t));
        t = Clock::now();
        (void)hipHostFree(p);
        tf.push_back(ms_since(t));
    }
    std::vector<double> gg(gaps.begin() + g0, gaps.end());
    report("GTT hipHostMalloc 32 MB", ta);
    report("GTT fill (memset) 32 MB", tt);
    report("GTT hipHostFree 32 MB", tf);
    report("kernel gap during GTT cycles", gg);

    std::vector<double> ra, rf, rt;
    const size_t g1 = gaps.size();
    const size_t huge = 2u << 20;
    for (int i = 0; i < reps; i++) {
        auto t = Clock::now();
        const size_t map_len = piece + huge;
        void* map = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        void* p = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(map) + huge - 1) & ~(uintptr_t)(huge - 1));
        madvise(p, piece, MADV_HUGEPAGE);
        if (hipHostRegister(p, piece, hipHostRegisterMapped) != hipSuccess) return 2;
        ra.push_back(ms_since(t));
        t = Clock::now();
        std::memset(p, i, piece);
        rt.push_back(ms_since(t));
        t = Clock::now();
        (void)hipHostUnregister(p);
        munmap(map, map_len);
        rf.push_back(ms_since(t));
    }
    std::vector<double> gr(gaps.begin() + g1, gaps.end());
    report("registered mmap+register 32 MB", ra);
    report("registered fill (memset) 32 MB", rt);
    report("registered unregister+munmap", rf);
    report("kernel gap during reg. cycles", gr);
    stop = true;
    th.join();
}
