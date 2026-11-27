// A second process that takes VRAM step by step, holds it and gives it back, to see what another GPU program does to
// the server (docs/architecture/expert-tiers.md, VRAM resilience). Prints one line per event with the sums of sysfs
// (VRAM used and total of the card, in MiB); no memset and no kernel, so it adds no GPU work of its own.
// Usage: vram_balloon STEP_MB MAX_STEPS STEP_PAUSE_S HOLD_S   Build: hipcc -O2 --offload-arch=gfx1200 vram_balloon.hip
// -o vram_balloon
#include <hip/hip_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <thread>
#include <vector>

static long long sysfs(const char* name) {
    for (int card = 0; card < 8; card++) {
        char path[128];
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/%s", card, name);
        std::ifstream in(path);
        long long total = 0, v = -1;
        if (!(in >> v)) continue;
        std::snprintf(path, sizeof path, "/sys/class/drm/card%d/device/mem_info_vram_total", card);
        std::ifstream t(path);
        t >> total;
        if (total > (4LL << 30)) return v;
    }
    return -1;
}

static double now() {
    using namespace std::chrono;
    return duration<double>(steady_clock::now().time_since_epoch()).count();
}

int main(int argc, char** argv) {
    const size_t step = (size_t)(argc > 1 ? std::atoi(argv[1]) : 512) << 20;
    const int max_steps = argc > 2 ? std::atoi(argv[2]) : 16;
    const double pause = argc > 3 ? std::atof(argv[3]) : 10, hold = argc > 4 ? std::atof(argv[4]) : 60;
    const double t0 = now();
    auto log = [&](const char* what, int n, const char* extra) {
        std::printf("%7.1f s  %-8s step %2d  held %5zu MiB  card used %6lld MiB  %s\n", now() - t0, what, n,
                    n * (step >> 20), sysfs("mem_info_vram_used") >> 20, extra);
        std::fflush(stdout);
    };
    (void)hipFree(nullptr);
    std::vector<void*> held;
    log("start", 0, "");
    for (int i = 1; i <= max_steps; i++) {
        void* p = nullptr;
        const hipError_t e = hipMalloc(&p, step);
        if (e != hipSuccess) {
            log("FAILED", i - 1, hipGetErrorString(e));
            break;
        }
        held.push_back(p);
        log("alloc", i, "");
        std::this_thread::sleep_for(std::chrono::duration<double>(pause));
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(hold));
    log("release", (int)held.size(), "");
    for (void* p : held) (void)hipFree(p);
    held.clear();
    std::this_thread::sleep_for(std::chrono::seconds(5));
    log("end", 0, "");
}
