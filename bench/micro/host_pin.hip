// What does pinned host memory allow? For docs/architecture/expert-tiers.md section 5b: (a) is a hipHostRegister of a
// file-backed mmap range accepted, and do its pages then leave the reclaimable set (MADV_PAGEOUT, mincore,
// /proc/meminfo); (b) what do register and unregister of a 32 MB piece cost; (c) can a hipHostMalloc piece be freed on
// its own, and do MemAvailable and the GTT use come back. Host memory only: no kernel, no VRAM allocation. Run once
// with HSA_USERPTR_FOR_PAGED_MEM=0 (the engine's setting) and once unset. Build: hipcc -O2 --offload-arch=gfx1200
// -std=c++20 host_pin.hip -o host_pin
#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define CK(x)                                                                                             \
    do {                                                                                                  \
        const hipError_t e_ = (x);                                                                        \
        if (e_ != hipSuccess) std::printf("  %s -> %s (line %d)\n", #x, hipGetErrorString(e_), __LINE__); \
    } while (0)

constexpr size_t PIECE = 32ull << 20;
using Clock = std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

static long meminfo_kb(const char* key) {
    FILE* f = std::fopen("/proc/meminfo", "r");
    char line[256];
    long v = -1;
    while (std::fgets(line, sizeof line, f))
        if (std::strncmp(line, key, std::strlen(key)) == 0) {
            std::sscanf(line + std::strlen(key), ": %ld", &v);
            break;
        }
    std::fclose(f);
    return v;
}
static long gtt_used_mb() {
    for (int c = 0; c < 8; c++) {
        std::string b = "/sys/class/drm/card" + std::to_string(c) + "/device/";
        FILE* t = std::fopen((b + "mem_info_vram_total").c_str(), "r");
        if (!t) continue;
        long long tot = 0;
        std::fscanf(t, "%lld", &tot), std::fclose(t);
        if (tot < (4ll << 30)) continue;
        FILE* f = std::fopen((b + "mem_info_gtt_used").c_str(), "r");
        long long u = 0;
        if (f) std::fscanf(f, "%lld", &u), std::fclose(f);
        return (long)(u >> 20);
    }
    return -1;
}
static size_t resident_pages(void* p, size_t n) {
    std::vector<unsigned char> v(n / 4096);
    if (mincore(p, n, v.data()) != 0) return 0;
    size_t r = 0;
    for (unsigned char c : v) r += c & 1;
    return r;
}

static void part_c() {  // pieces by hipHostMalloc, freed one by one
    std::printf("(c) hipHostMalloc pieces of 32 MB, free one\n");
    std::vector<void*> p(8);
    const long a0 = meminfo_kb("MemAvailable"), g0 = gtt_used_mb();
    auto t = Clock::now();
    for (auto& q : p) {
        CK(hipHostMalloc(&q, PIECE, hipHostMallocMapped | hipHostMallocNonCoherent));
        std::memset(q, 1, PIECE);
    }
    std::printf("  8 pieces allocated and touched in %.1f ms; MemAvailable %+ld MB, GTT %+ld MB\n", ms(t),
                (meminfo_kb("MemAvailable") - a0) / 1024, gtt_used_mb() - g0);
    const long a1 = meminfo_kb("MemAvailable"), g1 = gtt_used_mb();
    t = Clock::now();
    CK(hipHostFree(p[3]));
    std::printf("  one piece freed in %.2f ms; MemAvailable %+ld MB, GTT %+ld MB\n", ms(t),
                (meminfo_kb("MemAvailable") - a1) / 1024, gtt_used_mb() - g1);
    for (int i = 0; i < 8; i++)
        if (i != 3) CK(hipHostFree(p[i]));
}

static void part_b() {  // register / unregister anonymous memory
    std::printf("(b) hipHostRegister of anonymous 32 MB\n");
    void* m = mmap(nullptr, PIECE, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    std::memset(m, 1, PIECE);
    const long g0 = gtt_used_mb(), u0 = meminfo_kb("Unevictable"), l0 = meminfo_kb("Mlocked");
    auto t = Clock::now();
    CK(hipHostRegister(m, PIECE, hipHostRegisterMapped));
    const double reg = ms(t);
    std::printf("  register %.2f ms, GTT %+ld MB, Unevictable %+ld MB, Mlocked %+ld MB\n", reg, gtt_used_mb() - g0,
                (meminfo_kb("Unevictable") - u0) / 1024, (meminfo_kb("Mlocked") - l0) / 1024);
    t = Clock::now();
    CK(hipHostUnregister(m));
    std::printf("  unregister %.2f ms\n", ms(t));
    munmap(m, PIECE);
}

static void part_a() {  // file-backed
    std::printf("(a) hipHostRegister of a file-backed mapping (64 MB file, clean pages)\n");
    const char* path = "/var/tmp/claude/148/host_pin.dat";
    int fd = open(path, O_RDWR | O_CREAT | O_TRUNC, 0600);
    std::vector<char> buf(PIECE, 7);
    for (int i = 0; i < 2; i++) (void)!write(fd, buf.data(), PIECE);
    fsync(fd);
    const size_t n = 2 * PIECE;
    for (int pass = 0; pass < 2; pass++) {
        char* m = (char*)mmap(nullptr, n, PROT_READ, MAP_SHARED, fd, 0);
        volatile char s = 0;
        for (size_t o = 0; o < n; o += 4096) s += m[o];
        (void)s;
        const long u0 = meminfo_kb("Unevictable"), l0 = meminfo_kb("Mlocked");
        hipError_t e = pass ? hipSuccess : hipHostRegister(m, n, hipHostRegisterMapped);
        std::printf("  %s: %s", pass ? "control (not registered)" : "registered", hipGetErrorString(e));
        std::printf("; Unevictable %+ld MB, Mlocked %+ld MB\n", (meminfo_kb("Unevictable") - u0) / 1024,
                    (meminfo_kb("Mlocked") - l0) / 1024);
        const size_t before = resident_pages(m, n);
        const int r = madvise(m, n, MADV_PAGEOUT);
        std::printf("  resident pages %zu -> after MADV_PAGEOUT (rc %d) %zu\n", before, r, resident_pages(m, n));
        if (!pass && e == hipSuccess) CK(hipHostUnregister(m));
        munmap(m, n);
    }
    close(fd);
    unlink(path);
}

int main() {
    const char* v = std::getenv("HSA_USERPTR_FOR_PAGED_MEM");
    std::printf("HSA_USERPTR_FOR_PAGED_MEM=%s\n", v ? v : "unset");
    CK(hipSetDevice(0));
    part_c();
    part_b();
    part_a();
    return 0;
}
