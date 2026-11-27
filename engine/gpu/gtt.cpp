#include "gpu/gtt.h"

#include <hip/hip_runtime.h>
#include <sys/mman.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "backend/env.h"

namespace tona::gpu {

namespace {

struct Gtt {
    int64_t total = -1, used = -1;
};

// TONA_GTT_MARGIN_MB: GTT left free for the desktop's fallback placements and our later allocations (default 1024,
// the margin of cap_pin_budget; research/61-desktop-vram-reserve.md, option D)
int64_t gtt_margin() { return env::integer("TONA_GTT_MARGIN_MB", 1024) << 20; }

int64_t read_number(const std::string& path) {
    std::ifstream in(path);
    long long v = -1;
    return (in >> v) ? v : -1;
}

// sysfs of the GPU the process uses (not the iGPU): by its PCI address
Gtt read_gtt() {
    int dev = 0;
    char bus[32] = {};
    if (hipGetDevice(&dev) != hipSuccess || hipDeviceGetPCIBusId(bus, sizeof bus, dev) != hipSuccess) return {};
    std::string id = bus;
    std::transform(id.begin(), id.end(), id.begin(), [](unsigned char c) { return std::tolower(c); });
    const std::string base = "/sys/bus/pci/devices/" + id + "/";
    return {read_number(base + "mem_info_gtt_total"), read_number(base + "mem_info_gtt_used")};
}

constexpr int64_t kProbeBytes = 64LL << 20;

struct Probe {
    HostMemory kind = HostMemory::Unknown;
    int64_t grew = 0;
};

Probe run_probe() {
    const Gtt before = read_gtt();
    void* p = nullptr;
    // the flags of the expert buffers (gpu/mega_experts.cpp), so that the probe takes the same path
    if (before.used < 0 || hipHostMalloc(&p, kProbeBytes, hipHostMallocMapped | hipHostMallocNonCoherent) != hipSuccess)
        return {};
    const Gtt after = read_gtt();
    (void)hipHostFree(p);
    if (after.used < 0) return {};
    return {classify_host_memory(before.used, after.used, kProbeBytes), after.used - before.used};
}

const Probe& probe() {
    static const Probe p = run_probe();
    return p;
}

}  // namespace

HostMemory host_memory() { return probe().kind; }

void log_host_memory() {
    const Gtt g = read_gtt();
    const char* v = std::getenv("HSA_USERPTR_FOR_PAGED_MEM");
    const double mb = (double)probe().grew / 1048576.0;
    if (probe().kind == HostMemory::Gtt) {
        std::fprintf(stderr, "host memory: GTT (probe 64 MB grew the GTT use by %.0f MB; %.1f of %.1f GB in use)\n", mb,
                     (double)g.used / 1e9, (double)g.total / 1e9);
        return;
    }
    std::fprintf(stderr,
                 "host memory: userptr%s (probe 64 MB grew the GTT use by %.0f MB, HSA_USERPTR_FOR_PAGED_MEM=%s): page "
                 "migration can evict the queues of the process under memory pressure\n",
                 probe().kind == HostMemory::Unknown ? " (unconfirmed)" : "", mb, v ? v : "unset");
}

int64_t cap_pin_to_gtt(int64_t budget) {
    if (host_memory() != HostMemory::Gtt) return budget;
    const Gtt g = read_gtt();
    if (g.total < 0 || g.used < 0) return budget;
    const int64_t capped = cap_pin_budget(budget, g.total, g.used, gtt_margin());
    if (capped < budget)
        std::fprintf(stderr,
                     "host memory: pinned experts limited from %.1f to %.1f GB by the GTT (%.1f of %.1f GB in use)\n",
                     (double)budget / 1e9, (double)capped / 1e9, (double)g.used / 1e9, (double)g.total / 1e9);
    return capped;
}

int64_t gtt_room() {
    if (host_memory() != HostMemory::Gtt) return INT64_MAX;
    const Gtt g = read_gtt();
    if (g.total < 0 || g.used < 0) return INT64_MAX;
    return cap_pin_budget(INT64_MAX, g.total, g.used, gtt_margin());
}

namespace {
struct Registered {
    void* base;  // the registered buffer
    size_t len;
    void* map;  // the mmap that holds it (the buffer starts inside it, aligned to 2 MB)
    size_t map_len;
};
std::vector<Registered>& registered() {
    static std::vector<Registered> v;
    return v;
}
constexpr size_t kHugePage = 2u << 20;
}  // namespace

// Huge pages on purpose: the registration is a KFD SVM range with an MMU notifier, and every page the kernel migrates
// or reclaims in it stops all queues of the process until the range is validated again. 4 KB pages of this range are
// migrated by the compaction under page-cache pressure all the time; 2 MB pages are hardly ever (idle server, 4.1 GB
// registered, 120 s of pressure: 29.6 s of evicted time with 4 KB pages, 2.85 s with huge pages;
// research/measurements/2026-11-11-evict.md). No root and no sysctl: the mapping is aligned to 2 MB and madvised; if
// the kernel cannot give huge pages the memory is 4 KB pages as before.
void* register_host_memory(size_t bytes) {
    const size_t len = (bytes + 4095) & ~size_t(4095);
    const bool huge = env::integer("TONA_USERPTR_THP", 1) != 0 && len >= kHugePage;
    const size_t map_len = huge ? len + kHugePage : len;
    void* map = mmap(nullptr, map_len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (map == MAP_FAILED)
        throw std::runtime_error(std::format("mmap of {} bytes failed: {}", map_len, std::strerror(errno)));
    void* p = map;
    if (huge) {
        p = reinterpret_cast<void*>((reinterpret_cast<uintptr_t>(map) + kHugePage - 1) & ~(uintptr_t)(kHugePage - 1));
        madvise(p, len, MADV_HUGEPAGE);
    }
    const hipError_t e = hipHostRegister(p, len, hipHostRegisterMapped);
    if (e != hipSuccess) {
        munmap(map, map_len);
        throw std::runtime_error(std::format("hipHostRegister of {} bytes: {}", len, hipGetErrorString(e)));
    }
    registered().push_back({p, len, map, map_len});
    return p;
}

void release_registered_host_memory() {
    for (const Registered& r : registered()) {
        (void)hipHostUnregister(r.base);
        munmap(r.map, r.map_len);
    }
    registered().clear();
}

namespace {
std::string sysfs_mode(const char* path) {
    std::ifstream in(path);
    std::string t;
    std::getline(in, t);
    const size_t l = t.find('['), r = t.find(']');
    return l != std::string::npos && r != std::string::npos && r > l ? t.substr(l + 1, r - l - 1) : "unknown";
}

// Bytes of huge pages in the VMAs that are madvised for them (VmFlags "hg"): the registered buffers, nothing else of
// this process asks for huge pages.
size_t madvised_huge_bytes() {
    std::ifstream in("/proc/self/smaps");
    std::string line;
    size_t kb = 0, total = 0;
    while (std::getline(in, line)) {
        if (line.starts_with("AnonHugePages:"))
            kb = std::strtoull(line.c_str() + 14, nullptr, 10);
        else if (line.starts_with("VmFlags:") && line.contains(" hg"))
            total += kb * 1024;
    }
    return total;
}
}  // namespace

void log_registered_memory(size_t registered_bytes) {
    const std::string enabled = sysfs_mode("/sys/kernel/mm/transparent_hugepage/enabled"),
                      defrag = sysfs_mode("/sys/kernel/mm/transparent_hugepage/defrag");
    const size_t huge = madvised_huge_bytes();
    std::fprintf(stderr,
                 "host memory: transparent huge pages enabled=%s defrag=%s, %.1f of %.1f GB of the registered memory "
                 "on huge pages\n",
                 enabled.c_str(), defrag.c_str(), (double)huge / 1e9, (double)registered_bytes / 1e9);
    if (env::integer("TONA_USERPTR_THP", 1) != 0 && (double)huge < 0.9 * (double)registered_bytes)
        std::fprintf(stderr,
                     "warning: the registered memory is mostly 4 KB pages (transparent_hugepage=%s, defrag=%s): "
                     "page-cache pressure "
                     "then stalls the GPU queues for seconds (research/measurements/2026-11-11-evict.md); 'madvise' or "
                     "'always' is "
                     "needed, and free memory for 2 MB pages\n",
                     enabled.c_str(), defrag.c_str());
}

std::string gtt_full_hint() {
    if (host_memory() != HostMemory::Gtt) return {};
    const Gtt g = read_gtt();
    return std::format(
        " (the host memory is GTT, {:.1f} of {:.1f} GB in use; the buffers do not fit: set HSA_USERPTR_FOR_PAGED_MEM=1 "
        "to use userptr memory without this limit, at the price of queue evictions under memory pressure)",
        (double)g.used / 1e9, (double)g.total / 1e9);
}

}  // namespace tona::gpu
