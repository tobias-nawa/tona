// Hardware detection for the installer, without root and without ROCm: Linux and macOS (hw_macos.cpp); other
// platforms come with their backends (docs/backends/README.md).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "core/disk.h"

namespace tona::install {

struct Gpu {
    uint32_t pci_vendor = 0, pci_device = 0;
    std::string vendor;                     // "AMD", "NVIDIA", "Intel", "Apple"
    std::string name;                       // marketing name, if known
    std::string arch;                       // e.g. "gfx1200" (AMD, from the KFD topology), "apple-m1-max"
    int64_t vram_total = 0, vram_used = 0;  // bytes; used includes the desktop; Apple: the GPU working set
    bool integrated = false;
};

struct Hardware {
    std::string os;                   // "linux", "macos"
    std::string distro, distro_like;  // ID and ID_LIKE from /etc/os-release
    std::string distro_name;          // PRETTY_NAME
    std::string cpu, cpu_vendor;      // model name; vendor_id (AuthenticAMD, GenuineIntel)
    int cpu_family = 0;               // 26 (0x1A): AMD Zen 5
    int cores = 0, threads = 0;
    bool avx512 = false, avx512_vnni = false;
    int64_t ram_total = 0, ram_avail = 0;  // bytes
    double ram_gbs = 0;  // GB/s the CPU reads RAM at, measured at detection (measure_ram_gbs); 0: not measured
    std::vector<Gpu> gpus;
    int64_t disk_free = 0;    // bytes free under work/
    Disk disk;                // drive under work/
    std::string disk_path;    // the folder that was measured for (the install folder, or the checkout)
    DiskSpeed disk_speed;     // measured (measure_disk), otherwise measured = false
    bool kfd_access = false;  // /dev/kfd readable and writable (group render)
};

// CPU read bandwidth of RAM in GB/s: threads read their own slice of a 512 MiB buffer, best of 3 passes; 0 if the
// buffer cannot be allocated. Not part of detect(): the installer measures it once per run for the speed estimate.
double measure_ram_gbs(int threads);

// repo: root of the engine repository (for the free space under work/)
Hardware detect(const std::string& repo);

#ifdef __APPLE__
void detect_macos(Hardware& h);  // everything but the drive (hw_macos.cpp)
#endif

}  // namespace tona::install
