#include "core/hw.h"

#include <sys/statvfs.h>
#include <unistd.h>

#include <filesystem>
#include <format>
#include <set>
#include <sstream>
#include <thread>

#include "core/parse.h"
#include "core/sys.h"

namespace fs = std::filesystem;

namespace tona::install {

namespace {

#ifndef __APPLE__
using parse::field;
using parse::num;

// names of known cards (PCI device id); for others only vendor and id are shown
std::string gpu_name(uint32_t vendor, uint32_t device) {
    if (vendor == 0x1002 && device == 0x7590) return "Radeon RX 9060 XT";
    if (vendor == 0x1002 && device == 0x7550) return "Radeon RX 9070 / 9070 XT";
    return {};
}

std::string vendor_name(uint32_t pci_vendor) {
    switch (pci_vendor) {
        case 0x1002:
            return "AMD";
        case 0x10de:
            return "NVIDIA";
        case 0x8086:
            return "Intel";
        default:
            return "?";
    }
}

void detect_os(Hardware& h) {
    h.os = "linux";
    const std::string osr = read_file("/etc/os-release");
    h.distro = field(osr, "ID", '=');
    h.distro_like = field(osr, "ID_LIKE", '=');
    h.distro_name = field(osr, "PRETTY_NAME", '=');
}

void detect_cpu_and_memory(Hardware& h) {
    const std::string cpu = read_file("/proc/cpuinfo");
    h.cpu = field(cpu, "model name", ':');
    h.cpu_vendor = field(cpu, "vendor_id", ':');
    h.cpu_family = (int)num(field(cpu, "cpu family", ':'));
    h.cores = (int)num(field(cpu, "cpu cores", ':'));
    h.threads = (int)std::thread::hardware_concurrency();
    const std::string flags = field(cpu, "flags", ':');
    h.avx512 = parse::has_flag(flags, "avx512f");
    h.avx512_vnni = parse::has_flag(flags, "avx512_vnni");
    const std::string mem = read_file("/proc/meminfo");
    h.ram_total = num(field(mem, "MemTotal", ':')) * 1024;  // "… kB": stoll reads only the number
    h.ram_avail = num(field(mem, "MemAvailable", ':')) * 1024;
}

// number of the render node of a DRM device (renderD128 → 128), -1 if none
int render_minor(const fs::path& dev) {
    std::error_code ec;
    for (const auto& d : fs::directory_iterator(dev / "drm", ec)) {
        const std::string n = d.path().filename();
        if (n.starts_with("renderD")) return (int)num(n.substr(7));
    }
    return -1;
}

Gpu read_gpu(const fs::path& dev) {
    Gpu g;
    g.pci_vendor = (uint32_t)num(trim(read_file(dev / "vendor")));
    g.pci_device = (uint32_t)num(trim(read_file(dev / "device")));
    g.vendor = vendor_name(g.pci_vendor);
    g.name = gpu_name(g.pci_vendor, g.pci_device);
    g.vram_total = num(trim(read_file(dev / "mem_info_vram_total")));
    g.vram_used = num(trim(read_file(dev / "mem_info_vram_used")));
    // like the engine: below 4 GB of own VRAM a card counts as integrated (gpu/mega.hip, vram_free_sysfs)
    g.integrated = g.pci_vendor == 0x8086 || (g.pci_vendor == 0x1002 && g.vram_total < (4LL << 30));
    return g;
}

// KFD nodes carry gfx_target_version (120000 → gfx1200) and the number of their render node
void detect_amd_arch(std::vector<Gpu>& gpus, const std::vector<int>& minors) {
    std::error_code ec;
    for (const auto& n : fs::directory_iterator("/sys/class/kfd/kfd/topology/nodes", ec)) {
        const std::string props = read_file(n.path() / "properties");
        const int64_t v = num(field(props, "gfx_target_version", ' '));
        const int64_t minor = num(field(props, "drm_render_minor", ' '));
        if (v <= 0) continue;
        for (size_t i = 0; i < gpus.size(); i++)
            if (minors[i] == minor) gpus[i].arch = parse::gfx_name(v);
    }
}

// NVIDIA does not report VRAM through sysfs
void detect_nvidia_vram(std::vector<Gpu>& gpus) {
    if (!which("nvidia-smi")) return;
    const RunResult r = run("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits");
    for (const parse::NvidiaRow& row : parse::nvidia_rows(r.status == 0 ? r.out : "")) {
        for (Gpu& g : gpus) {
            if (g.vendor != "NVIDIA" || g.vram_total) continue;
            g.name = row.name, g.vram_total = row.total_mib << 20, g.vram_used = row.used_mib << 20;
            break;
        }
    }
}

void detect_gpus(Hardware& h) {
    std::error_code ec;
    std::set<std::string> cards;
    for (const auto& c : fs::directory_iterator("/sys/class/drm", ec)) {
        const std::string name = c.path().filename();
        if (name.starts_with("card") && !name.contains('-')) cards.insert(c.path());
    }
    std::vector<int> minors;
    for (const std::string& c : cards) {
        const fs::path dev = fs::path(c) / "device";
        const Gpu g = read_gpu(dev);
        if (!g.pci_vendor) continue;
        h.gpus.push_back(g);
        minors.push_back(render_minor(dev));
    }
    detect_amd_arch(h.gpus, minors);
    detect_nvidia_vram(h.gpus);
}

#endif

void detect_storage(Hardware& h, const std::string& repo) {
    const fs::path work = fs::path(repo) / "work";
    fs::path where = fs::exists(work) ? work : fs::path(repo);
    // a release root does not exist before the first installation: measure where it will be created
    while (!fs::exists(where) && where.has_parent_path() && where != where.parent_path()) where = where.parent_path();
    struct statvfs st{};
    if (statvfs(where.c_str(), &st) == 0) h.disk_free = (int64_t)st.f_bavail * (int64_t)st.f_frsize;
    h.disk = detect_disk(where.string());
    h.disk_path = repo;
}

}  // namespace

Hardware detect(const std::string& repo) {
    Hardware h;
#ifdef __APPLE__
    detect_macos(h);
#else
    detect_os(h);
    detect_cpu_and_memory(h);
    detect_gpus(h);
    h.kfd_access = access("/dev/kfd", R_OK | W_OK) == 0;
#endif
    detect_storage(h, repo);
    return h;
}

}  // namespace tona::install
