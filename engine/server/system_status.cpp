#include "server/system_status.h"

#include <cstdlib>
#include <filesystem>
#include <format>
#include <fstream>

#if defined(__APPLE__)
#include <sys/sysctl.h>

#include "backend/mem_macos.h"
#endif

namespace tona::server {

namespace {

// Number from a sysfs file (-1: missing)
double read_num(const std::string& path) {
    std::ifstream f(path);
    std::string text;
    if (!std::getline(f, text) || text.empty()) return -1;
    char* end = nullptr;
    const double v = std::strtod(text.c_str(), &end);
    return end == text.c_str() ? -1 : v;
}

std::string gpu_device() {
    std::string best;
    double most = 0;
    for (int i = 0; i < 8; i++) {
        const std::string d = std::format("/sys/class/drm/card{}/device", i);
        if (const double v = read_num(d + "/mem_info_vram_total"); v > most) most = v, best = d;
    }
    return best;
}

std::string hwmon_dir(const std::string& dev) {
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dev + "/hwmon", ec)) return e.path().string();
    return std::string();
}

#if defined(__APPLE__)
// Installed RAM and what can be taken without paging out (the arena's measure), in bytes. macOS has no /proc/meminfo
void read_ram(double& total, double& avail) {
    uint64_t ram = 0;
    size_t len = sizeof ram;
    total = sysctlbyname("hw.memsize", &ram, &len, nullptr, 0) == 0 ? static_cast<double>(ram) : 0;
    avail = backend::macos::available_bytes();
}
#else
// MemTotal and MemAvailable from /proc/meminfo, in bytes
void read_ram(double& total, double& avail) {
    total = avail = 0;
    std::ifstream f("/proc/meminfo");
    std::string key, unit;
    long long v = 0;
    while (f >> key >> v) {
        std::getline(f, unit);  // rest of the line ("kB")
        if (key == "MemTotal:") total = v * 1024.0;
        if (key == "MemAvailable:") avail = v * 1024.0;
    }
}
#endif

}  // namespace

json system_status() {
    static const std::string dev = gpu_device();
    static const std::string hw = hwmon_dir(dev);
    json g = {
        {"vram_used", read_num(dev + "/mem_info_vram_used")}, {"vram_total", read_num(dev + "/mem_info_vram_total")},
        {"busy", read_num(dev + "/gpu_busy_percent")},        {"power_w", read_num(hw + "/power1_average") / 1e6},
        {"power_cap_w", read_num(hw + "/power1_cap") / 1e6},  {"sclk_mhz", read_num(hw + "/freq1_input") / 1e6},
        {"temp_c", read_num(hw + "/temp1_input") / 1e3},      {"temp_junction_c", read_num(hw + "/temp2_input") / 1e3}};
#if defined(__APPLE__)
    // One memory for CPU and GPU, and no sysfs: the fields above stay -1 (unknown) and the client shows the server's
    // share of the RAM instead of a graphics memory
    g["unified"] = true;
#endif
    double total = 0, avail = 0;
    read_ram(total, avail);
    return {{"gpu", g}, {"ram", {{"total", total}, {"available", avail}}}};
}

}  // namespace tona::server
