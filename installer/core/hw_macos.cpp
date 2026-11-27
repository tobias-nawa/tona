// Hardware detection on macOS (Apple silicon): sysctl and the Mach VM statistics, no root, no Metal. The GPU shares
// the RAM; how much of it Metal lets a process keep resident is estimated here, the engine reads the exact value
// (MTLDevice.recommendedMaxWorkingSetSize) at start.
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <cctype>
#include <cstring>

#include "core/hw.h"

namespace tona::install {

namespace {

std::string sysctl_text(const char* name) {
    size_t len = 0;
    if (sysctlbyname(name, nullptr, &len, nullptr, 0) != 0 || len == 0) return {};
    std::string s(len, '\0');
    if (sysctlbyname(name, s.data(), &len, nullptr, 0) != 0) return {};
    s.resize(strnlen(s.c_str(), len));
    return s;
}

int64_t sysctl_number(const char* name) {
    int64_t v = 0;  // some values are 32 bits wide: they fill the low half of the zeroed value
    size_t len = sizeof v;
    return sysctlbyname(name, &v, &len, nullptr, 0) == 0 ? v : 0;
}

// Free, inactive and purgeable pages: what the system hands out without swapping
int64_t available_bytes() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &n) != KERN_SUCCESS)
        return 0;
    const auto pages = static_cast<int64_t>(vm.free_count) + vm.inactive_count + vm.purgeable_count;
    return pages * static_cast<int64_t>(vm_kernel_page_size);
}

// The GPU working set: the wired limit if the user set one (iogpu.wired_limit_mb), else 78 % of the RAM, the share
// measured on the M1 Max 32 GB (26.8 of 34.4 GB, engine/backend/metal/AGENTS.md); other sizes are not measured.
int64_t gpu_working_set(int64_t ram) {
    const int64_t mb = sysctl_number("iogpu.wired_limit_mb");
    return mb > 0 ? mb << 20 : ram / 100 * 78;
}

// "Apple M1 Max" -> "apple-m1-max", as in the backend's info name (metal-apple-m1-max)
std::string chip_id(std::string brand) {
    std::ranges::replace(brand, ' ', '-');
    std::ranges::transform(brand, brand.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return brand;
}

}  // namespace

void detect_macos(Hardware& h) {
    h.os = "macos";
    h.distro = "macos";
    h.distro_name = "macOS " + sysctl_text("kern.osproductversion");
    h.cpu = sysctl_text("machdep.cpu.brand_string");
    h.cpu_vendor = "Apple";
    h.cores = static_cast<int>(sysctl_number("hw.physicalcpu"));
    h.threads = static_cast<int>(sysctl_number("hw.logicalcpu"));
    h.ram_total = sysctl_number("hw.memsize");
    h.ram_avail = available_bytes();
    if (!h.cpu.starts_with("Apple M")) return;  // an Intel Mac: no Metal backend for it
    Gpu g;
    g.vendor = "Apple";
    g.name = h.cpu;
    g.arch = chip_id(h.cpu);
    g.vram_total = gpu_working_set(h.ram_total);
    h.gpus.push_back(g);
}

}  // namespace tona::install
#endif
