// macOS counterparts of the /proc reads in backend/budget.h (docs/use/macos.md). Mach statistics, no privileges needed.
#pragma once

#include <mach/mach.h>

namespace tona::backend::macos {

// Memory the process holds: phys_footprint (resident anonymous memory plus its compressed pages), the figure the
// kernel's memory limits and Activity Monitor use; the closest match to RssAnon + RssShmem on Linux
inline long long held_bytes() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return 0;
    return static_cast<long long>(info.phys_footprint);
}

// Memory that can be taken without paging out: free, inactive and purgeable pages, the counterpart of MemAvailable.
// unchecked: how close this tracks the point where macOS starts compressing and swapping (to be measured)
inline double available_bytes() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
        KERN_SUCCESS)
        return 0;
    vm_size_t page = 0;
    if (host_page_size(mach_host_self(), &page) != KERN_SUCCESS) return 0;
    return static_cast<double>(vm.free_count + vm.inactive_count + vm.purgeable_count) * static_cast<double>(page);
}

}  // namespace tona::backend::macos
