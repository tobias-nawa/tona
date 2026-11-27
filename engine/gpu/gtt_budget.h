// Host memory as GTT buffers (HSA_USERPTR_FOR_PAGED_MEM=0, backend/runtime_env.h): the kernel limits their sum
// (mem_info_gtt_total, half of the RAM by default: 32.5 GB on the reference PC), userptr memory is not limited by it.
// Pure functions, so they are unit-tested without a GPU; the sysfs and HIP side is gpu/gtt.cpp.
#pragma once

#include <algorithm>
#include <cstdint>

namespace tona::gpu {

enum class HostMemory { Gtt, Userptr, Unknown };

// Room for the pinned experts: the budget, but at most what is left of the GTT after the other buffers, minus a margin
// for allocations that come later. 1 GiB: the stash and the MTP rows are small and made before the pinned stage
// (research/measurements/2026-11-01-slow-decode.md).
inline int64_t cap_pin_budget(int64_t budget, int64_t gtt_total, int64_t gtt_used, int64_t margin = 1LL << 30) {
    return std::min(budget, std::max<int64_t>(0, gtt_total - gtt_used - margin));
}

// Which kind of memory did hipHostMalloc give? A probe allocation of probe_bytes grows mem_info_gtt_used by about
// probe_bytes for a GTT buffer and not at all for a userptr. Other processes change the counter too, so a result in
// between is Unknown and counts as userptr: no cap, a warning in the log.
inline HostMemory classify_host_memory(int64_t used_before, int64_t used_after, int64_t probe_bytes) {
    const int64_t grew = used_after - used_before;
    if (grew >= probe_bytes * 3 / 4) return HostMemory::Gtt;
    if (grew <= probe_bytes / 4) return HostMemory::Userptr;
    return HostMemory::Unknown;
}

}  // namespace tona::gpu
