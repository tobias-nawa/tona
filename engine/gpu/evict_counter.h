// The host side of the eviction credit (protocol and arithmetic in gpu/wait_clock.h): reads the KFD counter
// evicted_ms of this process and this GPU and answers the questions of waiting blocks.
#pragma once

#include <algorithm>
#include <charconv>
#include <cstdint>
#include <format>
#include <string>

#include "gpu/mega_mailbox_fwd.h"
#include "gpu/wait_clock.h"

namespace tona::gpu {

// Answer to a block: the growth of the counter since the launch, at most the wall time since the launch and at most
// the cap. A counter that went backwards (process restarted its accounting) answers 0.
inline uint32_t evict_answer_ms(uint64_t now_ms, uint64_t base_ms, uint64_t elapsed_ms) {
    if (now_ms <= base_ms) return 0;
    return (uint32_t)std::min({now_ms - base_ms, elapsed_ms, kEvictCapMs});
}

inline std::string kfd_evicted_path(int pid, uint32_t gpu_id) {
    return std::format("/sys/class/kfd/kfd/proc/{}/stats_{}/evicted_ms", pid, gpu_id);
}

// KFD's location_id of a PCI address "dddd:bb:dd.f" (bus << 8 | device << 3 | function), -1 if it does not parse
inline int64_t kfd_location_id(const std::string& pci) {
    // "dddd:bb:dd.f", all hex
    unsigned v[4] = {};
    const char* p = pci.data();
    const char* const end = p + pci.size();
    for (int i = 0; i < 4; i++) {
        const auto r = std::from_chars(p, end, v[i], 16);
        if (r.ec != std::errc() || (i < 3 && (r.ptr == end || *r.ptr != (i < 2 ? ':' : '.')))) return -1;
        p = r.ptr + 1;
    }
    return (int64_t)(v[1] << 8 | v[2] << 3 | v[3]);
}

// gpu_id of HIP device 0 in the KFD topology (the number in the sysfs paths and the SMI ioctl), 0 if not found
uint32_t kfd_gpu_id();

// Finds the counter once and logs the result (call after the HIP device is up). False: no credit, the deadline
// aborts as before.
bool evict_credit_init();
bool evict_credit_enabled();
// Baseline at a launch (right after the launch call: a late baseline can only lower the answer)
void evict_begin();
// Answers a pending request of a block, if there is one
void evict_serve(MegaMailbox* mb);
// hipStreamSynchronize(0) that keeps answering while the kernel runs
void evict_sync(MegaMailbox* mb);
// Test hook (build with TONA_WAIT_DIAG): refuses a simulated credit that could let the kernel run too long
void evict_check_test(int budget_ms);

}  // namespace tona::gpu
