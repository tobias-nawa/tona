// What kind of memory does hipHostMalloc give, and how much GTT is left (details in gpu/gtt_budget.h).
#pragma once

#include <cstdint>
#include <string>

#include "gpu/gtt_budget.h"

namespace tona::gpu {

// Once per process (cached): allocates a small probe buffer and reads mem_info_gtt_used before and after. Unknown if
// sysfs is not readable.
HostMemory host_memory();

// Writes one line to stderr: "host memory: GTT ..." or "host memory: userptr ...", so a runtime update that makes
// HSA_USERPTR_FOR_PAGED_MEM=0 ineffective shows up in every start log.
void log_host_memory();

// The budget for the pinned experts, capped to the free GTT if the host memory is GTT (cap_pin_budget). Logs a cap.
int64_t cap_pin_to_gtt(int64_t budget);

// What is left of the GTT for further buffers, minus the margin of cap_pin_budget; INT64_MAX if the host memory is not
// GTT (no limit then).
int64_t gtt_room();

// Anonymous memory registered with the GPU as userptr (mmap + hipHostRegister): for the part of the pinned experts that
// does not fit the GTT. The pages can be migrated, so a page migration evicts the queues, but only a part of the
// pinned memory is exposed. hipHostRegister only at load time, never while the kernel waits on the CPU. Returns the
// pointer, valid for the CPU and (hipHostGetDevicePointer) the GPU.
void* register_host_memory(size_t bytes);
void release_registered_host_memory();
// Start-log line: the THP mode of the kernel and how much of the registered memory is on huge pages (the madvised
// VMAs of this process), with a warning if TONA_USERPTR_THP is on and less than 90 % of it is.
void log_registered_memory(size_t registered_bytes);

// Text to append to an out-of-memory error of a host buffer when the host memory is GTT, otherwise empty.
std::string gtt_full_hint();

}  // namespace tona::gpu
