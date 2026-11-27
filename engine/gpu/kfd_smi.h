// KFD SMI events of this process (research/58-vram-instant-giveback.md, section 3.4): the kernel tells a process
// when its queues are evicted and restored, one text line per event. Plain C++ (no HIP) so the parsing is unit-tested;
// the thread that reads the device is gpu/kfd_smi.cpp. Diagnostics only: nothing in the engine reacts to an event yet.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace tona::gpu::smi {

// Event numbers of include/uapi/linux/kfd_ioctl.h (enum kfd_smi_event); the mask bit of event i is 1 << (i - 1)
enum Event : int {
    kVmFault = 1,
    kGpuPreReset = 3,
    kGpuPostReset = 4,
    kMigrateStart = 5,
    kMigrateEnd = 6,
    kQueueEviction = 9,
    kQueueRestore = 10,
    kUnmapFromGpu = 11,
};

constexpr uint64_t mask_of(int event) { return 1ULL << (event - 1); }

// Everything that explains an eviction or a stall. The page-fault events are left out: gfx1200 has no XNACK.
constexpr uint64_t kLoggedEvents = mask_of(kVmFault) | mask_of(kGpuPreReset) | mask_of(kGpuPostReset) |
                                   mask_of(kMigrateStart) | mask_of(kMigrateEnd) | mask_of(kQueueEviction) |
                                   mask_of(kQueueRestore) | mask_of(kUnmapFromGpu);

struct Line {
    int event = 0;     // the leading hexadecimal number
    int64_t ns = -1;   // kernel timestamp (CLOCK_BOOTTIME) of the events that carry one, else -1
    int pid = 0;       // "-<pid>" of those events
    int trigger = -1;  // QUEUE_EVICTION: enum KFD_QUEUE_EVICTION_TRIGGERS; MIGRATE_END and UNMAP: the trigger
    char tag = 0;      // QUEUE_RESTORE: 'R' if the restore failed and was rescheduled, else '0'
    std::string rest;  // the line after the event number, without the newline
};

// "9 123456789 -4242 d3f1 2" (QUEUE_EVICTION) and the other formats of kfd_ioctl.h (KFD_EVENT_FMT_*). False: not a
// line of this format (an empty line, no event number); the fields the line does not have stay at their defaults.
bool parse_line(std::string_view text, Line& out);

// Name of an event number or of an eviction trigger, for the log
const char* event_name(int event);
const char* eviction_trigger_name(int trigger);

// Cuts a stream of read() results into lines: a read may end inside a line, the rest comes with the next read.
class LineSplitter {
public:
    // Appends the bytes and returns the complete lines (without the newline)
    std::vector<std::string> feed(std::string_view bytes);

private:
    std::string pending_;
};

// One log line: "<arrival ns> <name> <fields...>" where arrival is the reader's CLOCK_BOOTTIME (same clock as the
// kernel timestamp, so latency = arrival - ns), then the raw line.
std::string format_log_line(int64_t arrival_ns, const Line& line);

// Starts the reader thread if TONA_KFD_SMI_LOG is set (path, "%p" becomes the PID). gpu_id: the KFD gpu_id of the
// GPU (0: unknown, nothing starts). Logs one line on stderr what it did. Safe to call twice; the thread ends with
// the process.
void start_log(uint32_t gpu_id);

}  // namespace tona::gpu::smi
