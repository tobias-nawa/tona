#include "gpu/kfd_smi.h"

#include <fcntl.h>
#include <poll.h>
#include <sys/ioctl.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <charconv>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <thread>

#include "backend/env.h"
#include "gpu/trace_path.h"

namespace tona::gpu::smi {

namespace {

bool parse_int(std::string_view& s, int64_t& v, int base = 10) {
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    const auto r = std::from_chars(s.data(), s.data() + s.size(), v, base);
    if (r.ec != std::errc()) return false;
    s.remove_prefix(r.ptr - s.data());
    return true;
}

}  // namespace

namespace {

bool has_timestamp(int ev) {
    return ev == kQueueEviction || ev == kQueueRestore || ev == kMigrateStart || ev == kMigrateEnd ||
           ev == kUnmapFromGpu;
}

// "%lld -%d" at the start of the stamped events
bool parse_stamp(std::string_view& s, Line& out) {
    int64_t ns = 0, pid = 0;
    if (!parse_int(s, ns) || s.empty() || s.front() != ' ') return false;
    s.remove_prefix(1);
    if (s.empty() || s.front() != '-') return false;
    s.remove_prefix(1);
    if (!parse_int(s, pid)) return false;
    out.ns = ns;
    out.pid = (int)pid;
    return true;
}

// QUEUE_EVICTION "... %x %d" (node, trigger), QUEUE_RESTORE "... %x %c" (node, 'R' = rescheduled)
void parse_queue_tail(std::string_view& s, Line& out) {
    int64_t node = 0;
    if (!parse_int(s, node, 16)) return;
    while (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    if (s.empty()) return;
    if (out.event == kQueueRestore) {
        out.tag = s.front();
        return;
    }
    int64_t t = 0;
    if (parse_int(s, t)) out.trigger = (int)t;
}

}  // namespace

bool parse_line(std::string_view text, Line& out) {
    out = Line{};
    while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.remove_suffix(1);
    std::string_view s = text;
    int64_t ev = 0;
    if (!parse_int(s, ev, 16) || ev <= 0 || ev > 64) return false;
    out.event = (int)ev;
    if (!s.empty() && s.front() == ' ') s.remove_prefix(1);
    out.rest = std::string(s);
    if (!has_timestamp(out.event) || !parse_stamp(s, out)) return true;
    if (out.event == kQueueEviction || out.event == kQueueRestore) parse_queue_tail(s, out);
    return true;
}

const char* event_name(int event) {
    switch (event) {
        case kVmFault:
            return "VMFAULT";
        case kGpuPreReset:
            return "GPU_PRE_RESET";
        case kGpuPostReset:
            return "GPU_POST_RESET";
        case kMigrateStart:
            return "MIGRATE_START";
        case kMigrateEnd:
            return "MIGRATE_END";
        case kQueueEviction:
            return "QUEUE_EVICTION";
        case kQueueRestore:
            return "QUEUE_RESTORE";
        case kUnmapFromGpu:
            return "UNMAP_FROM_GPU";
        default:
            return "EVENT";
    }
}

const char* eviction_trigger_name(int trigger) {
    switch (trigger) {
        case 0:
            return "svm";
        case 1:
            return "userptr";
        case 2:
            return "ttm";
        case 3:
            return "suspend";
        case 4:
            return "criu_checkpoint";
        case 5:
            return "criu_restore";
        default:
            return "unknown";
    }
}

std::vector<std::string> LineSplitter::feed(std::string_view bytes) {
    std::vector<std::string> lines;
    pending_.append(bytes);
    size_t start = 0, nl;
    while ((nl = pending_.find('\n', start)) != std::string::npos) {
        lines.emplace_back(pending_, start, nl - start);
        start = nl + 1;
    }
    pending_.erase(0, start);
    return lines;
}

std::string format_log_line(int64_t arrival_ns, const Line& line) {
    std::string s = std::to_string(arrival_ns) + " " + event_name(line.event);
    if (line.event == kQueueEviction) s += std::string(" trigger=") + eviction_trigger_name(line.trigger);
    if (line.ns >= 0)
        s += " kernel_ns=" + std::to_string(line.ns) + " latency_us=" + std::to_string((arrival_ns - line.ns) / 1000);
    s += " | " + line.rest;
    return s;
}

namespace {

// struct kfd_ioctl_smi_events_args of kfd_ioctl.h, and AMDKFD_IOC_SMI_EVENTS = _IOWR('K', 0x1F, args)
struct SmiArgs {
    uint32_t gpuid;
    uint32_t anon_fd;
};
constexpr unsigned long kIocSmiEvents = _IOWR('K', 0x1F, SmiArgs);

int64_t boottime_ns() {
    timespec ts{};
    clock_gettime(CLOCK_BOOTTIME, &ts);
    return (int64_t)ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void reader(int fd, std::FILE* log) {
    LineSplitter splitter;
    char buf[4096];
    for (;;) {
        pollfd p{fd, POLLIN, 0};
        if (poll(&p, 1, -1) < 0) continue;
        const ssize_t n = read(fd, buf, sizeof buf);
        const int64_t arrival = boottime_ns();
        if (n <= 0) continue;
        for (const std::string& l : splitter.feed(std::string_view(buf, (size_t)n))) {
            Line line;
            if (!parse_line(l, line)) continue;
            // flushed per line: the process usually ends in an abort
            std::fprintf(log, "%s\n", format_log_line(arrival, line).c_str());
            std::fflush(log);
        }
    }
}

}  // namespace

void start_log(uint32_t gpu_id) {
    static std::atomic<bool> started{false};
    const char* const path = env::text("TONA_KFD_SMI_LOG");
    if (!path || gpu_id == 0 || started.exchange(true)) return;
    const std::string file = trace::expand_pid(path, (long)getpid());
    const int kfd = open("/dev/kfd", O_RDWR | O_CLOEXEC);
    SmiArgs args{gpu_id, 0};
    if (kfd < 0 || ioctl(kfd, kIocSmiEvents, &args) != 0) {
        std::fprintf(stderr, "KFD SMI log: not available (%s)\n", std::strerror(errno));
        if (kfd >= 0) close(kfd);
        return;
    }
    close(kfd);  // the event file stays valid without it
    const int fd = (int)args.anon_fd;
    const uint64_t mask = kLoggedEvents;
    std::FILE* log = std::fopen(file.c_str(), "w");
    if (!log || write(fd, &mask, sizeof mask) != (ssize_t)sizeof mask) {
        std::fprintf(stderr, "KFD SMI log: cannot start (%s)\n", log ? "subscription refused" : file.c_str());
        if (log) std::fclose(log);
        close(fd);
        return;
    }
    std::fprintf(log, "%lld START gpu_id=%u (timestamps: CLOCK_BOOTTIME ns)\n", (long long)boottime_ns(), gpu_id);
    std::fflush(log);
    std::thread(reader, fd, log).detach();
    std::fprintf(stderr, "KFD SMI log: %s (events of this process only, no root)\n", file.c_str());
}

}  // namespace tona::gpu::smi
