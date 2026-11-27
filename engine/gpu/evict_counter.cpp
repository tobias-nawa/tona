#include "gpu/evict_counter.h"

#include <fcntl.h>
#include <hip/hip_runtime.h>
#include <unistd.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/mega_types.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

namespace {

using Clock = std::chrono::steady_clock;

struct State {
    int fd = -1;
    bool enabled = false;
    uint64_t base_ms = 0;
    Clock::time_point t0;
};

State& state() {
    static State s;
    return s;
}

#ifdef TONA_WAIT_DIAG
// TONA_TEST_EVICT_MS: the answer to every request, instead of the counter (a simulated freeze does not raise it)
int64_t test_ms() {
    static const int64_t v = env::integer("TONA_TEST_EVICT_MS", 0);
    return v;
}
#else
int64_t test_ms() { return 0; }
#endif

bool read_counter(uint64_t& ms) {
    char buf[32] = {};
    // pread at 0: sysfs runs the show function again for every read
    const ssize_t n = pread(state().fd, buf, sizeof buf - 1, 0);
    if (n <= 0) return false;
    ms = std::strtoull(buf, nullptr, 10);
    return true;
}

std::string read_word(const std::string& path) {
    std::ifstream in(path);
    std::string w;
    in >> w;
    return w;
}

// gpu_id of the KFD node with this location (the topology lists the iGPU and the CPU too), else 0
uint32_t gpu_id_of(int64_t location, long domain) {
    for (int n = 0; n < 16; n++) {
        const std::string dir = std::format("/sys/class/kfd/kfd/topology/nodes/{}/", n);
        std::ifstream in(dir + "properties");
        if (!in) break;
        std::string key;
        long long v;
        long long loc = -1, dom = -1;
        while (in >> key >> v) {
            if (key == "location_id") loc = v;
            if (key == "domain") dom = v;
        }
        if (loc == location && dom == domain && loc != 0)
            return (uint32_t)std::strtoul(read_word(dir + "gpu_id").c_str(), nullptr, 10);
    }
    return 0;
}

}  // namespace

uint32_t kfd_gpu_id() {
    char pci[32] = {};
    if (hipDeviceGetPCIBusId(pci, sizeof pci, 0) != hipSuccess) return 0;
    const int64_t loc = kfd_location_id(pci);
    if (loc < 0) return 0;
    return gpu_id_of(loc, std::strtol(pci, nullptr, 16));
}

namespace {

std::string find_path() {
    const uint32_t id = kfd_gpu_id();
    return id ? kfd_evicted_path((int)getpid(), id) : "";
}

}  // namespace

bool evict_credit_init() {
    State& s = state();
    if (env::text("TONA_EVICT_CREDIT") && !env::flag("TONA_EVICT_CREDIT")) {
        std::fprintf(stderr, "evicted_ms counter: off (TONA_EVICT_CREDIT=0), the pass deadline counts evictions\n");
        return s.enabled = false;
    }
    if (test_ms() > 0) {
        std::fprintf(stderr, "evicted_ms counter: TEST, every request is answered with %lld ms\n",
                     (long long)test_ms());
        return s.enabled = true;
    }
    const std::string path = find_path();
    s.fd = path.empty() ? -1 : open(path.c_str(), O_RDONLY | O_CLOEXEC);
    uint64_t ms;
    s.enabled = s.fd >= 0 && read_counter(ms);
    if (s.enabled)
        std::fprintf(stderr,
                     "evicted_ms counter: %s (%llu ms so far), evictions do not count against the pass deadline\n",
                     path.c_str(), (unsigned long long)ms);
    else
        std::fprintf(stderr, "evicted_ms counter: not available (%s), the pass deadline counts evictions\n",
                     path.empty() ? "GPU not found in the KFD topology" : path.c_str());
    return s.enabled;
}

bool evict_credit_enabled() { return state().enabled; }

void evict_begin() {
    State& s = state();
    if (!s.enabled) return;
    s.t0 = Clock::now();
    uint64_t ms;
    if (test_ms() > 0) return;
    if (read_counter(ms))
        s.base_ms = ms;
    else
        s.enabled = false;
}

void evict_serve(MegaMailbox* mb) {
    State& s = state();
    if (!s.enabled) return;
    const uint32_t req = __atomic_load_n(&mb->ev_req, __ATOMIC_ACQUIRE);
    if (req == __atomic_load_n(&mb->ev_ack, __ATOMIC_RELAXED)) return;
    const uint64_t elapsed =
        (uint64_t)std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - s.t0).count();
    uint64_t now = 0;
    uint32_t ms = 0;
    if (test_ms() > 0)
        ms = (uint32_t)test_ms();
    else if (read_counter(now))
        ms = evict_answer_ms(now, s.base_ms, elapsed);
    __atomic_store_n(&mb->ev_ms, ms, __ATOMIC_RELAXED);
    __atomic_store_n(&mb->ev_ack, req, __ATOMIC_RELEASE);
    std::fprintf(stderr, "gpu::Mega: eviction credit %u ms (counter +%llu ms, %llu ms since the launch)\n", ms,
                 (unsigned long long)(now > s.base_ms ? now - s.base_ms : 0), (unsigned long long)elapsed);
}

void evict_sync(MegaMailbox* mb) {
    if (!state().enabled) {
        HIPCHECK(hipStreamSynchronize(nullptr));
        return;
    }
    // hipStreamQuery does not block: the kernel may still ask while it runs on after the last router
    while (hipStreamQuery(nullptr) == hipErrorNotReady) {
        evict_serve(mb);
        __builtin_ia32_pause();
    }
    HIPCHECK(hipStreamSynchronize(nullptr));
}

void evict_check_test(int budget_ms) {
    if (test_ms() <= 0) return;
    // the kernel must not run longer than today's deadline plus grace (1.5 s + 0.3 s): its real run time is about
    // the freeze plus the pass, and the credit is granted for the freeze
    if (budget_ms <= 0 || budget_ms + test_ms() >= 1000)
        throw std::runtime_error(
            std::format("TONA_TEST_EVICT_MS={}: TONA_MEGA_BUDGET_MS ({}) plus the simulated credit must be "
                        "below 1000 ms",
                        (long long)test_ms(), budget_ms));
}

}  // namespace tona::gpu
