// TONA_METAL_ARENA_PURGEABLE: the arena's pin blocks as memory the system may take back at once, instead of a free-RAM
// reserve (research/53-mac-purgeable-arena.md). A block is volatile while no pass reads it; macOS purges volatile
// blocks before it compresses anything, in the order they were marked (a 10 GB balloon took 1.5 GB of them within a
// second and compressed nothing, research/measurements/2026-11-16-mac-purgeable-probe.md). A stop makes the blocks it
// routes to non-volatile (0.3 us each) and reads again only the routed experts of a purged block. Blocks stay
// non-volatile after use until other programs' demand shows (demand(), every 50 ms) or the pressure level is warn:
// macOS takes volatile memory even when nobody needs it, and a hold after use either lost decode or held most of the
// arena (research/measurements/2026-11-16-mac-purgeable-hold.md; the hold's code last lived in b9640fe). A GPU read of
// a volatile block is undefined (MTLResource.h): a stop wakes every block it routes to before the GPU reads it, and
// demand() marks only blocks the running pass has not routed to.
#include <mach/mach.h>
#include <sys/sysctl.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "backend/metal/arena.h"

namespace tona::metal {

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

// Bytes of this process's memory in the compressor: the fill's machine-wide count cannot tell our own fresh blocks
// (anonymous pages the GPU has not touched yet) from other programs' memory
int64_t own_compressed() {
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
        return -1;
    return (int64_t)info.compressed;
}

// The dispatch source reports changes of the level only: a machine already at warn when the server starts sends
// nothing (no event in 2026-11-16-mac-purgeable-balloon.md), so the monitor reads the level itself. 1 normal, 2 warn,
// 4 critical (sys/kern_memorystatus.h)
void Arena::read_level() {
    int level = 0;
    size_t len = sizeof level;
    if (sysctlbyname("kern.memorystatus_vm_pressure_level", &level, &len, nullptr, 0) == 0) pressure_ = level >= 2;
}

void Arena::mark_volatile(size_t k) {
    ArenaBlock& b = blocks_[k];
    if ([b.buf setPurgeableState:MTLPurgeableStateVolatile] == MTLPurgeableStateEmpty) purged(k);
    b.vol = true, stats_.settled++;
}

// The pins of a block are one run of pins_ (lay_out places them in order)
void Arena::purged(size_t k) {
    const ArenaBlock& b = blocks_[k];
    const auto lo = std::lower_bound(pin_at_.begin(), pin_at_.end(), b.at) - pin_at_.begin();
    const auto hi = std::lower_bound(pin_at_.begin(), pin_at_.end(), b.at + b.bytes) - pin_at_.begin();
    std::fill(pin_ok_.begin() + lo, pin_ok_.begin() + hi, 0);
    stats_.purged++;
}

// Block k non-volatile for the running pass; false: it was purged under demand and is freed (given way: the cold path
// reads its experts)
bool Arena::wake_block(size_t k) {
    ArenaBlock& b = blocks_[k];
    if (b.vol && [b.buf setPurgeableState:MTLPurgeableStateKeepCurrent] == MTLPurgeableStateEmpty) {
        free_block(k), given_++;
        return false;
    }
    if (b.vol) {
        if ([b.buf setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateEmpty) purged(k);
        b.vol = false, stats_.woken++;
    }
    b.used = passes_;
    return true;
}

void Arena::wake(uint32_t l, const int32_t* ids, int n) {
    if (!purge_) return;
    const double t0 = now_s();
    Jobs jobs;
    std::unique_lock<std::mutex> hold(state_mu_);
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (e < 0 || (uint32_t)e >= experts_) continue;
        const int64_t s = slot_[(size_t)l * experts_ + e];
        const int32_t p = pin_of_[(size_t)l * experts_ + e];
        if (s < 0 || p < 0) continue;
        if (!wake_block((size_t)(&block((uint64_t)s) - blocks_.data()))) continue;
        if (!pin_ok_[p]) point(l, e, (uint64_t)s, &jobs), pin_ok_[p] = 1, stats_.refilled++;
    }
    hold.unlock();
    for (const auto& [piece, dst] : jobs) refill_bytes_ += (int64_t)piece->bytes;
    read_pieces(jobs);
    stats_.wake_s += now_s() - t0;
}

// Only after a pass completed (the pass lock is held) or while idle: Metal refuses a volatile mark on a buffer a
// command buffer still uses
void Arena::settle() {
    if (!purge_) return;
    std::lock_guard<std::mutex> lock(state_mu_);
    std::vector<size_t> idle;
    for (size_t k = 0; k + 1 < blocks_.size(); k++) {
        const ArenaBlock& b = blocks_[k];
        if (b.buf && !b.vol && b.at < cache_at_) idle.push_back(k);
    }
    // The longest unused first; never used: the least used by the profile (the highest blocks) first
    std::stable_sort(idle.begin(), idle.end(), [&](size_t a, size_t c) {
        return blocks_[a].used != blocks_[c].used ? blocks_[a].used < blocks_[c].used : a > c;
    });
    for (size_t k : idle) mark_volatile(k);
    log_pressure(pressure_, idle.size());
}

void Arena::demand() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
            KERN_SUCCESS ||
        host_page_size(mach_host_self(), &page) != KERN_SUCCESS)
        return;
    const int64_t own = own_compressed();
    const auto machine = (int64_t)(vm.compressions - compressions_) * (int64_t)page;
    const bool first = compressions_ == 0;
    compressions_ = vm.compressions;
    const int64_t refill = refill_bytes_.load();
    const int64_t ours = std::max<int64_t>(0, own - own_at_) + (refill - refill_at_);
    own_at_ = own, refill_at_ = refill;
    if (first) return;
    samples_++;
    if (machine <= ours) return;
    // Not routed by the running pass: its later stops wake what they route to before the GPU reads it
    std::lock_guard<std::mutex> lock(state_mu_);
    if (last_demand_ > 0) dgaps_[dgaps_n_++ % dgaps_.size()] = samples_ - last_demand_;
    last_demand_ = samples_.load(), stats_.demands++;
    size_t marked = 0;
    for (size_t k = blocks_.size() - 1; k-- > 0;) {  // the least used first: macOS purges in the order of the marks
        const ArenaBlock& b = blocks_[k];
        if (b.buf && !b.vol && b.at < cache_at_ && b.used < passes_) mark_volatile(k), marked++;
    }
    if (marked)
        std::fprintf(stderr, "Metal arena: other programs compressed %.0f MB: %zu blocks volatile at %.3f\n",
                     (double)(machine - ours) / 1e6, marked,
                     std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count());
}

uint64_t Arena::demand_gap() const {
    const size_t n = std::min(dgaps_n_, dgaps_.size());
    if (n == 0) return 1;
    std::array<uint64_t, 64> g = dgaps_;
    std::nth_element(g.begin(), g.begin() + (ptrdiff_t)(n / 2), g.begin() + (ptrdiff_t)n);
    return std::max<uint64_t>(1, g[n / 2]);
}

// At the safe point before a pass. One growth step (the governor's, a mechanism size) once wait_ samples came without
// other programs' demand; the wait_ samples after it judge it
void Arena::regrow() {
    std::lock_guard<std::mutex> lock(state_mu_);
    const uint64_t s = samples_, base = demand_gap();
    if (step_pending_) {
        if (s < step_at_ + wait_) return;  // judged over wait_ samples after the step
        step_pending_ = false;
        const bool failed = last_demand_ > step_at_;
        wait_ = failed ? wait_ * 2 : std::max(base, wait_ / 2);
        std::fprintf(stderr,
                     "Metal arena: read back a step, %s; holds %.2f GB, gave back %llu blocks so far, next after %llu "
                     "clean samples\n",
                     failed ? "other programs compressed after it" : "no other compression", (double)held_ / 1e9,
                     (unsigned long long)given_, (unsigned long long)wait_);
    }
    wait_ = std::max(wait_, base);
    if (held_ >= (int64_t)size_ || s < last_demand_ + wait_) return;
    if (grow(backend::RamGovernorConfig{}.grow_step) == 0) return;
    step_at_ = samples_, step_pending_ = true;
}

// One line when a pressure episode starts and one when it ends, with the wall clock, to line up with a balloon's log
void Arena::log_pressure(bool on, size_t marked) {
    if (on == in_pressure_) return;
    in_pressure_ = on;
    const double t = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
    if (on) {
        refilled_at_ = stats_.refilled;
        std::fprintf(stderr, "Metal arena: pressure at %.3f: %zu blocks (%.2f GB) volatile at once\n", t, marked,
                     (double)marked * (double)block_ / 1e9);
    } else {
        std::fprintf(stderr, "Metal arena: pressure over at %.3f: %llu experts read again meanwhile\n", t,
                     (unsigned long long)(stats_.refilled - refilled_at_));
    }
}

// The fill marked each block volatile as it came in, the most used first, so the system would take those first. All
// back to non-volatile, then volatile again from the least used: the purge order is the order of the marks
void Arena::purge_order() {
    if (!purge_) return;
    pin_of_.assign((size_t)layers_ * experts_, -1);
    pin_ok_.assign(pins_.size(), 0);
    for (size_t i = 0; i < pins_.size(); i++) {
        pin_of_[(size_t)pins_[i].first * experts_ + pins_[i].second] = (int32_t)i;
        pin_ok_[i] = block(pin_at_[i]).buf != nil;
    }
    for (size_t k = 0; k + 1 < blocks_.size(); k++) {
        ArenaBlock& b = blocks_[k];
        if (!b.buf || !b.vol || b.at >= cache_at_) continue;
        if ([b.buf setPurgeableState:MTLPurgeableStateNonVolatile] == MTLPurgeableStateEmpty) purged(k);
        b.vol = false;
    }
    read_level();
    if (pressure_) settle();  // on demand only: volatile memory is taken even when nobody needs it
}

}  // namespace tona::metal
