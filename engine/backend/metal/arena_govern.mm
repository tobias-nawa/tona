// The arena under memory pressure: a monitor samples free RAM every 250 ms and listens to the dispatch memory-pressure
// source (macOS's counterpart of PSI, research/44-memory-priority.md); backend::RamGovernor turns that into a byte
// delta; blocks are freed, the cache's last slots first, then the least used pins, and read again when RAM is free.
// Only while no pass runs (PassScope), so no command buffer reads a freed block.
#include <dispatch/dispatch.h>
#include <mach/mach.h>
#include <sys/mman.h>
#include <sys/stat.h>

#include <algorithm>
#include <chrono>
#include <cstdio>

#include "backend/env.h"
#include "backend/metal/arena.h"

namespace tona::metal {

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

const char* state_name(backend::RamState s) {
    switch (s) {
        case backend::RamState::calm:
            return "calm";
        case backend::RamState::band:
            return "band";
        case backend::RamState::tight:
            return "tight";
        case backend::RamState::short_of_memory:
            return "short";
    }
    return "?";
}

// Free pages alone, without the file cache free_ram() counts: dropping cache moves pages from one to the other
int64_t free_pages() {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
        KERN_SUCCESS)
        return 0;
    return (int64_t)vm.free_count * (int64_t)getpagesize();
}

}  // namespace

Arena::PassScope::PassScope(Arena* a) : a_(a) {
    if (a_) a_->begin_pass();
}

Arena::PassScope::~PassScope() {
    if (a_) a_->end_pass();
}

void Arena::start_governor() {
    churn_ = env::integer("TONA_METAL_ARENA_CHURN", 0);
    if (!env::flag("TONA_METAL_ARENA_GOVERN")) return;
    // Test without displacing other programs: free RAM reads TONA_METAL_ARENA_GOVERN_TEST_GB less for
    // TONA_METAL_ARENA_GOVERN_TEST_S seconds after the calibration, so the governor frees blocks and reads them again
    bias_ = (int64_t)(env::number("TONA_METAL_ARENA_GOVERN_TEST_GB", 0.0) * 1e9);
    bias_s_ = env::number("TONA_METAL_ARENA_GOVERN_TEST_S", 0.0);
    std::atomic<bool>* pressure = &pressure_;
    dispatch_source_t src = dispatch_source_create(
        DISPATCH_SOURCE_TYPE_MEMORYPRESSURE, 0,
        DISPATCH_MEMORYPRESSURE_NORMAL | DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL,
        dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
    dispatch_source_set_event_handler(src, ^{
      const uintptr_t level = dispatch_source_get_data(src);
      pressure->store((level & (DISPATCH_MEMORYPRESSURE_WARN | DISPATCH_MEMORYPRESSURE_CRITICAL)) != 0);
      const double t = std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
      std::fprintf(stderr, "Metal arena: memory pressure event %s at %.3f\n",
                   level & DISPATCH_MEMORYPRESSURE_CRITICAL ? "critical"
                   : level & DISPATCH_MEMORYPRESSURE_WARN   ? "warn"
                                                            : "normal",
                   t);
    });
    dispatch_resume(src);
    pressure_src_ = src;
    reserve_ = purge_ ? 0 : (int64_t)(reserve_gb() * 1e9);
    gov_ = backend::RamGovernor(backend::RamGovernorConfig::from_reserve(reserve_));
    mon_ = std::thread([this] { monitor(); });
}

void Arena::stop_governor() {
    stop_ = true;
    if (mon_.joinable()) mon_.join();
    if (pressure_src_) dispatch_source_cancel((dispatch_source_t)pressure_src_), pressure_src_ = nil;
}

// RAM that can be taken without paging out: free, file-backed and purgeable pages (the page cache is reclaimed
// first, as MemAvailable on Linux). Our weight mappings are not among them: Metal wires the pages of a resident
// buffer (wired 3.2 GB idle, 13.2 GB during a pass, 2026-11-15-mac-coder-arena.md)
int64_t Arena::free_ram(int64_t* compressed) const {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
            KERN_SUCCESS ||
        host_page_size(mach_host_self(), &page) != KERN_SUCCESS)
        return 0;
    const auto pages = (int64_t)vm.free_count + vm.external_page_count + vm.purgeable_count;
    if (compressed) *compressed = (int64_t)vm.compressions * (int64_t)page;
    return pages * (int64_t)page;
}

// TONA_METAL_ARENA_GOVERN: the blocks in growth steps at start (backend::fill_step), each while it leaves the growth
// floor free and the step before did not make the OS compress other programs' memory; the rest the governor grows
// later. Free RAM here is raw, before the calibration
void Arena::fill_in_steps() {
    const int64_t dropped = drop_cache();
    reserve_ = purge_ ? 0 : (int64_t)(reserve_gb() * 1e9);  // purgeable: no reserve, the system takes blocks back
    const auto c = backend::RamGovernorConfig::from_reserve(reserve_);
    const double t0 = now_s();
    const int64_t own0 = own_compressed();
    int64_t before = 0, after = 0, compressed = 0, total = 0, d = 0;
    int64_t free = free_ram(&before);
    while ((d = backend::fill_step(c, free, held_, (int64_t)size_, compressed)) > 0) {
        grow(d);
        free = free_ram(&after);
        compressed = after - before, total += compressed, before = after;
    }
    const char* why = held_ >= (int64_t)size_        ? "the budget"
                      : compressed > c.grow_step / 2 ? "the OS compressed memory"
                                                     : "the free RAM floor";
    std::fprintf(stderr,
                 "Metal arena: filled %.2f of %.2f GB in steps in %.1f s, stopped by %s; free RAM %.2f GB (raw), "
                 "compressed meanwhile %.2f GB (of our process %+.2f GB); the experts' page cache dropped first: %.2f "
                 "GB\n",
                 (double)held_ / 1e9, (double)size_ / 1e9, now_s() - t0, why, (double)free / 1e9, (double)total / 1e9,
                 (double)(own_compressed() - own0) / 1e9, (double)dropped / 1e9);
}

// The page cache of the routed experts' file ranges, dropped (msync MS_INVALIDATE; clean pages, no root needed): the
// arena reads them with F_NOCACHE only, but a server without the arena before it left them cached (20 GB of DM-T).
// The OS takes the room for the fill from its inactive pages, ours and other programs' alike, and compresses theirs
// (a fill of 14.5 GB compressed 13 GB with 10.8 GB of the file cached, 2026-11-16-mac-dmt-arena-ab.md); dropped, the
// fill takes free pages. Inner whole pages only: the edge pages may hold other tensors. Returns the rise of free pages
int64_t Arena::drop_cache() const {
    const auto page = (uint64_t)getpagesize();
    const int64_t before = free_pages();
    for (int fd : fds_) {
        struct stat st{};
        if (fstat(fd, &st) != 0) continue;
        void* a = mmap(nullptr, (size_t)st.st_size, PROT_READ, MAP_SHARED, fd, 0);
        if (a == MAP_FAILED) continue;
        for (uint32_t l = 0; l < layers_; l++)
            for (int k = 0; k < 3; k++) {
                const ExpertPiece &f = piece(l, k, 0), &e = piece(l, k, (int)experts_ - 1);
                if (f.fd != fd) continue;
                const uint64_t from = (f.file_at + f.skew + page - 1) / page * page, to = e.file_at + e.bytes - page;
                if (to > from) msync((uint8_t*)a + from, to - from, MS_INVALIDATE);
            }
        munmap(a, (size_t)st.st_size);
    }
    return free_pages() - before;
}

// Free RAM swings with the passes: Metal wires our weight mappings during a pass, idle they count as file pages
// (1.8 GB of swing with the Coder, 2026-11-15-mac-arena-balloon.md). The governor sees the least free RAM of the
// last WINDOW_S seconds, the value while passes run, calibrated once the first passes fill the window: then free
// RAM is R by construction (the budget left R), and what the page counts read less (kernel memory, about 1.4 GB on
// the M1 Max) is added to every sample, at most 3 GB
void Arena::sample_free() {
    const double t = now_s();
    const int64_t raw = free_ram();
    if (window_reset_.exchange(false)) window_at_ = 0;
    window_[window_at_++ % window_.size()] = raw;
    const int64_t least = *std::min_element(window_.begin(), window_.begin() + std::min(window_at_, window_.size()));
    if (!calibrated_ && last_pass_ > 0) {
        if (first_pass_ < 0) first_pass_ = t;
        if (t - first_pass_ >= WINDOW_S && t - last_pass_ < 1.0) {
            calib_ = budget_ < 0 ? std::clamp<int64_t>(reserve_ - least, 0, 3'000'000'000) : 0;
            calibrated_ = true, bias_until_ = t + bias_s_;
            std::fprintf(stderr, "Metal arena: governor on, reserve %.2f GB, free RAM calibrated by %+.2f GB\n",
                         (double)reserve_ / 1e9, (double)calib_ / 1e9);
        }
    }
    free_ = least + calib_ - (t < bias_until_ ? bias_ : 0), sample_++;
}

// Between passes the engine thread applies the governor (begin_pass); while the engine is idle for a second the
// monitor does, with the larger idle steps (1 s: 18 times the longest gap between two passes of one request, 54.9 ms
// in a prefill of 8000 tokens, research/measurements/2026-11-16-mac-purgeable-prefill2-server-ab.md). The purgeable
// blocks go volatile on other programs' demand (demand(), every 50 ms)
void Arena::monitor() {
    for (uint64_t tick = 0; !stop_; tick++) {
        // 50 ms: other programs' demand is seen within one sample (a 1 GB/s balloon compressed 1.6 GB in the 0.7 s
        // the arena took to react to the pressure level, 2026-11-16-mac-purgeable-balloon.md); the free-RAM window
        // keeps its 250 ms
        std::this_thread::sleep_for(std::chrono::milliseconds(purge_ ? 50 : 250));
        if (purge_) demand();
        if (purge_ && tick % 5 != 0) continue;
        sample_free();
        if (purge_) read_level();
        const double idle = now_s() - last_pass_;
        if (idle < 1.0 || !mu_.try_lock()) continue;
        if (!purge_) govern(false);
        mu_.unlock();
    }
}

void Arena::govern(bool pass_running) {
    sample_seen_ = sample_;
    if (!calibrated_) return;
    backend::RamSample s;
    s.now_s = now_s(), s.free = free_, s.pass_running = pass_running, s.pressure = pressure_;
    s.held = held_, s.budget = (int64_t)size_;
    const backend::RamState was = gov_.state();
    const int64_t d = gov_.step(s);
    if (gov_.state() != was)  // a line per change of state: why the arena does not grow is visible in the log
        std::fprintf(stderr, "Metal arena: %s (free %.2f GB), holds %.2f GB\n", state_name(gov_.state()),
                     (double)s.free / 1e9, (double)held_ / 1e9);
    // Growth only while passes run: idle, free RAM reads the swing higher than in use, and a grow then would turn
    // into a release at the next request. At the safe point before a pass: 512 MB read in about 0.1 s
    if (d == 0 || (d > 0 && now_s() - last_pass_ > WINDOW_S)) return;
    // The window restarts after a change: its least value from before the step would ask for the same step again
    // (a 3 GB test pressure freed 7.5 GB instead of 1.75, 2026-11-15-mac-arena-balloon.md)
    window_reset_ = true;
    const int64_t done = d < 0 ? -shrink(-d) : grow(d);
    std::fprintf(stderr, "Metal arena: %s %.0f MB (free %.2f GB, %s), holds %.2f GB\n", d < 0 ? "freed" : "read",
                 (double)(done < 0 ? -done : done) / 1e6, (double)s.free / 1e9, state_name(gov_.state()),
                 (double)held_ / 1e9);
}

void Arena::begin_pass() {
    mu_.lock();
    passes_++;
    if (churn_ > 0 && passes_ % (uint64_t)churn_ == 0) {  // test: a quarter of the blocks out, then all back
        const bool out = (passes_ / (uint64_t)churn_) % 2;
        const int64_t done = out ? shrink((int64_t)size_ / 4) : grow((int64_t)size_);
        std::fprintf(stderr, "Metal arena: churn %s %.0f MB, holds %.2f GB\n", out ? "freed" : "read",
                     (double)done / 1e6, (double)held_ / 1e9);
    } else if (purge_) {
        regrow();
    } else if (mon_.joinable() && sample_seen_ != sample_) {
        govern(true);
    }
}

void Arena::end_pass() {
    last_pass_ = now_s();
    if (pressure_) settle();  // the pass completed: its blocks may go volatile
    mu_.unlock();
}

// The highest blocks first: the cache's last slots of every layer, then the least used pins; never the staging slots.
// In a pool of its own: on the monitor thread, which has none, a freed buffer stayed in memory until the thread ended
// (the governor freed 11 GB of blocks and free RAM did not rise, 2026-11-15-mac-arena-balloon.md)
int64_t Arena::shrink(int64_t bytes) {
    finish_ahead();
    int64_t done = 0;
    @autoreleasepool {
        for (size_t k = blocks_.size() - 1; k-- > 0 && done < bytes;)
            if (blocks_[k].buf) done += (int64_t)blocks_[k].bytes, free_block(k);
    }
    return done;
}

// The lowest freed blocks first: the most used pins, then the cache
int64_t Arena::grow(int64_t bytes) {
    finish_ahead();
    int64_t done = 0;
    @autoreleasepool {
        for (size_t k = 0; k + 1 < blocks_.size() && done < bytes; k++)
            if (!blocks_[k].buf) read_block(k), done += (int64_t)blocks_[k].bytes;
    }
    return done;
}

void Arena::free_block(size_t k) {
    ArenaBlock& b = blocks_[k];
    const auto in = [&](uint64_t at) { return at >= b.at && at < b.at + b.bytes; };
    if (b.at < cache_at_) {
        for (size_t i = 0; i < pins_.size(); i++)
            if (in(pin_at_[i])) {
                const auto [l, e] = pins_[i];
                slot_[(size_t)l * experts_ + e] = -1;
                point(l, e, staging_, nullptr);
            }
    } else {
        for (size_t i = 0; i < coff_.size(); i++)
            if (in(coff_[i])) {
                const auto l = (uint32_t)(i / lru_);
                if (cexp_[i] >= 0) cof_[(size_t)l * experts_ + cexp_[i]] = -1, point(l, cexp_[i], staging_, nullptr);
                cexp_[i] = -2;
            }
    }
    residency(b.buf, false);
    b.buf = nil, b.vol = false, b.used = 0;
    held_ -= (int64_t)b.bytes;
}

void Arena::read_block(size_t k) {
    ArenaBlock& b = blocks_[k];
    b.buf = [dev_->dev() newBufferWithLength:b.bytes options:MTLResourceStorageModeShared];
    if (!b.buf) throw std::runtime_error("Metal arena: cannot allocate a block again");
    b.vol = false;
    held_ += (int64_t)b.bytes;
    residency(b.buf, true);
    const auto in = [&](uint64_t at) { return at >= b.at && at < b.at + b.bytes; };
    if (b.at >= cache_at_) {
        for (size_t i = 0; i < coff_.size(); i++)
            if (in(coff_[i])) cexp_[i] = -1, ctick_[i] = 0;
        return;
    }
    read_pins(k);
}

void Arena::read_pins(size_t k) {
    const ArenaBlock& b = blocks_[k];
    const auto in = [&](uint64_t at) { return at >= b.at && at < b.at + b.bytes; };
    Jobs jobs;
    for (size_t i = 0; i < pins_.size(); i++) {
        if (!in(pin_at_[i])) continue;
        const auto [l, e] = pins_[i];
        if (int16_t& c = cof_[(size_t)l * experts_ + e]; c >= 0) cexp_[(size_t)l * lru_ + c] = -1, c = -1;
        slot_[(size_t)l * experts_ + e] = (int64_t)pin_at_[i];
        point(l, e, pin_at_[i], &jobs);
        if (!pin_ok_.empty()) pin_ok_[i] = 1;
    }
    read_pieces(jobs);
}

void Arena::residency(id<MTLBuffer> b, bool add) {
    auto* s = (id<MTLResidencySet>)rset_;
    if (add)
        [s addAllocation:b];
    else
        [s removeAllocation:b];
    [s commit];
}

}  // namespace tona::metal
