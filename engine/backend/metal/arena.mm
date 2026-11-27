#include "backend/metal/arena.h"

#include <dispatch/dispatch.h>
#include <fcntl.h>
#include <mach/mach.h>
#include <sys/sysctl.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <utility>

#include "backend/budget.h"
#include "backend/env.h"
#include "backend/expert_profile.h"
#include "backend/expert_tiers.h"
#include "backend/metal/host_math.h"
#include "backend/metal/weights.h"
#include "backend/priority.h"

namespace tona::metal {

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

uint64_t round_up(uint64_t n, uint64_t a) { return (n + a - 1) / a * a; }

// The file without the page cache: a cold read must not evict the arena's neighbours or the rest of the system, and
// the read-ahead of a fault brought in pages evicted before use
// (research/measurements/2026-11-15-mac-coder-baseline.md, "The cold read path on macOS"). Aligned reads take xnu's
// direct path.
int open_nocache(const std::string& path) {
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error("Metal arena: cannot open " + path);
    fcntl(fd, F_NOCACHE, 1);
    fcntl(fd, F_RDAHEAD, 0);
    return fd;
}

}  // namespace

namespace {

// The arena's budget in bytes, -1: auto, 0: no arena. Without TONA_METAL_ARENA_GB auto when the weights the GPU reads
// (all but the n-gram table) exceed the working set: in the page cache the Coder re-read 23 to 43 GB per pass and
// pushed other programs' memory to the compressor (research/measurements/2026-11-15-mac-coder-baseline.md)
int64_t arena_request(Device& dev, const Qwen4ExpModel& m) {
    const char* v = env::text("TONA_METAL_ARENA_GB");
    if (v && *v) {
        if (std::strcmp(v, "auto") == 0) return -1;
        return std::max<int64_t>(0, (int64_t)(env::number("TONA_METAL_ARENA_GB", 0.0) * 1e9));
    }
    uint64_t gpu = 0;
    for (const TensorInfo& t : m.gguf().tensors())
        if (&t != m.ple_embd()) gpu += t.bytes;
    return gpu + (uint64_t)(backend::RAM_SLACK_GB * 1e9) > dev.dev().recommendedMaxWorkingSetSize ? -1 : 0;
}

}  // namespace

std::unique_ptr<Arena> Arena::create(Device& dev, const Qwen4ExpModel& m) {
    const int64_t budget = arena_request(dev, m);
    if (budget == 0) return nullptr;
    const char* path = env::text("TONA_EXPERT_PROFILE");
    const auto layout = backend::ExpertLayout::uniform(m.hp().n_layer, m.hp().n_expert);
    std::vector<uint32_t> profile =
        path ? backend::read_expert_profile(path, layout, "Metal arena") : std::vector<uint32_t>{};
    auto a = std::unique_ptr<Arena>(new Arena(dev, m, budget, profile));
    if (profile.empty()) {  // no ranking: the pins are arbitrary, the cache of recent experts still works
        std::fprintf(stderr,
                     "Metal arena: no expert profile (TONA_EXPERT_PROFILE, --profile), all experts ranked equal\n");
        a->profile_.assign((size_t)layout.total(), 1);
    }
    a->loaded_ = std::move(profile);
    if (env::text("TONA_EXPERT_PROFILE_OUT")) a->use_.assign((size_t)layout.total(), 0), a->write_on_signal();
    return a;
}

Arena::Arena(Device& dev, const Qwen4ExpModel& m, int64_t budget, std::vector<uint32_t> profile)
    : budget_(budget), profile_(std::move(profile)), layers_(m.hp().n_layer), experts_(m.hp().n_expert) {
    const Gguf& g = m.gguf();
    for (int f = 0; f < g.files(); f++) fds_.push_back(open_nocache(g.file_path(f)));
    const auto page = (uint64_t)getpagesize();
    pieces_.resize((size_t)layers_ * 3 * experts_);
    span_.assign(layers_, 0);
    mslot_.assign((size_t)layers_ * 3, 0);
    for (uint32_t l = 0; l < layers_; l++) {
        const Layer& L = m.layers()[l];
        const T ts[3] = {L.gate_exps, L.up_exps, L.down_exps};
        for (int k = 0; k < 3; k++) {
            const Mat s = Weights::shape(*ts[k]);
            if (s.experts != experts_) throw std::runtime_error("Metal arena: " + ts[k]->name + ": expert count");
            where_[ts[k]] = {l, k};
            mslot_[(size_t)l * 3 + k] = round_up(s.expert_bytes, page) + page;
            span_[l] += mslot_[(size_t)l * 3 + k];
            for (uint32_t e = 0; e < experts_; e++) {
                const uint64_t at = g.file_offset(*ts[k]) + (uint64_t)e * s.expert_bytes;
                ExpertPiece& p = pieces_[((size_t)l * 3 + k) * experts_ + e];
                p.fd = fds_[ts[k]->file];
                p.file_at = at / page * page;
                p.bytes = round_up(at + s.expert_bytes, page) - p.file_at;
                p.skew = at - p.file_at;
            }
        }
    }
    tab_ = [dev.dev() newBufferWithLength:(NSUInteger)layers_ * 3 * experts_ * 8 options:MTLResourceStorageModeShared];
    if (!tab_) throw std::runtime_error("Metal arena: cannot allocate the address table");
}

void Arena::load(Device& dev, uint64_t mapped, uint64_t gpu) {
    dev_ = &dev;
    const bool steps = env::flag("TONA_METAL_ARENA_GOVERN");
    purge_ = steps && env::flag("TONA_METAL_ARENA_PURGEABLE");
    plan(budget_ >= 0 ? budget_ : auto_budget(dev, mapped, gpu));
    allocate(dev, !steps);
    fill();
    if (steps) fill_in_steps();
    purge_order();
    start_governor();
}

namespace {

void print_stats(const ArenaStats& s) {
    const double n = s.stops ? (double)s.stops : 1.0;
    std::fprintf(stderr,
                 "Metal arena: %llu stops (%llu with cold experts), per stop: wait %.3f ms (GPU %.3f), read %.3f ms, "
                 "%.2f experts, %.2f MB; %.2f GB read in %.2f s (%.2f GB/s); ahead: predicted %llu, routed %llu, both "
                 "%llu\n",
                 (unsigned long long)s.stops, (unsigned long long)s.layers, s.waited * 1e3 / n, s.gpu * 1e3 / n,
                 s.seconds * 1e3 / n, (double)s.experts / n, (double)s.bytes / 1e6 / n, (double)s.bytes / 1e9,
                 s.seconds, s.seconds > 0 ? (double)s.bytes / 1e9 / s.seconds : 0.0, (unsigned long long)s.predicted,
                 (unsigned long long)s.routed, (unsigned long long)s.useful);
    if (s.settled)
        std::fprintf(stderr,
                     "Metal arena: purgeable blocks made volatile %llu, non-volatile %llu, found purged %llu; experts "
                     "read again %llu; wakes %.2f s\n",
                     (unsigned long long)s.settled, (unsigned long long)s.woken, (unsigned long long)s.purged,
                     (unsigned long long)s.refilled, s.wake_s);
}

}  // namespace

// Physical memory minus what other programs hold and what the engine needs besides the arena. Taken at start only:
// a program that grows later pushes the arena's pages to the compressor (shrinking under memory pressure is the next
// step).
// The free RAM the budget leaves, R of the governor's thresholds (RamGovernorConfig::from_reserve). With TONA_RAM_GB
// the user's share is already outside the limit (the installer asks how much RAM the user keeps); without it (a
// developer start) TONA_METAL_ARENA_FREE_GB, default 2. TIERED_CACHE_MIN_GB is no page-cache floor here (cold reads
// use F_NOCACHE) but room for what passes wire beyond the idle page counts; TONA_PRIORITY decides who gets it
// (backend::pass_room_gb)
double Arena::reserve_gb() {
    const double user_gb = backend::ram_limit_gb() >= 0 ? 0.0 : env::number("TONA_METAL_ARENA_FREE_GB", 2.0);
    const double pass_room_gb = backend::pass_room_gb(backend::priority());
    return backend::RAM_SLACK_GB + backend::prompt_reserve_gb() + pass_room_gb + user_gb;
}

int64_t Arena::auto_budget(Device& dev, uint64_t mapped, uint64_t gpu) const {
    vm_statistics64_data_t vm{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    vm_size_t page = 0;
    uint64_t ram = 0;
    size_t len = sizeof ram;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
            KERN_SUCCESS ||
        host_page_size(mach_host_self(), &page) != KERN_SUCCESS || sysctlbyname("hw.memsize", &ram, &len, nullptr, 0))
        throw std::runtime_error("Metal arena: cannot read the memory statistics");
    const double keep_gb = reserve_gb();
    MemNow n;
    // Our buffers count whole: untouched (the KV cache before the first long prompt) they are not in the page counts
    const auto all = (int64_t)(((uint64_t)vm.wire_count + vm.internal_page_count + vm.compressor_page_count) * page);
    n.ram = ram, n.taken = (uint64_t)std::max<int64_t>(0, all - backend::ram_held_bytes()) + (gpu - mapped);
    n.mapped = mapped, n.keep = (uint64_t)(keep_gb * 1e9);
    n.working_set = dev.dev().recommendedMaxWorkingSetSize, n.gpu = gpu;
    n.slack = (uint64_t)(backend::RAM_SLACK_GB * 1e9), n.max_buffer = dev.dev().maxBufferLength;
    const int64_t start = arena_auto_budget(n);
    // Up to the working set, the step fill decides: TONA_METAL_ARENA_PURGEABLE (the blocks are volatile, the system
    // takes them back at once) or TONA_METAL_ARENA_CEILING (test only: without purgeable blocks one such start counted
    // 15.6 GB compressed machine-wide, research/53-mac-purgeable-arena.md)
    int64_t b = purge_ || (env::flag("TONA_METAL_ARENA_GOVERN") && env::flag("TONA_METAL_ARENA_CEILING"))
                    ? arena_ceiling(n)
                    : start;
    if (backend::ram_limit_gb() >= 0)  // TONA_RAM_GB: what the engine may hold besides the page cache
        b = std::min<int64_t>(
            b, (int64_t)((backend::ram_limit_gb() - backend::RAM_SLACK_GB - backend::prompt_reserve_gb()) * 1e9) -
                   backend::ram_held_bytes());
    std::fprintf(stderr,
                 "Metal arena: budget %.2f GB, at start %.2f GB (RAM %.1f GB, held by all processes %.1f, our weights "
                 "mapped %.1f, kept %.1f; working set %.1f, our buffers %.1f)\n",
                 (double)b / 1e9, (double)std::max<int64_t>(start, 0) / 1e9, (double)ram / 1e9, (double)n.taken / 1e9,
                 (double)mapped / 1e9, keep_gb, (double)n.working_set / 1e9, (double)gpu / 1e9);
    return std::max<int64_t>(b, 0);
}

Arena::~Arena() {
    stop_governor();
    if (ahead_group_) dispatch_group_wait((dispatch_group_t)ahead_group_, DISPATCH_TIME_FOREVER);
    for (int fd : fds_) close(fd);
    if (stats_.stops) print_stats(stats_);
    try {
        write_profile();
    } catch (const std::exception& e) {  // a destructor must not throw; the profile is lost, the exit goes on
        std::fprintf(stderr, "Metal arena: profile not written: %s\n", e.what());
    }
}

// tona-server has no shutdown path: SIGINT or SIGTERM would end it without destructors. With
// TONA_EXPERT_PROFILE_OUT both write the profile between passes (under the pass lock) and then end the process
void Arena::write_on_signal() {
    for (int k = 0; k < 2; k++) {
        const int sig = k ? SIGTERM : SIGINT;
        std::signal(sig, SIG_IGN);
        dispatch_source_t src = dispatch_source_create(DISPATCH_SOURCE_TYPE_SIGNAL, (uintptr_t)sig, 0,
                                                       dispatch_get_global_queue(QOS_CLASS_UTILITY, 0));
        dispatch_source_set_event_handler(src, ^{
          std::lock_guard<std::mutex> hold(mu_);
          write_profile();
          std::_Exit(0);
        });
        dispatch_resume(src);
        sig_src_[k] = src;  // kept: a released source no longer fires
    }
}

// TONA_EXPERT_PROFILE_OUT: at exit the sum of the loaded profile and the uses counted here, as on the PC
// (gpu/mega_init.cpp). Counted at each layer's stop of the arena, every routed expert of every token, prefill included
void Arena::write_profile() const {
    const char* out = env::text("TONA_EXPERT_PROFILE_OUT");
    if (!out || use_.empty()) return;
    std::vector<uint32_t> sum = use_;
    if (loaded_.size() == sum.size())
        for (size_t i = 0; i < sum.size(); i++) sum[i] += loaded_[i];
    if (!backend::write_expert_profile(out, backend::ExpertLayout::uniform(layers_, experts_), sum))
        std::fprintf(stderr, "Metal arena: cannot write the profile %s\n", out);
}

void Arena::note_stop(double waited, double gpu) {
    stats_.stops++, stats_.waited += waited, stats_.gpu += gpu;
    static const int64_t every = env::integer("TONA_METAL_ARENA_LOG", 0);
    if (every > 0 && (int64_t)stats_.stops % every == 0) print_stats(stats_), stats_ = {};
}

namespace {

// Every expert fits: no cache, it would hold RAM that no cold expert uses. Experts the governor frees later are the
// least used and come through the staging slots
int lru_fitting(const std::vector<uint64_t>& span, uint32_t experts, int lru, int64_t budget) {
    uint64_t all_pins = 0, stage_all = 0;
    for (uint64_t s : span) all_pins += (uint64_t)experts * s, stage_all = std::max<uint64_t>(stage_all, experts * s);
    return budget >= (int64_t)(all_pins + stage_all) ? 0 : lru;
}

}  // namespace

// Pins by the shared ranking, with room for the staging slots of the layer with the most cold experts (a prefill
// chunk may route to all of them; with the governor all can be cold). The 70 % floor of the PC
// (backend::MIN_PARTIAL_PIN_FRACTION) does not apply: it compares partial pins with the page cache, which on this
// machine re-reads the whole model every pass.
void Arena::plan(int64_t budget) {
    const auto layout = backend::ExpertLayout::uniform(layers_, experts_);
    std::vector<size_t> bytes(span_.begin(), span_.end());
    // 32: at equal memory the fewest reads per stop (simulation and measurement, 2026-11-15-mac-coder-arena.md)
    lru_ = (int)std::max<int64_t>(0, env::integer("TONA_METAL_ARENA_LRU", 32));
    // 32 MB: about 15 experts of the Coder; the PC's piece size (docs/architecture/expert-tiers.md section 4)
    block_ = round_up(std::max<uint64_t>((uint64_t)env::integer("TONA_METAL_ARENA_BLOCK_MB", 32) << 20,
                                         *std::max_element(span_.begin(), span_.end())),
                      (uint64_t)getpagesize());
    const bool all_cold = env::flag("TONA_METAL_ARENA_GOVERN") || env::integer("TONA_METAL_ARENA_CHURN", 0) > 0;
    lru_ = lru_fitting(span_, experts_, lru_, budget);
    uint64_t cache = 0;
    for (uint32_t l = 0; l < layers_; l++) cache += (uint64_t)lru_ * span_[l];
    int64_t pins = budget - (int64_t)cache;
    std::vector<std::vector<int>> chosen;
    for (int it = 0; it < 8; it++) {
        chosen = backend::select_pinned(layout, profile_, bytes, 0, pins);
        staging_bytes_ = 0;
        int64_t pinned = 0;
        for (uint32_t l = 0; l < layers_; l++) {
            const size_t cold = all_cold ? experts_ : experts_ - chosen[l].size();
            staging_bytes_ = std::max<uint64_t>(staging_bytes_, cold * span_[l]);
            pinned += (int64_t)(chosen[l].size() * span_[l]);
        }
        if (pinned + (int64_t)staging_bytes_ + (int64_t)cache <= budget) break;
        pins -= pinned + (int64_t)staging_bytes_ + (int64_t)cache - budget;
    }
    pins_.clear();
    for (uint32_t l = 0; l < layers_; l++)
        for (int e : chosen[l]) pins_.emplace_back(l, e);
    // Most used first, ties as select_pinned: freeing the last blocks frees the least used experts
    std::stable_sort(pins_.begin(), pins_.end(), [&](const auto& a, const auto& b) {
        return profile_[layout.offset(a.first) + a.second] > profile_[layout.offset(b.first) + b.second];
    });
    size_t n = pins_.size();
    while (n > 0 && (int64_t)lay_out(n) > budget) n -= std::min<size_t>(n, 1 + (lay_out(n) - budget) / span_[0]);
    pins_.resize(n);
    lay_out(n);
    slot_.assign((size_t)layers_ * experts_, -1);
    uint64_t uses = 0, all = 0;
    for (size_t i = 0; i < n; i++) slot_[(size_t)pins_[i].first * experts_ + pins_[i].second] = (int64_t)pin_at_[i];
    for (size_t i = 0; i < profile_.size(); i++) all += profile_[i], uses += slot_[i] >= 0 ? profile_[i] : 0;
    std::fprintf(stderr,
                 "Metal arena: %zu of %zu experts (%.1f %% of the profile's uses) in %.2f GB, cache %d per layer "
                 "(%.2f GB), staging %.0f MB, blocks of %llu MB\n",
                 n, slot_.size(), all ? 100.0 * (double)uses / (double)all : 0.0, (double)cache_at_ / 1e9, lru_,
                 (double)cache / 1e9, (double)staging_bytes_ / 1e6, (unsigned long long)(block_ >> 20));
}

// Offsets of the first n pins, the cache slots and the staging slots, without gaps; a block ends before the expert
// that would make it longer than block_, so no expert straddles two blocks (blocks of a fixed size left 3 % unused
// and cost 1 to 2 tok/s, 2026-11-15-mac-coder-arena.md)
uint64_t Arena::lay_out(size_t n) {
    starts_.clear();
    uint64_t at = 0;
    const auto next = [&](uint64_t span) {
        if (starts_.empty() || at + span - starts_.back() > block_) starts_.push_back(at);
        const uint64_t a = at;
        at += span;
        return a;
    };
    pin_at_.resize(n);
    for (size_t i = 0; i < n; i++) pin_at_[i] = next(span_[pins_[i].first]);
    cache_at_ = at;
    if (lru_ > 0) starts_.push_back(at);  // the cache in blocks of its own
    coff_.assign((size_t)layers_ * lru_, 0);
    for (int s = 0; s < lru_; s++)
        for (uint32_t l = 0; l < layers_; l++) coff_[(size_t)l * lru_ + s] = next(span_[l]);
    staging_ = at;
    size_ = staging_ + staging_bytes_;
    return size_;
}

ArenaBlock& Arena::block(uint64_t at) {
    if (at >= staging_) return blocks_.back();
    return blocks_[(size_t)(std::upper_bound(starts_.begin(), starts_.end(), at) - starts_.begin()) - 1];
}

// The blocks up to the end of their region's slots, the staging slots as one block, all in one residency set:
// without it the first pass that bound the arena waited seconds for it. Not all: the staging slots only, the rest
// comes in steps (fill_in_steps)
void Arena::allocate(Device& dev, bool all) {
    blocks_.clear();
    for (size_t k = 0; k < starts_.size(); k++) {
        const uint64_t end = k + 1 < starts_.size() ? starts_[k + 1] : staging_;
        if (end > starts_[k]) blocks_.push_back({nil, starts_[k], end - starts_[k]});
    }
    starts_.clear();
    for (const ArenaBlock& b : blocks_) starts_.push_back(b.at);
    blocks_.push_back({nil, staging_, std::max<uint64_t>(staging_bytes_, (uint64_t)getpagesize())});
    if (@available(macOS 15.0, *)) {
        MTLResidencySetDescriptor* d = [MTLResidencySetDescriptor new];
        d.initialCapacity = blocks_.size() + 1;
        NSError* err = nil;
        id<MTLResidencySet> s = [dev.dev() newResidencySetWithDescriptor:d error:&err];
        if (!s) throw std::runtime_error("Metal arena: no residency set");
        [s addAllocation:tab_];
        rset_ = s;
        [dev.queue() addResidencySet:s];
    } else {
        throw std::runtime_error("Metal arena: needs macOS 15 (residency sets)");
    }
    for (ArenaBlock& b : blocks_) {
        if (!all && &b != &blocks_.back()) continue;
        b.buf = [dev.dev() newBufferWithLength:b.bytes options:MTLResourceStorageModeShared];
        if (!b.buf) throw std::runtime_error("Metal arena: cannot allocate " + std::to_string(b.bytes) + " bytes");
        held_ += (int64_t)b.bytes;
        [(id<MTLResidencySet>)rset_ addAllocation:b.buf];
    }
    [(id<MTLResidencySet>)rset_ commit];
    [(id<MTLResidencySet>)rset_ requestResidency];
}

uint8_t* Arena::host(uint64_t at) {
    const ArenaBlock& b = block(at);
    return (uint8_t*)b.buf.contents + (at - b.at);
}

uint64_t Arena::gpu(uint64_t at) {
    const ArenaBlock& b = block(at);
    return b.buf.gpuAddress + (at - b.at);
}

void Arena::point(uint32_t l, int e, uint64_t at, Jobs* jobs) {
    for (int k = 0; k < 3; k++) {
        const ExpertPiece& p = piece(l, k, e);
        tab(l, k)[e] = gpu(at) + p.skew;
        if (jobs) jobs->emplace_back(&p, host(at));
        at += mslot_[(size_t)l * 3 + k];
    }
}

// Every resident expert of an allocated block read once; the others point at the first staging slot until a pass
// reads them or their block comes
void Arena::fill() {
    Jobs jobs;
    for (uint32_t l = 0; l < layers_; l++)
        for (uint32_t e = 0; e < experts_; e++) {
            int64_t& s = slot_[(size_t)l * experts_ + e];
            if (s >= 0 && !block((uint64_t)s).buf) s = -1;
            point(l, (int)e, s >= 0 ? (uint64_t)s : staging_, s >= 0 ? &jobs : nullptr);
        }
    cexp_.assign((size_t)layers_ * lru_, -1), ctick_.assign((size_t)layers_ * lru_, 0);
    cof_.assign((size_t)layers_ * experts_, -1);
    for (size_t i = 0; i < coff_.size(); i++)
        if (!block(coff_[i]).buf) cexp_[i] = -2;
    const double t0 = now_s();
    read_pieces(jobs);
    std::fprintf(stderr, "Metal arena: resident experts read in %.1f s\n", now_s() - t0);
}

// Parallel reads, one request per worker of the concurrent queue: 2 MB reads reach 13 GB/s at depth 16 against 3.5 at
// depth 1 (research/measurements/2026-11-11-mac-ssd.md)
void Arena::read_pieces(const Jobs& jobs) {
    const auto* list = jobs.data();
    std::atomic<int> failed_flag{0};
    std::atomic<int>* failed = &failed_flag;
    dispatch_apply(jobs.size(), dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^(size_t i) {
      const ExpertPiece& p = *list[i].first;
      uint8_t* dst = list[i].second;
      for (uint64_t done = 0; done < p.bytes;) {
          const ssize_t r = pread(p.fd, dst + done, p.bytes - done, (off_t)(p.file_at + done));
          if (r <= 0) {
              failed->store(1);
              return;
          }
          done += (uint64_t)r;
      }
    });
    if (failed_flag.load()) throw std::runtime_error("Metal arena: an expert read failed");
}

std::vector<std::string> Arena::tensor_names() const {
    std::vector<std::string> out;
    out.reserve(where_.size());
    for (const auto& [t, lm] : where_) out.push_back(t->name);
    return out;
}

Buf Arena::table(T t) const {
    const auto& [l, k] = where_.at(t);
    return {tab_, ((uint64_t)l * 3 + (uint64_t)k) * experts_ * 8};
}

}  // namespace tona::metal
