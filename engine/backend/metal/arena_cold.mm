// The arena's cold path: the cache of recently read experts, the lookahead and the reads at a layer's stop
#include <dispatch/dispatch.h>

#include <chrono>
#include <cstdio>
#include <stdexcept>

#include "backend/env.h"
#include "backend/metal/arena.h"

namespace tona::metal {

namespace {

double now_s() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

}  // namespace

uint64_t Arena::place(uint32_t l, int e, bool keep, uint64_t& staging, bool& read) {
    const size_t c0 = (size_t)l * lru_;
    int16_t& at = cof_[(size_t)l * experts_ + e];
    read = at < 0;
    if (at < 0 && keep && lru_ > 0) {
        int v = -1;  // a slot not used by this stop, the least recently used
        for (int k = 0; k < lru_; k++)
            if (cexp_[c0 + k] != -2 && ctick_[c0 + k] < tick_ && (v < 0 || ctick_[c0 + k] < ctick_[c0 + v])) v = k;
        if (v >= 0) {
            if (cexp_[c0 + v] >= 0) cof_[(size_t)l * experts_ + cexp_[c0 + v]] = -1;
            cexp_[c0 + v] = e, at = (int16_t)v;
        }
    }
    if (at < 0) {
        const uint64_t s = staging;
        staging += span_[l];
        return s;
    }
    ctick_[c0 + at] = tick_;
    return coff_[c0 + at];
}

void Arena::finish_ahead() {
    if (!ahead_group_) return;
    dispatch_group_wait((dispatch_group_t)ahead_group_, DISPATCH_TIME_FOREVER);
    ahead_group_ = nil;
    if (ahead_failed_.exchange(0)) throw std::runtime_error("Metal arena: an expert read failed");
}

void Arena::prefetch(uint32_t l, const int32_t* ids, int n) {
    finish_ahead();
    tick_++, ticked_ = true;
    ahead_.clear();
    ahead_set_.assign(experts_, 0);
    ahead_layer_ = l;
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (e < 0 || (uint32_t)e >= experts_ || ahead_set_[e] || slot_[(size_t)l * experts_ + e] >= 0) continue;
        ahead_set_[e] = 1;
        stats_.predicted++;
        if (cof_[(size_t)l * experts_ + e] >= 0) continue;  // cached already
        uint64_t none = 0;
        bool read = true;
        const uint64_t at = place(l, e, true, none, read);
        if (cof_[(size_t)l * experts_ + e] < 0) continue;  // no free slot: the stop reads it if it is routed
        point(l, e, at, &ahead_);
    }
    if (ahead_.empty()) return;
    dispatch_group_t g = dispatch_group_create();
    ahead_group_ = g;
    dispatch_group_async(g, dispatch_get_global_queue(QOS_CLASS_USER_INITIATED, 0), ^{
      try {
          read_pieces(ahead_);
      } catch (...) {
          ahead_failed_.store(1);
      }
    });
}

void Arena::read_cold(uint32_t l, const int32_t* ids, int n, bool keep) {
    const double t0 = now_s();
    finish_ahead();
    wake(l, ids, n);  // before the GPU reads them: a volatile block's read is undefined
    Jobs jobs;
    std::vector<uint8_t> seen(experts_, 0);
    uint64_t staging = staging_, bytes = 0;
    if (!ticked_) tick_++;
    ticked_ = false;
    const bool predicted = ahead_layer_ == l;
    ahead_layer_ = UINT32_MAX;
    for (int i = 0; i < n; i++) {
        const int e = ids[i];
        if (!use_.empty() && e >= 0 && (uint32_t)e < experts_) use_[(size_t)l * experts_ + e]++;
        if (e < 0 || (uint32_t)e >= experts_ || seen[e] || slot_[(size_t)l * experts_ + e] >= 0) continue;
        seen[e] = 1;
        if (predicted) stats_.routed++, stats_.useful += ahead_set_[e];
        bool read = true;
        point(l, e, place(l, e, keep, staging, read), read ? &jobs : nullptr);
    }
    for (const auto& [p, dst] : jobs) bytes += p->bytes;
    trace(l, ids, n, seen);
    if (jobs.empty()) return;
    read_pieces(jobs);
    stats_.layers++, stats_.experts += jobs.size() / 3, stats_.bytes += bytes, stats_.seconds += now_s() - t0;
}

// TONA_METAL_ARENA_TRACE=file: per stop the layer, the distinct routed experts and which of them were cold, for
// offline cache simulations
void Arena::trace(uint32_t l, const int32_t* ids, int n, const std::vector<uint8_t>& cold) {
    static std::FILE* f = [] {
        const char* p = env::text("TONA_METAL_ARENA_TRACE");
        return p ? std::fopen(p, "w") : nullptr;
    }();
    if (!f) return;
    std::vector<uint8_t> routed(experts_, 0);
    std::fprintf(f, "%u", l);
    for (int i = 0; i < n; i++)
        if (ids[i] >= 0 && (uint32_t)ids[i] < experts_ && !routed[ids[i]])
            routed[ids[i]] = 1, std::fprintf(f, " %d%s", ids[i], cold[ids[i]] ? "c" : "");
    std::fputc('\n', f);
}

}  // namespace tona::metal
