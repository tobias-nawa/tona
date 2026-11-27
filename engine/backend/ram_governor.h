// How much RAM the warm expert tier (the PC's pinned experts, the Mac's arena) may hold while other programs grow and
// shrink next to the engine: a pure state machine, docs/architecture/expert-tiers.md section 5 and
// research/44-memory-priority.md. Input every sample: free RAM, the time, whether a pass runs, an OS pressure signal
// (PSI on Linux, the dispatch memory-pressure source on macOS) and the tier's size and budget. Output: a byte delta for
// the tier, < 0 release, > 0 grow. The caller applies it between passes in whole pieces.
//
// The thresholds follow one formula for every machine (RamGovernorConfig::from_reserve): R is the free RAM the start
// budget leaves, the free RAM of an undisturbed start. Measured free RAM is calibrated so that such a start reads R
// (memory the page counts miss belongs to the measurement, not to the thresholds). The fixed defaults below are
// section 5's starting values for the PC; the Mac balloon run showed why they cannot be absolute: on 34 GB a growth
// threshold of 10 GB free kept a shrunk arena small for good (research/measurements/2026-11-15-mac-arena-balloon.md).
#pragma once

#include <algorithm>
#include <cstdint>

namespace tona::backend {

struct RamGovernorConfig {
    int64_t low = 4'000'000'000;             // below: tight, release until `target`
    int64_t target = 6'000'000'000;          // free RAM a release aims at
    int64_t high = 10'000'000'000;           // at or above for `calm_s`: calm, may grow
    int64_t grow_floor = 8'000'000'000;      // a grow step must leave this much free
    int64_t emergency = 2'000'000'000;       // below (or OS pressure): short, emergency steps
    int64_t grow_step = 512'000'000;         // per `grow_every_s`
    int64_t release_step = 260'000'000;      // per gap between passes (8 pieces of the PC)
    int64_t emergency_step = 1'040'000'000;  // per gap in an emergency (32 pieces)
    double grow_every_s = 5, calm_s = 30, calm_max_s = 600;
    double repeat_s = 60;  // a tight event within this time of the last one doubles the calm time

    // From the reserve R: short below R/4, tight below R/2, a release aims at 3R/4; growth starts at R plus a step of
    // free RAM and goes on, a step at a time, while a step leaves R free, so the tier returns to its full budget once
    // the other program is gone. A fresh start (free = R) is in the dead band [R/2, R + step); the band is wider than a
    // step; a tight event frees at least R/4, so its end lifts free RAM over the growth threshold (R >= 2.1 GB)
    static RamGovernorConfig from_reserve(int64_t r) {
        RamGovernorConfig c;
        c.emergency = r / 4, c.low = r / 2, c.target = r * 3 / 4;
        c.high = r + c.grow_step, c.grow_floor = r - c.grow_step;
        return c;
    }
};

// The tier's first fill at start, a growth step at a time instead of the whole budget at once: the next step while it
// leaves the growth floor free and the last one did not make the OS compress memory (`compressed`: bytes compressed
// machine-wide during the last step; more than half a step is the OS making room in other programs' memory, not
// background). The first fill of the Mac arena after another server compressed 29 to 31 GB
// (research/measurements/2026-11-16-mac-dmt-arena-ab.md). What is left, the governor grows when RAM is free. 0: stop
inline int64_t fill_step(const RamGovernorConfig& c, int64_t free, int64_t held, int64_t budget, int64_t compressed) {
    const int64_t step = std::min(c.grow_step, std::max<int64_t>(0, budget - held));
    if (step <= 0 || compressed > c.grow_step / 2 || free - step < c.grow_floor) return 0;
    return step;
}

enum class RamState { calm, band, tight, short_of_memory };

struct RamSample {
    double now_s = 0;
    int64_t free = 0;           // RAM that can be taken without paging out (MemAvailable and cgroup room; macOS: free,
                                // file-backed and purgeable pages)
    bool pass_running = false;  // between passes steps are bounded, while idle a release takes all it needs at once
    bool pressure = false;      // the OS says others stall (PSI "some" of the own cgroup, macOS warn or critical)
    int64_t held = 0;           // bytes in the tier now
    int64_t budget = 0;         // bytes the tier may hold at most (the start budget)
};

class RamGovernor {
public:
    explicit RamGovernor(RamGovernorConfig c = {}) : c_(c), calm_wait_(c.calm_s) {}

    int64_t step(const RamSample& s) {
        state_ = classify(s);
        // A tight episode goes on until `target`: between passes it frees in small steps, and stopping at `low` would
        // leave free RAM just under the growth threshold once the other program is gone
        if (state_ == RamState::band && in_tight_ && s.free < c_.target && s.held > 0) state_ = RamState::tight;
        if (state_ == RamState::tight || state_ == RamState::short_of_memory) return release(s);
        if (state_ == RamState::band && !growing_) {
            calm_since_ = -1;
            return 0;
        }
        return grow(s);  // calm, or the band during a growth that began calm
    }

    RamState state() const { return state_; }
    double calm_wait_s() const { return calm_wait_; }

private:
    RamState classify(const RamSample& s) const {
        if (s.pressure || s.free < c_.emergency) return RamState::short_of_memory;
        if (s.free < c_.low) return RamState::tight;
        return s.free >= c_.high ? RamState::calm : RamState::band;
    }

    // Entering a tight episode within repeat_s of the last one doubles the calm time (build bursts repeat); a long
    // calm stretch resets it
    int64_t release(const RamSample& s) {
        if (!in_tight_) {
            if (last_tight_ >= 0 && s.now_s - last_tight_ <= c_.repeat_s)
                calm_wait_ = std::min(calm_wait_ * 2, c_.calm_max_s);
            in_tight_ = true;
        }
        last_tight_ = s.now_s, calm_since_ = -1, growing_ = false;
        const int64_t need = std::max<int64_t>(0, c_.target - s.free);
        const int64_t cap =
            !s.pass_running ? need : (state_ == RamState::short_of_memory ? c_.emergency_step : c_.release_step);
        return -std::min({need, cap, std::max<int64_t>(0, s.held)});
    }

    int64_t grow(const RamSample& s) {
        in_tight_ = false;
        if (!growing_) {
            if (calm_since_ < 0) calm_since_ = s.now_s;
            if (last_tight_ >= 0 && s.now_s - last_tight_ > c_.calm_max_s) calm_wait_ = c_.calm_s;
            growing_ = s.now_s - calm_since_ >= calm_wait_ && (last_tight_ < 0 || s.now_s - last_tight_ >= calm_wait_);
            if (!growing_) return 0;
        }
        const bool paced = last_grow_ < 0 || s.now_s - last_grow_ >= c_.grow_every_s;
        const int64_t room = std::max<int64_t>(0, s.budget - s.held);
        const int64_t step = std::min(c_.grow_step, room);
        if (!paced || step <= 0 || s.free - step < c_.grow_floor) return 0;
        last_grow_ = s.now_s;
        return step;
    }

    RamGovernorConfig c_;
    RamState state_ = RamState::band;
    double calm_wait_;
    double calm_since_ = -1, last_tight_ = -1, last_grow_ = -1;
    bool in_tight_ = false, growing_ = false;
};

}  // namespace tona::backend
