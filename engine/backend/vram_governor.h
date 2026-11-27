// When the elastic slot pool gives VRAM back and takes it again (docs/architecture/expert-tiers.md, section 5a). Pure
// state machine: free VRAM and the time in, a number of steps out; the engine applies a step between two passes. A step
// is the same number of slots in every layer.
#pragma once

#include <algorithm>
#include <cstdint>

#include "backend/ram_governor.h"

namespace tona::backend {

struct VramMarks {
    int64_t reserve = 1024ll << 20;    // R: free VRAM that the plan keeps for the desktop (TONA_VRAM_RESERVE_MB)
    int64_t step_bytes = 400ll << 20;  // memory returned by one step (all layers); also the grow step of the thresholds
    double calm_s = 30, calm_max_s = 600, grow_gap_s = 5, doubling_window_s = 60;
};

// The thresholds are the ones of the RAM governor, one formula of R (RamGovernorConfig::from_reserve): short below R/4,
// tight below R/2, a release aims at 3R/4, growth starts at R plus a step and goes on to the full pool. Not measured on
// a loaded card yet (expert-tiers.md 5a).
inline RamGovernorConfig vram_config(const VramMarks& m) {
    RamGovernorConfig c = RamGovernorConfig::from_reserve(m.reserve);
    c.grow_step = m.step_bytes;
    // The default 512 MB step of the RAM governor is a Mac figure; a VRAM step is what the pool returns in one piece
    c.high = m.reserve + m.step_bytes, c.grow_floor = m.reserve - m.step_bytes;
    c.calm_s = m.calm_s, c.calm_max_s = m.calm_max_s, c.grow_every_s = m.grow_gap_s, c.repeat_s = m.doubling_window_s;
    return c;
}

// A start that is planned with the reserve begins with that much free VRAM; what is allocated after the plan
// (diagnostics, lazily touched buffers) and the alignment of the blocks eat into it. It must stay above the tight mark,
// or the first sample would give back memory that nothing took: the same mistake as the Mac arena governor's first
// request.
inline int64_t free_after_plan(const VramMarks& m, int64_t allocated_after_plan) {
    return m.reserve - allocated_after_plan;
}
inline int64_t shrink_below(const VramMarks& m) { return vram_config(m).low; }

// A wrapper that counts in steps (slots of all layers) around the shared RamGovernor, which counts in bytes.
class VramGovernor {
public:
    // steps_alive: steps that are mapped now; steps_floor: the pool never goes below this; steps_budget: steps that can
    // still be grown with fresh address space (the never-remap rule makes this finite). The pool never grows beyond the
    // steps it started with: more slots than planned would take the reserve.
    VramGovernor(const VramMarks& m, int steps_alive, int steps_floor, int steps_budget)
        : m_(m),
          gov_(vram_config(m)),
          alive_(steps_alive),
          floor_(steps_floor),
          budget_(steps_budget),
          ceiling_(steps_alive) {}

    // Result: < 0 shrink by that many steps, > 0 grow by that many (always 1), 0 nothing. Never acts while a pass runs.
    // The caller reports what it did with applied(), also when it could do less than asked.
    int sample(int64_t free_bytes, double now_s, bool pass_running) {
        if (pass_running) return 0;
        // The first sample is the start: if it is already below the tight mark, memory that was meant to stay free is
        // spent on something of our own, not taken by another program. The reserve becomes what is there, the governor
        // starts inside its dead band.
        if (!seen_) {
            seen_ = true;
            if (free_bytes < shrink_below(m_)) {
                m_.reserve = free_bytes, rebased_ = true;
                gov_ = RamGovernor(vram_config(m_));
            }
        }
        RamSample s;
        s.now_s = now_s, s.free = free_bytes, s.pass_running = false;
        // held is what can still be given back (above the floor), budget the most it may hold: no room to grow without
        // fresh address space
        const int64_t step = m_.step_bytes;
        s.held = (int64_t)(alive_ - floor_) * step;
        s.budget = s.held + (budget_ > 0 ? (int64_t)std::max(0, ceiling_ - alive_) * step : 0);
        const int64_t d = gov_.step(s);
        if (d < 0) return -(int)std::min<int64_t>((-d + step - 1) / step, alive_ - floor_);
        return d > 0 ? 1 : 0;
    }

    void applied(int steps) {
        alive_ += steps;
        if (steps > 0) budget_ -= steps;
    }

    // The pool could not grow (address space used up or refused): no more tries
    void no_budget() { budget_ = 0; }
    bool rebased() const { return rebased_; }
    int alive() const { return alive_; }
    int budget() const { return budget_; }
    double calm_s() const { return gov_.calm_wait_s(); }

private:
    VramMarks m_;
    RamGovernor gov_;
    int alive_, floor_, budget_, ceiling_;
    bool seen_ = false, rebased_ = false;
};

}  // namespace tona::backend
