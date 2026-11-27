// backend/ram_governor.h: states, step bounds, calm time and its doubling, and a balloon trace without oscillation
#include <algorithm>
#include <cstdint>

#include "backend/ram_governor.h"
#include "tests/test_support.h"

using tona::backend::RamGovernor;
using tona::backend::RamSample;
using tona::backend::RamState;

#define EXPECT(c) tona::test::check((c), __FILE__, __LINE__, #c)

namespace {

constexpr int64_t GB = 1'000'000'000, MB = 1'000'000;

RamSample at(double t, int64_t free, bool pass = true, int64_t held = 10 * GB) {
    RamSample s;
    s.now_s = t, s.free = free, s.pass_running = pass, s.held = held, s.budget = 12 * GB;
    return s;
}

void dead_band() {
    RamGovernor g;
    for (int i = 0; i < 100; i++) EXPECT(g.step(at(i, 7 * GB)) == 0);
    EXPECT(g.state() == RamState::band);
}

// Between passes a bounded step, while idle all that is missing up to the target, never more than the tier holds
void release_steps() {
    RamGovernor g;
    EXPECT(g.step(at(0, 3 * GB)) == -260 * MB);
    EXPECT(g.state() == RamState::tight);
    EXPECT(g.step(at(1, 3 * GB, false)) == -3 * GB);
    EXPECT(g.step(at(2, 1 * GB)) == -1040 * MB);
    EXPECT(g.state() == RamState::short_of_memory);
    EXPECT(g.step(at(3, 1 * GB, true, 100 * MB)) == -100 * MB);
    RamSample p = at(4, 7 * GB);
    p.pressure = true;  // the OS signal alone: short, but the target is reached already
    EXPECT(g.step(p) == 0);
    EXPECT(g.state() == RamState::short_of_memory);
}

// Calm for 30 s before the first grow, then 512 MB every 5 s, up to the budget, never below the grow floor
void growth() {
    RamGovernor g;
    int64_t held = 10 * GB, grown = 0;
    for (int t = 0; t <= 60; t++) {
        const int64_t d = g.step(at(t, 11 * GB, true, held));
        if (d > 0 && t < 30) EXPECT(false);
        held += d, grown += d;
    }
    EXPECT(grown == 2 * GB);  // 30, 35, 40, 45 s, then the budget of 12 GB is full
    RamGovernor f;
    for (int t = 0; t <= 60; t++) EXPECT(f.step(at(t, 8 * GB + 400 * MB, true, 0)) == 0);  // band
    RamGovernor h;
    int64_t sum = 0;
    for (int t = 0; t <= 60; t++) sum += h.step(at(t, 10 * GB, true, 0));
    EXPECT(sum == 7 * (512 * MB));  // 30 to 60 s every 5 s
}

// A tight event within 60 s of the last doubles the calm time, up to 10 minutes; a long calm resets it
void hysteresis() {
    RamGovernor g;
    double t = 0;
    for (int k = 0; k < 6; k++) {
        g.step(at(t, 3 * GB));
        t += 40;
        g.step(at(t, 11 * GB));
        t += 10;
    }
    EXPECT(g.calm_wait_s() == 600);
    g.step(at(t + 700, 11 * GB));
    EXPECT(g.calm_wait_s() == 30);
}

// Balloon: 20 GB for 20 s every 90 s next to a tier that frees what it releases. One release episode and no grow
// while the balloon is up, per cycle
void balloon() {
    RamGovernor g;
    int64_t held = 12 * GB;
    int episodes = 0;
    bool releasing = false;
    for (int t = 0; t < 600; t++) {
        const bool up = t % 90 >= 30 && t % 90 < 50;
        const int64_t free = 30 * GB - held - (up ? 20 * GB : 0);
        const int64_t d = g.step(at(t, free, true, held));
        if (d < 0 && !releasing) episodes++;
        releasing = d < 0;
        if (up) EXPECT(d <= 0);
        held += d;
    }
    EXPECT(episodes >= 1 && episodes <= 7);  // at most one per balloon cycle
    EXPECT(held >= 0 && held <= 12 * GB);
}

// A machine for the invariants of RamGovernorConfig::from_reserve: RAM, what the rest holds, the reserve R the start
// budget leaves; the tier starts full (free = R)
struct Machine {
    int64_t ram, others, reserve;
    int64_t budget() const { return ram - others - reserve; }
};

// The configurations the invariants must hold for: 16, 32 and 64 GB with the installer's limit (R 3 GB: slack,
// prompts, page-cache floor), the M1 Max with the Coder (34.4 GB, about 16 GB besides the arena, R 5 GB: plus the
// developer's 2 GB)
const Machine MACHINES[] = {
    {16 * GB, 9 * GB, 3 * GB}, {32 * GB, 14 * GB, 3 * GB}, {64 * GB, 20 * GB, 3 * GB}, {34'400 * MB, 16 * GB, 5 * GB}};

struct Run {
    int64_t held, shrinks = 0, grows = 0, min_held;
    bool band_at_start = true;
};

// Every 0.25 s for `seconds`, `foreign(t)` bytes of another program; counts shrink and grow episodes
template <class F>
Run simulate(const Machine& m, double seconds, F foreign) {
    RamGovernor g(tona::backend::RamGovernorConfig::from_reserve(m.reserve));
    Run r{m.budget(), 0, 0, m.budget()};
    int last = 0;
    for (int i = 0; i < (int)(seconds * 4); i++) {
        const double t = i * 0.25;
        RamSample s = at(t, m.ram - m.others - foreign(t) - r.held, true, r.held);
        s.budget = m.budget();
        const int64_t d = g.step(s);
        if (t < 60 && g.state() != RamState::band) r.band_at_start = false;
        const int sign = d < 0 ? -1 : d > 0 ? 1 : 0;
        if (sign && sign != last) (sign < 0 ? r.shrinks : r.grows)++;
        if (sign) last = sign;
        r.held += d, r.min_held = std::min(r.min_held, r.held);
    }
    return r;
}

void reserve_invariants() {
    for (const Machine& m : MACHINES) {
        const auto c = tona::backend::RamGovernorConfig::from_reserve(m.reserve);
        EXPECT(c.high - c.low > c.grow_step);  // (c) the dead band is wider than a growth step
        const Run fresh = simulate(m, 120, [](double) { return int64_t{0}; });
        EXPECT(fresh.band_at_start && fresh.shrinks == 0 && fresh.grows == 0);  // (a)
        // (b) another program takes 60 % of R for a minute: the tier shrinks, then returns to its full budget
        const Run burst = simulate(m, 900, [&](double t) { return t < 60 ? m.reserve * 6 / 10 : 0; });
        EXPECT(burst.min_held < m.budget() && burst.held == m.budget());
        EXPECT(burst.shrinks == 1 && burst.grows == 1);
        // (d) the same program stays: one shrink, no growth, no pendulum
        const Run steady = simulate(m, 1200, [&](double) { return m.reserve * 6 / 10; });
        EXPECT(steady.shrinks == 1 && steady.grows == 0 && steady.held < m.budget());
    }
}

// The first fill in steps: to the full budget when the machine has the room the budget assumed; when another program
// holds 2 GB more, it stops with the growth floor free; and at once when a step made the OS compress
int64_t fill(const Machine& m, int64_t extra, int64_t compressed_per_step) {
    const auto c = tona::backend::RamGovernorConfig::from_reserve(m.reserve);
    int64_t held = 0, compressed = 0;
    while (true) {
        const int64_t d = tona::backend::fill_step(c, m.ram - m.others - extra - held, held, m.budget(), compressed);
        if (d == 0) return held;
        held += d, compressed = compressed_per_step;
    }
}

void fill_steps() {
    for (const Machine& m : MACHINES) {
        const auto c = tona::backend::RamGovernorConfig::from_reserve(m.reserve);
        EXPECT(fill(m, 0, 0) == m.budget());
        const int64_t partial = fill(m, 2 * GB, 0);
        EXPECT(partial < m.budget() && m.ram - m.others - 2 * GB - partial >= c.grow_floor);
        EXPECT(fill(m, 0, c.grow_step) == c.grow_step);     // the first step compressed: no second
        EXPECT(fill(m, 0, c.grow_step / 4) == m.budget());  // background compression does not stop it
    }
}

}  // namespace

int main() {
    return tona::test::run("ram_governor", [] {
        dead_band();
        release_steps();
        growth();
        hysteresis();
        balloon();
        reserve_invariants();
        fill_steps();
    });
}
