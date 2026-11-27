// The never-remap ledger and the VRAM governor state machine (backend/va_ledger.h, backend/vram_governor.h).
#include <cstdint>

#include "backend/va_ledger.h"
#include "backend/vram_governor.h"
#include "tests/test_support.h"

namespace {

using namespace tona::backend;

constexpr int64_t MiB = 1ll << 20;

void test_ledger_never_remaps() {
    VaLedger L;
    EXPECT(L.map(1000, 100));
    EXPECT(!L.map(1050, 100));  // overlaps a live range
    EXPECT(!L.map(950, 100));
    EXPECT(L.is_live(1000, 100) && L.is_live(1010, 20));
    EXPECT(!L.is_live(1000, 101) && !L.is_live(999, 10));
    EXPECT(L.unmap(1000));
    EXPECT(!L.is_live(1000, 1));             // dead: no write path may touch it
    EXPECT(!L.map(1000, 100));               // the same range again: refused
    EXPECT(!L.map(1020, 10));                // a part of it: refused
    EXPECT(!L.map(900, 150));                // one that covers it: refused
    EXPECT(L.map(1100, 100));                // directly behind it: fine
    EXPECT(!L.unmap(1000) && !L.unmap(77));  // twice or never mapped
    EXPECT(L.refused() == 7 && L.live_count() == 1);
}

void test_governor_shrink() {
    VramMarks m;  // R 1024 MiB, step 400 MiB: tight below 512, a release aims at 768, short below 256
    VramGovernor g(m, 10, 2, 20);
    EXPECT(g.sample(900 * MiB, 1, false) == 0);   // above the mark: the reserve absorbs it
    EXPECT(g.sample(400 * MiB, 2, true) == 0);    // a pass runs: never
    EXPECT(g.sample(400 * MiB, 3, false) == -1);  // 368 MiB short of the target: one step
    g.applied(-1);
    EXPECT(g.sample(800 * MiB, 4, false) == 0);   // free is 800 now (target 768): the episode ends
    EXPECT(g.sample(100 * MiB, 5, false) == -2);  // idle: all that is needed at once (668 MiB: two steps)
    g.applied(-2);
    EXPECT(g.alive() == 7);
    EXPECT(g.sample(0, 6, false) == -2);  // 768 MiB: two steps
    g.applied(-2);
    VramGovernor f(m, 3, 2, 20);
    EXPECT(f.sample(1024 * MiB, 0, false) == 0);
    EXPECT(f.sample(0, 1, false) == -1);  // never below the floor of 2 steps
    f.applied(-1);
    EXPECT(f.sample(0, 2, false) == 0);
}

void test_governor_grow_and_hysteresis() {
    VramMarks m;
    VramGovernor g(m, 8, 2, 2);
    g.applied(-3);                                  // alive 5, the ceiling stays 8
    EXPECT(g.sample(2200 * MiB, 100, false) == 0);  // calm only just began
    EXPECT(g.sample(2200 * MiB, 129, false) == 0);
    EXPECT(g.sample(2200 * MiB, 131, false) == 1);  // 31 s calm
    g.applied(1);
    EXPECT(g.sample(2200 * MiB, 133, false) == 0);  // one step per 5 s
    EXPECT(g.sample(2200 * MiB, 137, false) == 1);
    g.applied(1);
    EXPECT(g.budget() == 0);
    EXPECT(g.sample(2200 * MiB, 300, false) == 0);  // the fresh address space is used up
    VramGovernor full(m, 4, 0, 10);                 // a pool that is whole never grows beyond its plan
    EXPECT(full.sample(9000 * MiB, 0, false) == 0 && full.sample(9000 * MiB, 500, false) == 0);
}

void test_governor_calm_doubles() {
    VramMarks m;
    VramGovernor g(m, 8, 2, 100);
    EXPECT(g.sample(1100 * MiB, 0, false) == 0);  // free 1100: inside the band, no growth
    EXPECT(g.sample(400 * MiB, 10, false) == -1);
    g.applied(-1);
    EXPECT(g.sample(2000 * MiB, 20, false) == 0);  // the other program is gone for a moment
    EXPECT(g.sample(400 * MiB, 40, false) == -1);  // a second one within 60 s: calm time doubles
    g.applied(-1);
    EXPECT(g.calm_s() == 60);
    EXPECT(g.sample(2000 * MiB, 100, false) == 0);
    EXPECT(g.sample(2000 * MiB, 150, false) == 0);  // only 50 s calm
    EXPECT(g.sample(2000 * MiB, 161, false) == 1);  // 61 s
}

// A fresh start never begins below the tight mark (the Mac arena governor gave back 6.7 GB on its first request
// because its own untouched buffers were not counted)
void test_fresh_start_inside_the_dead_band() {
    VramMarks m;
    // The plan keeps `reserve` free; 150 MiB of diagnostics and granule alignment afterwards still leaves more than the
    // mark
    EXPECT(free_after_plan(m, 150 * MiB) >= shrink_below(m));
    EXPECT(free_after_plan(m, 600 * MiB) < shrink_below(m));  // the limit: 512 MiB of unplanned allocations
    // A start that is below the mark anyway: no shrink, the reserve is rebased to what is free
    VramGovernor g(m, 4, 0, 10);
    EXPECT(g.sample(400 * MiB, 0, false) == 0 && g.rebased());
    EXPECT(g.sample(390 * MiB, 1, false) == 0);   // inside the new band (mark 200 MiB)
    EXPECT(g.sample(150 * MiB, 2, false) == -1);  // a real drop below the new mark
    // A start inside the band is not rebased
    VramGovernor h(m, 4, 0, 10);
    EXPECT(h.sample(900 * MiB, 0, false) == 0 && !h.rebased());
    VramGovernor k(m, 4, 0, 10);  // a start at exactly R
    EXPECT(k.sample(m.reserve, 0, false) == 0 && !k.rebased());
}

// The invariants of the shared thresholds for the 16 GB card (R = 1 GiB), the same four as the RAM governor's
struct Sim {
    int steps;
    int64_t other_from_s, other_until_s, other;  // another program holds `other` bytes in this window
    int end_alive = 0, changes = 0;
};

int run_sim(const VramMarks& m, Sim& S) {
    const int steps = S.steps;
    const int64_t other_from_s = S.other_from_s, other_until_s = S.other_until_s, other = S.other;
    VramGovernor g(m, steps, 2, 1000);
    int last_dir = 0, actions = 0;
    S.changes = 0;
    int alive = steps;
    for (int t4 = 0; t4 < 4 * 3000; t4++) {
        const double t = t4 * 0.25;
        const int64_t o = (t >= other_from_s && t < other_until_s) ? other : 0;
        const int64_t free = m.reserve + (int64_t)(steps - alive) * m.step_bytes - o;
        const int n = g.sample(free, t, false);
        if (n == 0) continue;
        g.applied(n);
        alive += n, actions++;
        const int dir = n < 0 ? -1 : 1;
        if (last_dir != 0 && dir != last_dir) S.changes++;
        last_dir = dir;
    }
    S.end_alive = alive;
    return actions;
}

void test_invariants_of_the_16gb_card() {
    VramMarks m;
    const RamGovernorConfig c = vram_config(m);
    EXPECT(c.high - c.low > m.step_bytes);            // the dead band is wider than a grow step
    EXPECT(c.low < m.reserve && m.reserve < c.high);  // a fresh start (free = R) is inside it
    Sim none{12, 0, 0, 0};                            // nothing happens without another program
    EXPECT(run_sim(m, none) == 0 && none.end_alive == 12);
    Sim burst{12, 100, 1300, 2048 * MiB};  // a program takes 2 GiB for 20 minutes and leaves
    EXPECT(run_sim(m, burst) > 0);
    EXPECT(burst.end_alive == 12);          // regrown to the full budget after the pressure ended
    EXPECT(burst.changes == 1);             // one shrink phase, one grow phase: no oscillation
    Sim stays{12, 100, 100000, 700 * MiB};  // a program that stays: steady, no change of direction
    run_sim(m, stays);
    EXPECT(stays.end_alive < 12 && stays.end_alive >= 2 && stays.changes == 0);
}

}  // namespace

int main() {
    return tona::test::run("test_vram_governor", [] {
        test_ledger_never_remaps();
        test_governor_shrink();
        test_governor_grow_and_hysteresis();
        test_governor_calm_doubles();
        test_invariants_of_the_16gb_card();
        test_fresh_start_inside_the_dead_band();
    });
}
