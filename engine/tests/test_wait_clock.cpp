// Tests for the time limits of the waits in the megakernel (gpu/wait_clock.h): gaps between polls, cap, deadline
#include <cstdio>

#include "gpu/wait_clock.h"

static int g_fail = 0;
#define EXPECT(c)                                                                       \
    do {                                                                                \
        if (!(c)) {                                                                     \
            fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                                   \
        }                                                                               \
    } while (0)

using namespace tona::gpu;

namespace {

constexpr uint64_t kUs = 100;  // wall_clock64 ticks per us on gfx1200 (100 MHz)
constexpr uint64_t kMs = 1000 * kUs;

// Limit 1,000 ms, deadline at 1,500 ms, 2 ms gap threshold, cap 5 s, as in a default launch
WaitLimits limits(uint64_t deadline = 1500 * kMs) {
    return {1000 * kMs, deadline, 2 * kMs, 5000 * kMs, 0, 0, (unsigned)kUs, 0, nullptr, nullptr};
}

void test_gap_below_and_above_freeze() {
    const WaitLimits w = limits();
    WaitClock c(w, 10 * kMs);
    EXPECT(c.poll(10 * kMs + 50 * kUs, w) == 50 * kUs);  // a normal poll: nothing moves
    EXPECT(c.t0 == 10 * kMs && c.dl == w.deadline);
    EXPECT(c.poll(10 * kMs + 50 * kUs + 2 * kMs, w) == 2 * kMs);  // exactly the threshold: still a poll
    EXPECT(c.t0 == 10 * kMs);
    const uint64_t now = 10 * kMs + 50 * kUs + 2 * kMs + 1200 * kMs;  // a wave that did not run for 1.2 s
    EXPECT(c.poll(now, w) == 1200 * kMs);
    EXPECT(c.t0 == 10 * kMs + 1200 * kMs && c.dl == 1500 * kMs + 1200 * kMs);
    EXPECT(c.expired(now, w) == 0);  // 1.2 s of freeze do not count as waiting
}

void test_hang_still_found() {
    const WaitLimits w = limits();
    WaitClock c(w, 0);
    uint64_t now = 0;
    for (int i = 0; i < 1000; i++) c.poll(now += 1000 * kUs, w);  // polls every ms, 1,000 ms in all
    EXPECT(c.expired(now, w) == 0);
    c.poll(now += 1000 * kUs, w);
    EXPECT(c.expired(now, w) == 1);  // limit passed
}

void test_freeze_off() {
    WaitLimits w = limits();
    w.freeze_ticks = ~0ull;
    WaitClock c(w, 0);
    c.poll(1200 * kMs, w);
    EXPECT(c.t0 == 0);
    EXPECT(c.expired(1200 * kMs, w) == 1);  // the gap counts as waiting, as before the fix
}

void test_cap() {
    const WaitLimits w = limits();
    WaitClock c(w, 0);
    uint64_t now = 0;
    for (int i = 0; i < 5; i++) c.poll(now += 900 * kMs, w);  // five freezes of 0.9 s: 4.5 s of wall time
    EXPECT(c.expired(now, w) == 0);                           // the shifts keep the wait alive
    c.poll(now += 900 * kMs, w);
    EXPECT((c.expired(now, w) & 16) != 0);  // 5.4 s of wall time: the cap ends it whatever the shifts add up to
}

void test_deadline() {
    const WaitLimits w = limits();
    WaitClock c(w, 1450 * kMs);
    EXPECT(c.expired(1499 * kMs, w) == 0);  // before the deadline
    EXPECT(c.expired(1520 * kMs, w) ==
           0);  // behind the deadline, but the wait has lasted 70 ms and the pass is 20 ms late
    EXPECT(c.expired(1560 * kMs, w) == 2);  // the wait has lasted over 100 ms
    WaitClock s(w, 1700 * kMs);             // a wait that starts 200 ms behind the deadline
    EXPECT(s.expired(1710 * kMs, w) == 0);  // short wait, 210 ms late: let through
    EXPECT(s.expired(1800 * kMs, w) == 0);  // exactly 300 ms late and 100 ms of waiting: let through
    EXPECT(s.expired(1801 * kMs, w) == 2);  // over 300 ms late: ends even a short wait
    WaitClock t(w, 1795 * kMs);
    EXPECT(t.expired(1801 * kMs, w) == 2);  // 6 ms of waiting, 301 ms behind the deadline: the safety net of the pass
}

void test_no_deadline() {
    const WaitLimits w = limits(~0ull);
    WaitClock c(w, 0);
    EXPECT(c.expired(900 * kMs, w) == 0);
    c.poll(900 * kMs + 50 * kMs * 40, w);  // a freeze with no deadline: the shift must not overflow ~0
    EXPECT(c.dl == ~0ull);
}

}  // namespace

int main() {
    test_gap_below_and_above_freeze();
    test_hang_still_found();
    test_freeze_off();
    test_cap();
    test_deadline();
    test_no_deadline();
    if (g_fail) {
        fprintf(stderr, "%d expectation(s) failed\n", g_fail);
        return 1;
    }
    puts("wait_clock: ok");
    return 0;
}
