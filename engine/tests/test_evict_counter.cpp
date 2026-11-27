// Tests for the eviction credit: the protocol between a waiting block and the host (gpu/wait_clock.h,
// evict_extend) and the host's answer (gpu/evict_counter.h), on the CPU. The host is simulated inside sleep().
#include <cstdio>
#include <functional>

#include "gpu/evict_counter.h"
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

constexpr uint64_t kUs = 100;  // ticks per us (100 MHz)
constexpr uint64_t kMs = 1000 * kUs;

// One launch: the words of the bar, the mailbox, a clock the test moves, a host that answers or not
struct Rig {
    unsigned long long ext[4] = {}, start = 1000 * kUs;
    unsigned mb[4] = {};
    uint64_t t = 0;
    int host_ms = -1;  // answer to a request (-1: the host does not answer)
    int asked = 0;
    WaitLimits w{1000 * kMs, 1500 * kMs, 2 * kMs, 5000 * kMs, 0, 0, (unsigned)kUs, 0, nullptr, nullptr};
    EvictLink link{ext, &start, mb};
    bool extend(WaitClock& c) {
        return evict_extend(
            link, w.deadline, w.us_ticks, c.dl, [&] { return t; },
            [&] {
                t += kMs;  // one ms per pause
                if (mb[0] != mb[1] && host_ms >= 0) mb[2] = (unsigned)host_ms, mb[1] = mb[0], asked++;
            });
    }
};

void test_credit_arithmetic() {
    EXPECT(evict_credit_ticks(400, 900 * kMs, kMs) == 400 * kMs);
    EXPECT(evict_credit_ticks(4000, 900 * kMs, kMs) == 900 * kMs);      // never more than the wall time of the pass
    EXPECT(evict_credit_ticks(999999, ~0ull / 2, kMs) == 30000 * kMs);  // the cap
    EXPECT(evict_credit_ticks(0, 900 * kMs, kMs) == 0);
}

void test_host_answer() {
    EXPECT(evict_answer_ms(1500, 1000, 800) == 500);
    EXPECT(evict_answer_ms(1500, 1000, 300) == 300);  // the counter cannot have grown more than the time passed
    EXPECT(evict_answer_ms(1000, 1000, 800) == 0);
    EXPECT(evict_answer_ms(100, 1000, 800) == 0);  // the counter went backwards
    EXPECT(evict_answer_ms(100000, 0, 99999) == kEvictCapMs);
}

void test_kfd_names() {
    EXPECT(kfd_location_id("0000:0f:00.0") == 3840);
    EXPECT(kfd_location_id("0000:03:00.0") == 768);
    EXPECT(kfd_location_id("junk") == -1);
    EXPECT(kfd_evicted_path(42, 7773) == "/sys/class/kfd/kfd/proc/42/stats_7773/evicted_ms");
}

// All blocks computed while the queues were evicted for 4.4 s; no wave polled, so nothing shifted
void test_eviction_while_computing() {
    Rig r;
    r.host_ms = 4400;
    WaitClock c(r.w, 5400 * kMs);  // a wait starts when the queues run again
    r.t = 5400 * kMs;
    EXPECT(c.expired(r.t, r.w) == 2);  // 4,4 s behind a deadline of 1,5 s: the deadline alone ends it
    EXPECT(r.extend(c));
    EXPECT(c.dl == 1500 * kMs + 4400 * kMs);
    EXPECT(c.expired(r.t, r.w) == 0);
    EXPECT(r.asked == 1 && r.ext[0] == 0 && r.ext[1] == 4400 * kMs);
}

// A wave that polls without gaps for a block that never arrives is not rescued by the credit: the limit of the wait
// ends it
void test_hang_not_credited() {
    Rig r;
    r.host_ms = 4400;
    WaitClock c(r.w, 0);
    EXPECT((c.expired(1100 * kMs, r.w) & 1) != 0);  // kind != 2: the caller does not ask
}

void test_no_answer_aborts() {
    Rig r;
    WaitClock c(r.w, 2000 * kMs);
    r.t = 2000 * kMs;
    EXPECT(!r.extend(c));
    EXPECT(r.t - 2000 * kMs > kEvictAskMs * kMs && r.t - 2000 * kMs < (kEvictAskMs + 5) * kMs);  // about 50 ms
    EXPECT(r.ext[0] == 2 && c.dl == 1500 * kMs);
    const uint64_t t = r.t;
    EXPECT(!r.extend(c));  // refusal is final for the launch: no second wait of 50 ms
    EXPECT(r.t == t);
}

void test_answer_clamped_to_elapsed() {
    Rig r;
    r.host_ms = 20000;  // a counter that jumped: pass started at 1000 ms, it is 2400 ms now
    r.t = 2400 * kMs;
    WaitClock c(r.w, r.t);
    EXPECT(r.extend(c));
    EXPECT(r.ext[1] <= (r.t - r.start) && r.ext[1] > 1300 * kMs);  // at most the wall time since the pass started
    EXPECT(c.dl <= r.w.deadline + (r.t - r.start));
}

void test_second_eviction_and_no_news() {
    Rig r;
    r.host_ms = 800;
    r.t = 2500 * kMs;
    WaitClock c(r.w, r.t);
    EXPECT(r.extend(c) && r.ext[1] == 800 * kMs);
    r.t = 4000 * kMs;  // deadline 2300 passed again, the host still says 800: nothing new
    EXPECT(!r.extend(c) && r.ext[0] == 2);
    Rig q;
    q.host_ms = 800;
    q.t = 2500 * kMs;
    WaitClock d(q.w, q.t);
    EXPECT(q.extend(d));
    q.host_ms = 1900;  // a second eviction since: more credit
    q.t = 4000 * kMs;
    EXPECT(q.extend(d) && q.ext[1] == 1900 * kMs && d.dl == 1500 * kMs + 1900 * kMs);
}

void test_other_block_asks() {
    Rig r;
    r.ext[0] = 1;  // another block has asked and waits for the answer
    WaitClock c(r.w, 2000 * kMs);
    r.t = 2000 * kMs;
    EXPECT(r.extend(c) && c.dl == 1500 * kMs && r.mb[0] == 0);  // poll again, no second request
    r.ext[1] = 600 * kMs, r.ext[0] = 0;                         // the answer came
    EXPECT(r.extend(c) && c.dl == 1500 * kMs + 600 * kMs);      // taken over without asking
    EXPECT(r.mb[0] == 0);
}

void test_no_link() {
    Rig r;
    r.link.mb = nullptr;
    WaitClock c(r.w, 2000 * kMs);
    EXPECT(!r.extend(c));
}

}  // namespace

int main() try {
    test_credit_arithmetic();
    test_host_answer();
    test_kfd_names();
    test_eviction_while_computing();
    test_hang_not_credited();
    test_no_answer_aborts();
    test_answer_clamped_to_elapsed();
    test_second_eviction_and_no_news();
    test_other_block_asks();
    test_no_link();
    if (g_fail) {
        fprintf(stderr, "%d expectation(s) failed\n", g_fail);
        return 1;
    }
    puts("evict_counter: ok");
    return 0;
} catch (...) {
    return 2;
}
