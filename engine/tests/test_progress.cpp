// Startup progress and the prefill counters behind /status.
#include <thread>
#include <vector>

#include "model/progress.h"
#include "tests/test_support.h"

namespace {

void test_phases() {
    EXPECT(!tona::progress::ready() && tona::progress::phases().empty());
    tona::progress::add(5);  // no phase yet: ignored
    tona::progress::begin("Experts", 10);
    tona::progress::add(2.5);
    tona::progress::add(2.5);
    tona::progress::begin("Upload");  // ends the running phase
    tona::progress::set_total(8);
    tona::progress::add(1);
    auto p = tona::progress::phases();
    EXPECT(p.size() == 2 && p[0].name == "Experts" && p[0].done == 5 && p[0].total == 10);
    EXPECT(p[0].t1 >= p[0].t0 && p[1].t1 < 0 && p[1].total == 8 && p[1].done == 1);
    EXPECT(!tona::progress::ready());
    tona::progress::finish();
    p = tona::progress::phases();
    EXPECT(tona::progress::ready() && p[1].t1 >= p[1].t0);
    EXPECT(tona::progress::now() >= p[1].t1);
}

void test_concurrent_adds() {
    tona::progress::begin("Parallel", 4000);
    std::vector<std::thread> threads;
    threads.reserve(4);
    for (int t = 0; t < 4; t++)
        threads.emplace_back([] {
            for (int i = 0; i < 1000; i++) tona::progress::add(1);
        });
    for (auto& t : threads) t.join();
    EXPECT(tona::progress::phases().back().done == 4000);
}

void test_prefill_counters() {
    int64_t done = -1, total = -1;
    tona::progress::prefill_state(done, total);
    EXPECT(done == 0 && total == 0);
    tona::progress::prefill(100, 5000);
    tona::progress::prefill_state(done, total);
    EXPECT(done == 100 && total == 5000);
    tona::progress::prefill_done(2500);
    tona::progress::prefill_state(done, total);
    EXPECT(done == 2500 && total == 5000);
    tona::progress::prefill(0, 0);
    tona::progress::prefill_state(done, total);
    EXPECT(done == 0 && total == 0);
}

}  // namespace

int main() {
    return tona::test::run("test_progress", [] {
        test_phases();
        test_concurrent_adds();
        test_prefill_counters();
    });
}
