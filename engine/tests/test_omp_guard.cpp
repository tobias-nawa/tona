// OmpErrors: an exception thrown inside an OpenMP region reaches the caller after the region instead of ending the
// program; the first one is kept, the iterations after a failure are skipped, a region without failure is unchanged.
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

#include "backend/omp_guard.h"
#include "tests/test_guard.h"

using tona::backend::OmpErrors;

namespace {

int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

constexpr int N = 4096;

// The loop of read_misses: one failing iteration in the middle
void test_error_propagates() {
    std::vector<int> done(N, 0);
    OmpErrors err;
#pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < N; i++)
        err.run([&] {
            if (i == 1234) throw std::runtime_error("pread: injected");
            done[(size_t)i] = 1;
        });
    EXPECT(err.failed());
    bool caught = false;
    try {
        err.rethrow();
    } catch (const std::runtime_error& e) {
        caught = std::string(e.what()) == "pread: injected";
    }
    EXPECT(caught);
    EXPECT(done[1234] == 0);
    // The iteration right before it ran: same thread, same static chunk. Iteration 0 is another thread's, which a
    // loaded runner may start only after the failure, and then skips it.
    EXPECT(done[1233] == 1);
}

// Several failures: exactly one exception comes out, no terminate
void test_many_failures_give_one_error() {
    OmpErrors err;
#pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < N; i++)
        err.run([&] {
            if (i % 7 == 0) throw std::runtime_error("many");
        });
    int thrown = 0;
    try {
        err.rethrow();
    } catch (const std::exception&) {
        thrown++;
    }
    EXPECT(thrown == 1);
}

void test_no_failure_is_silent() {
    std::vector<int> done(N, 0);
    OmpErrors err;
#pragma omp parallel for schedule(static) num_threads(4)
    for (int i = 0; i < N; i++) err.run([&] { done[(size_t)i] = i + 1; });
    EXPECT(!err.failed());
    err.rethrow();  // does not throw
    long sum = 0;
    for (int v : done) sum += v;
    EXPECT(sum == (long)N * (N + 1) / 2);
}

// A loop without OpenMP (compiled without the flag) behaves the same
void test_serial_loop() {
    OmpErrors err;
    for (int i = 0; i < 10; i++)
        err.run([&] {
            if (i == 3) throw std::logic_error("serial");
        });
    bool caught = false;
    try {
        err.rethrow();
    } catch (const std::logic_error&) {
        caught = true;
    }
    EXPECT(caught);
}

}  // namespace

int main() {
    return tona::test::guard("omp_guard", [&] {
        test_error_propagates();
        test_many_failures_give_one_error();
        test_no_failure_is_silent();
        test_serial_loop();
        if (fails) return 1;
        std::printf("test_omp_guard: ok\n");
        return 0;
    });
}
