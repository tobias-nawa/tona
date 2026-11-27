// backend/stall.h: the time limits of the stash copy.
#include <type_traits>

#include "backend/stall.h"
#include "tests/test_support.h"

namespace {

using namespace tona::backend;

// 2 s base plus the time at the floor rate: a whole 32 MB chunk (HALF_BYTES) gets just over 2.1 s
void test_chunk_deadline() {
    EXPECT(chunk_deadline_s(0) == CHUNK_BASE_S);
    const double chunk = chunk_deadline_s(size_t(32) << 20);
    EXPECT(chunk > 2.1 && chunk < 2.2);
    EXPECT(chunk_deadline_s(size_t(64) << 20) > chunk);  // monotonic
}

// The floor is far below a healthy copy: 32 MB at 10 GB/s take 3 ms, so the limit holds ~700x reserve
void test_reserve_over_healthy_rate() {
    const double healthy_s = (double)(size_t(32) << 20) / 10e9;
    EXPECT(chunk_deadline_s(size_t(32) << 20) > 500 * healthy_s);
}

// A copy that is slow but runs (50 MB/s on a chunk) stays under the limit only if the floor allows: 32 MB at 250 MB/s
// is the slowest rate that passes
void test_slowest_passing_rate() {
    const size_t chunk = size_t(32) << 20;
    EXPECT((double)chunk / chunk_deadline_s(chunk) > 15e6);   // an effective rate of at least 15 MB/s passes
    EXPECT((double)chunk / chunk_deadline_s(chunk) < 250e6);  // never above the floor itself
}

void test_stall_constants() {
    EXPECT(STALL_GRACE_S >= 30 && STREAM_IDLE_S >= 10);
    EXPECT(EXIT_GPU_STALLED == 75);  // EX_TEMPFAIL: the service restarts on a non-zero code
    EXPECT((std::is_base_of_v<std::runtime_error, CopyStalled>));
}

}  // namespace

int main() {
    return tona::test::run("test_stall", [] {
        test_chunk_deadline();
        test_reserve_over_healthy_rate();
        test_slowest_passing_rate();
        test_stall_constants();
    });
}
