// gpu/trace_path.h: the PID in the TONA_ALLOC_TRACE path.
#include "gpu/trace_path.h"
#include "tests/test_support.h"

namespace {

using tona::gpu::trace::expand_pid;

void test_no_placeholder() {
    EXPECT(expand_pid("/x/alloc-trace.txt", 42) == "/x/alloc-trace.txt");
    EXPECT(expand_pid("", 42).empty());
}

void test_one_placeholder() { EXPECT(expand_pid("/x/alloc-trace-%p.txt", 4711) == "/x/alloc-trace-4711.txt"); }

void test_two_placeholders() { EXPECT(expand_pid("%p/t-%p.txt", 7) == "7/t-7.txt"); }

void test_other_percent_kept() {
    EXPECT(expand_pid("100%/a%", 5) == "100%/a%");
    EXPECT(expand_pid("%d-%p", 5) == "%d-5");
}

}  // namespace

int main() {
    return tona::test::run("test_trace_path", [] {
        test_no_placeholder();
        test_one_placeholder();
        test_two_placeholders();
        test_other_percent_kept();
    });
}
