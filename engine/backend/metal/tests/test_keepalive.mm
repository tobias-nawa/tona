// The keepalive's window (keepalive.h): nothing before the first pass, nothing while passes come faster than the
// interval, due inside the window after a pass, not after
#include "backend/metal/keepalive.h"
#include "tests/test_support.h"

using tona::metal::Keepalive;

namespace {

void test_nothing_due_before_the_first_pass() {
    const Keepalive k(nil, 0, 180);
    EXPECT(!k.due(Keepalive::Clock::now()));
}

void test_due_inside_the_window_only() {
    Keepalive k(nil, 500, 180);
    k.touch();
    const auto t0 = Keepalive::Clock::now();
    EXPECT(!k.due(t0 + std::chrono::milliseconds(100)));  // a pass came just now
    EXPECT(k.due(t0 + std::chrono::seconds(170)));
    EXPECT(!k.due(t0 + std::chrono::seconds(190)));
}

}  // namespace

int main() {
    return tona::test::run("keepalive", [] {
        test_nothing_due_before_the_first_pass();
        test_due_inside_the_window_only();
    });
}
