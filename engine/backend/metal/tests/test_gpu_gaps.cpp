// The GPU gap probe: a gap between two command buffers splits into wake, backend, outside, inputs and submit, and
// command buffers outside the recorded calls or across a long pause make no gap
#include <cmath>

#include "backend/metal/gpu_gaps.h"
#include "tests/test_support.h"

using tona::metal::GpuGaps;

namespace {

bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

// Times in ms, as the probe reports them
GpuGaps::Cb cb(double prepared, double committed, double start, double end, double waited) {
    return {prepared * 1e-3, committed * 1e-3, start * 1e-3, end * 1e-3, waited * 1e-3, {}, 0, 0};
}

void test_a_gap_across_two_calls_splits_into_its_parts() {
    GpuGaps g(1000);
    g.enter(GpuGaps::kTarget, 0);
    g.add(cb(0, 1, 2, 10, 11));
    g.leave(12e-3);
    g.enter(GpuGaps::kStep, 14e-3);
    g.add(cb(15, 16, 17, 20, 20.5));
    g.leave(21e-3);
    const GpuGaps::Means m = g.means();  // per target pass: one here
    EXPECT(near(m.busy, 11));
    EXPECT(near(m.wake, 1) && near(m.backend, 2) && near(m.outside, 2) && near(m.inputs, 1) && near(m.submit, 1));
    EXPECT(near(m.encode_other, 1) && near(m.encode_target, 0));  // the gap's buffer belongs to a step
    EXPECT(near(m.idle, 7));
    EXPECT(near(m.by_step[GpuGaps::kTarget][GpuGaps::kStep], 7));
}

void test_the_parts_of_inputs_count_with_their_gap() {
    GpuGaps g(1000);
    g.enter(GpuGaps::kTarget, 0);
    GpuGaps::Cb first = cb(0, 1, 2, 10, 11);
    first.parts = {1e-3, 1e-3, 1e-3};  // the first buffer has no gap before it: its parts are not counted
    g.add(first);
    GpuGaps::Cb second = cb(12, 14, 15, 18, 19);
    second.parts = {0.1e-3, 0.5e-3, 0.25e-3};
    g.add(second);
    g.leave(20e-3);
    const GpuGaps::Means m = g.means();  // per target pass: two here
    EXPECT(near(m.history, 0.05) && near(m.rows, 0.25) && near(m.positions, 0.125) && near(m.inputs, 1));
    EXPECT(near(m.encode_target, 1 - 0.05 - 0.25 - 0.125) && near(m.encode_other, 0));
}

void test_the_idle_inside_a_split_pass_is_idle_not_busy() {
    GpuGaps g(1000);
    g.enter(GpuGaps::kTarget, 0);
    GpuGaps::Cb c = cb(0, 1, 2, 10, 11);
    c.inner = 0.5e-3;  // the second half started 0.5 ms after the first ended,
    c.late = 0.2e-3;   // 0.2 ms of it before the host had committed it
    g.add(c);
    g.add(cb(12, 13, 14, 20, 21));  // a pass without a split: half of the passes over 0.1 ms
    g.leave(22e-3);
    const GpuGaps::Means m = g.means();
    EXPECT(near(m.busy, (7.5 + 6) / 2) && near(m.split, 0.25) && near(m.split_late, 0.1) && near(m.split_big, 50) &&
           near(m.split_max, 0.5));
}

void test_a_gap_inside_one_call_has_no_outside_part() {
    GpuGaps g(1000);
    g.enter(GpuGaps::kTarget, 0);
    g.add(cb(0, 1, 2, 10, 11));
    g.add(cb(13, 14, 15, 18, 19));
    g.leave(20e-3);
    const GpuGaps::Means m = g.means();  // per target pass: two here
    EXPECT(near(m.backend, 1) && near(m.outside, 0) && near(m.idle, 2.5));
}

void test_buffers_outside_calls_and_long_pauses_make_no_gap() {
    GpuGaps g(1000);
    g.enter(GpuGaps::kTarget, 0);
    g.add(cb(0, 1, 2, 10, 11));
    g.leave(12e-3);
    g.add(cb(13, 14, 15, 18, 19));  // outside any call: cuts the chain
    g.enter(GpuGaps::kTarget, 20e-3);
    g.add(cb(20, 21, 22, 30, 31));
    g.add(cb(2000, 2001, 2002, 2010, 2011));  // two seconds later: a pause between requests
    g.leave(2012e-3);
    const GpuGaps::Means m = g.means();
    EXPECT(near(m.idle, 0));
}

}  // namespace

int main() {
    return tona::test::run("gpu gaps", [] {
        test_a_gap_across_two_calls_splits_into_its_parts();
        test_the_parts_of_inputs_count_with_their_gap();
        test_the_idle_inside_a_split_pass_is_idle_not_busy();
        test_a_gap_inside_one_call_has_no_outside_part();
        test_buffers_outside_calls_and_long_pauses_make_no_gap();
    });
}
