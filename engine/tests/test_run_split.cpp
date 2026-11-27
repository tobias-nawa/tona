// How a stretch of a prompt is split between the prefill and forward: the rule that makes a hit on a checkpoint
// compute the same tokens as a cold run of the whole prompt.
#include "server/run_split.h"
#include "tests/test_support.h"

using tona::server::RunSplit;
using tona::server::split_run;

namespace {

constexpr size_t MIN = 32;

bool is(RunSplit s, size_t prefill, size_t forward) { return s.prefill == prefill && s.forward == forward; }

// Cold run: all but the last token through the prefill, the last through forward
void test_whole_prompt() {
    EXPECT(is(split_run(1000, true, false, MIN, true), 999, 1));
    EXPECT(is(split_run(33, true, false, MIN, true), 32, 1));
    EXPECT(is(split_run(32, true, false, MIN, true), 0, 32));  // short prompt: forward only, as before
    EXPECT(is(split_run(1, true, false, MIN, true), 0, 1));
}

// A checkpoint inside the prompt: everything through the prefill, no forward token, whatever its length
void test_checkpoint_inside() {
    EXPECT(is(split_run(2048, false, false, MIN, true), 2048, 0));
    EXPECT(is(split_run(5, false, false, MIN, true), 5, 0));
    EXPECT(is(split_run(5, false, true, MIN, true), 5, 0));
}

// The rest behind a checkpoint: short rests stay in the prefill (the cold run prefilled them)
void test_rest_behind_checkpoint() {
    EXPECT(is(split_run(41, true, true, MIN, true), 40, 1));
    EXPECT(is(split_run(20, true, true, MIN, true), 19, 1));
    EXPECT(is(split_run(2, true, true, MIN, true), 1, 1));
    EXPECT(is(split_run(1, true, true, MIN, true), 0, 1));
}

// The cut does not change what the tokens go through: a prompt split at any place has the same tokens in the prefill
// as the whole prompt, except the one behind the last cut
void test_cuts_keep_the_partition() {
    constexpr size_t N = 9000;
    for (size_t cut : {1, 31, 32, 2048, 4097, 6144, 8999}) {
        const RunSplit head = split_run(cut, false, false, MIN, true);
        const RunSplit rest = split_run(N - cut, true, true, MIN, true);
        EXPECT(head.prefill + rest.prefill == N - 1);
        EXPECT(head.forward == 0 && rest.forward == 1);
    }
}

void test_prefill_off() {
    EXPECT(is(split_run(1000, true, false, MIN, false), 0, 1000));
    EXPECT(is(split_run(1000, false, true, MIN, false), 0, 1000));
    EXPECT(is(split_run(0, true, false, MIN, true), 0, 0));
}

}  // namespace

int main() {
    return tona::test::run("test_run_split", [] {
        test_whole_prompt();
        test_checkpoint_inside();
        test_rest_behind_checkpoint();
        test_cuts_keep_the_partition();
        test_prefill_off();
    });
}
