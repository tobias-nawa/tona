// Tests for the suffix trie and draft building (SuffixDecoding)
#include <cstdio>
#include <vector>

#include "spec/suffix.h"
#include "tests/test_guard.h"

static int g_fail = 0;
#define EXPECT(c)                                                                       \
    do {                                                                                \
        if (!(c)) {                                                                     \
            fprintf(stderr, "%s:%d: expectation failed: %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                                   \
        }                                                                               \
    } while (0)

using namespace tona::spec;

namespace {

// 1 2 3 4 5 9 1 2 3 4 5 7 1 2: "1 2" is followed by "3 4 5" twice
struct Fixture {
    SuffixTrie t{8};
    Matcher m{t, 6};
    Fixture() {
        for (int32_t x : {1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5, 7, 1, 2}) {
            t.append(x);
            m.push(x);
        }
    }
};

const int32_t kActual[] = {3, 4, 5, 7, 1};

void test_counts(Fixture& f) {
    EXPECT(f.t.node(f.t.child(0, 1)).count == 3);
    EXPECT(f.t.node(f.t.child(f.t.child(0, 1), 2)).count == 3);
    EXPECT(f.m.matches().size() >= 2);  // "2" and "1 2" found
}

// Path: 3 4 5, then a tie 9/7 gives one candidate with probability 1/2
void test_path(Fixture& f, SpecParams& p) {
    p.alpha = 4;
    p.max_spec = 4;
    const Draft d = propose(f.t, f.m.matches(), p);
    EXPECT(d.size() == 4);
    EXPECT(d.token[0] == 3 && d.token[1] == 4 && d.token[2] == 5);
    EXPECT(d.prob[0] == 1.0f && d.prob[3] == 0.5f);
    EXPECT(accepted(d, kActual, 5) == 3 + (d.token[3] == 7));
    const int32_t wrong[] = {8, 3};
    EXPECT(accepted(d, wrong, 2) == 0);
}

// Tree: after 3 4 5 both branches 9 and 7
void test_tree(Fixture& f, SpecParams& p) {
    p.tree = true;
    p.max_spec = 5;
    const Draft d = propose(f.t, f.m.matches(), p);
    EXPECT(d.size() == 5);
    EXPECT(accepted(d, kActual, 5) == 4);
    const int32_t other[] = {3, 4, 5, 9};
    EXPECT(accepted(d, other, 4) == 4);
}

// min_prob cuts off the uncertain branch
void test_min_prob(Fixture& f, SpecParams& p) {
    p.tree = false;
    p.min_prob = 0.6f;
    EXPECT(propose(f.t, f.m.matches(), p).size() == 3);
}

// Depth limit: no nodes deeper than max_depth
void test_depth() {
    SuffixTrie s(2);
    for (int32_t x : {1, 2, 3, 4}) s.append(x);
    EXPECT(s.child(s.child(s.child(0, 1), 2), 3) < 0);
    EXPECT(s.child(s.child(0, 2), 3) >= 0);
}

}  // namespace

int main() {
    return tona::test::guard("suffix", [&] {
        Fixture f;
        SpecParams p;
        test_counts(f);
        test_path(f, p);
        test_tree(f, p);
        test_min_prob(f, p);
        test_depth();
        if (g_fail) {
            fprintf(stderr, "%d expectations failed\n", g_fail);
            return 1;
        }
        printf("test_suffix: ok\n");
        return 0;
    });
}
