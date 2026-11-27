// Tests for the online calibration of the drafter probabilities
#include <cmath>
#include <cstdio>
#include <random>

#include "spec/draft_calib.h"
#include "tests/test_guard.h"

static int g_fail = 0;
#define EXPECT(c)                                                                         \
    do {                                                                                  \
        if (!(c)) {                                                                       \
            fprintf(stderr, "%s:%d: expectation violated: %s\n", __FILE__, __LINE__, #c); \
            g_fail++;                                                                     \
        }                                                                                 \
    } while (0)

using tona::spec::DraftCalib;

static void test_raw_without_observations() {
    DraftCalib<3> c0;
    EXPECT(std::fabs(c0.predict(0.8, 0) - 0.8) < 1e-9);
    EXPECT(std::fabs(c0.predict(0.3, 2) - 0.3) < 1e-9);
}

// Overconfident drafter: true acceptance sigmoid(0.5 * logit(q) - 0.5 - 0.3 * i)
static double truth(double q, int i) { return 1 / (1 + std::exp(-(0.5 * std::log(q / (1 - q)) - 0.5 - 0.3 * i))); }

static void feed_overconfident(DraftCalib<3>& c) {
    std::mt19937_64 rng(7);
    std::uniform_real_distribution<double> uq(0.05, 0.99), u(0, 1);
    for (int pass = 0; pass < 4000; pass++) {
        double q[3];
        int a = 0;
        for (int i = 0; i < 3; i++) q[i] = uq(rng);
        while (a < 3 && u(rng) < truth(q[a], a)) a++;
        c.observe(q, 3, a);
    }
}

static void test_fit_overconfident() {
    DraftCalib<3> c(512, 16);
    feed_overconfident(c);
    for (int i = 0; i < 3; i++)
        for (double q : {0.3, 0.6, 0.9}) {
            const double d = c.predict(q, i) - truth(q, i);
            if (std::fabs(d) > 0.06)
                fprintf(stderr, "position %d, q %.1f: %.3f instead of %.3f\n", i, q, c.predict(q, i), truth(q, i));
            EXPECT(std::fabs(d) < 0.06);
        }
    EXPECT(std::fabs(c.params()[0] - 0.5) < 0.15);
}

// Only position 0 checked (width 1): the others stay near the start value
static void test_width_one() {
    DraftCalib<3> c1;
    for (int pass = 0; pass < 200; pass++) {
        const double q[3] = {0.9, 0.9, 0.9};
        c1.observe(q, 1, pass % 2);
    }
    EXPECT(std::isfinite(c1.predict(0.9, 2)));
}

int main() {
    return tona::test::guard("draft_calib", [&] {
        test_raw_without_observations();
        test_fit_overconfident();
        test_width_one();
        if (g_fail) return 1;
        printf("draft_calib: ok\n");
        return 0;
    });
}
