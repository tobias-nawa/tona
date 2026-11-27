// The hold rule of server/hold_policy.h
#include <cmath>
#include <cstdio>
#include <vector>

#include "server/hold_policy.h"

using namespace tona::server;

static int fails = 0;

static void check(bool ok, const char* what) {
    if (!ok) std::printf("failed: %s\n", what), fails++;
}

static bool near(double a, double b) { return std::fabs(a - b) < 1e-9; }

// Too few pauses or a rate of zero: no hold
static void test_no_basis() {
    check(near(hold_seconds({5, 6, 7}, 20), 0), "three pauses");
    check(near(hold_seconds({5, 6, 7, 8}, 0), 0), "rate zero");
}

// Pauses alike: all covered (cost = mean pause / 1 = 7 s <= 20), with the margin
static void test_alike() {
    const std::vector<double> g = {5, 6, 7, 8, 9, 6, 7, 8};
    check(near(hold_seconds(g, 20, 1.0), 9), "alike, no margin");
    check(near(hold_seconds(g, 20, 1.1), 9 * 1.1), "alike, margin");
}

// One long absence among short pauses: covering it costs (7 x 6 + 600) / 8 = 80 s per save, over the rate of 20, so the
// hold stops at the usual pause; at the rate of 200 it is covered
static void test_absence() {
    const std::vector<double> lunch = {6, 6, 6, 6, 6, 6, 6, 600};
    check(near(hold_seconds(lunch, 20, 1.0), 6), "absence not covered");
    check(near(hold_seconds(lunch, 200, 1.0), 600), "absence covered at a high rate");
    check(near(hold_seconds({60, 60, 60, 60}, 20, 1.0), 0), "every pause over the rate");
}

// The window keeps the newest kWindow pauses
static void test_window() {
    PauseWindow w;
    for (int i = 0; i < 40; i++) w.add(i);
    check(w.gaps().size() == kWindow, "window size");
    check(near(w.gaps().front(), 8) && near(w.gaps().back(), 39), "window content");
}

int main() {
    test_no_basis();
    test_alike();
    test_absence();
    test_window();
    return fails ? 1 : 0;
}
