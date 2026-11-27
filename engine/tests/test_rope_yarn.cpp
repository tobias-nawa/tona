// backend/rope_yarn.h against reference values computed by tools/gen_yarn_ref.py (the formula of transformers
// `_compute_yarn_parameters`, independent of the multiplier form of the header), and the environment switch.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <vector>

#include "backend/rope_yarn.h"
#include "tests/test_support.h"

namespace {

constexpr double BASE = 1.0e7;

double native_freq(int i) { return std::pow(BASE, -2.0 * i / 64); }

void test_reference(const char* path) {
    std::ifstream in(path);
    EXPECT(in.good());
    double factor = 0, mscale = 0;
    int cases = 0;
    while (in >> factor >> mscale) {
        const auto t = tona::rope::yarn_table(factor, tona::rope::yarn_default_mscale(factor), BASE);
        EXPECT(std::fabs(t.mscale - mscale) < 1e-6);
        for (int i = 0; i < tona::rope::YARN_PAIRS; i++) {
            double ref = 0;
            in >> ref;
            EXPECT(std::fabs(native_freq(i) * t.mul[i] - ref) <= 1e-12 * ref);
        }
        cases++;
    }
    EXPECT(cases == 4);
}

void test_shape_and_off() {
    // pairs 0-14 keep the native frequency, 22-31 run at 1/4 (research/32 section 1.2)
    const auto t = tona::rope::yarn_table(4.0, 1.0, BASE);
    for (int i = 0; i <= 14; i++) EXPECT(t.mul[i] == 1.0);
    for (int i = 22; i < 32; i++) EXPECT(t.mul[i] == 0.25);
    EXPECT(t.mul[15] < 1.0 && t.mul[15] > t.mul[21] && t.mul[21] > 0.25);
    EXPECT(t.mscale == 1.0f);
    const auto off = tona::rope::yarn_table(1.0, 1.7, BASE);  // factor 1 is native whatever mscale says
    EXPECT(off.mscale == 1.0f && off.factor == 1.0);
    for (int i = 0; i < 32; i++) EXPECT(off.mul[i] == 1.0);
    EXPECT(std::fabs(tona::rope::yarn_default_mscale(4.0) - 1.1386294361) < 1e-9);
}

void test_env() {
    unsetenv("TONA_ROPE_YARN"), unsetenv("TONA_YARN_MSCALE");
    EXPECT(tona::rope::yarn_from_env(BASE).factor == 1.0);
    for (const char* off : {"off", "0", ""}) {
        setenv("TONA_ROPE_YARN", off, 1);
        EXPECT(tona::rope::yarn_from_env(BASE).factor == 1.0);
    }
    setenv("TONA_ROPE_YARN", "4", 1);
    auto t = tona::rope::yarn_from_env(BASE);
    EXPECT(t.factor == 4.0 && std::fabs(t.mscale - 1.1386294f) < 1e-6f);
    setenv("TONA_YARN_MSCALE", "1.0", 1);
    EXPECT(tona::rope::yarn_from_env(BASE).mscale == 1.0f);
    unsetenv("TONA_ROPE_YARN"), unsetenv("TONA_YARN_MSCALE");
}

}  // namespace

int main(int argc, char** argv) {
    return tona::test::run("test_rope_yarn", [&] {
        test_reference(argc > 1 ? argv[1] : "tests/data/yarn_ref.txt");
        test_shape_and_off();
        test_env();
    });
}
