// Bounds of the expert-grouped MoE rows (gpu/prefill_check.h): a CPU model of k_route/k_scatter fills the rowmap, every
// row must lie in the range of its expert and inside the C * ne rows that carve_moe gives b.O.
#include <cmath>
#include <cstdint>
#include <vector>

#include "gpu/prefill_check.h"
#include "tests/test_support.h"

using namespace tona::gpu;

namespace {

struct Grouping {
    std::vector<int32_t> cnt, off, rowmap;
};

// What the scatter kernel does: row = off[e] + running fill of the expert
Grouping group(const std::vector<int32_t>& sel, int E) {
    Grouping g;
    g.cnt.assign(E, 0), g.off.assign(E, 0), g.rowmap.resize(sel.size());
    for (int32_t e : sel) g.cnt[e]++;
    for (int e = 1; e < E; e++) g.off[e] = g.off[e - 1] + g.cnt[e - 1];
    std::vector<int32_t> fill(E, 0);
    for (size_t a = 0; a < sel.size(); a++) g.rowmap[a] = g.off[sel[a]] + fill[sel[a]]++;
    return g;
}

std::vector<int32_t> pseudo_sel(int assignments, int E) {
    std::vector<int32_t> sel(assignments);
    uint32_t x = 12345;
    for (auto& e : sel) x = x * 1664525u + 1013904223u, e = (int32_t)((x >> 8) % (uint32_t)E);
    return sel;
}

void test_scatter_rows_are_inside() {
    const int E = 256, ne = 8, C = 6144;
    for (int n : {1, 100, 5120, 6144}) {
        const Grouping g = group(pseudo_sel(n * ne, E), E);
        const int64_t rows = (int64_t)n * ne;
        EXPECT(pf_counts_ok(g.cnt.data(), E, rows));
        EXPECT(rows <= (int64_t)C * ne);  // b.O holds C * ne rows
        for (int32_t r : g.rowmap) EXPECT(r >= 0 && r < rows);
    }
}

void test_every_row_in_its_expert_range() {
    const int E = 64, ne = 4, n = 1000;
    const std::vector<int32_t> sel = pseudo_sel(n * ne, E);
    const Grouping g = group(sel, E);
    for (size_t a = 0; a < sel.size(); a++)
        EXPECT(pf_row_ok(g.rowmap[a], sel[a], E, (int64_t)n * ne, g.off.data(), g.cnt.data()));
}

void test_violations_are_found() {
    const int E = 8, ne = 2, n = 10;
    const std::vector<int32_t> sel = pseudo_sel(n * ne, E);
    const Grouping g = group(sel, E);
    const int64_t rows = (int64_t)n * ne;
    const int e = sel[0];
    EXPECT(!pf_row_ok(rows, e, E, rows, g.off.data(), g.cnt.data()));                 // one row past the chunk
    EXPECT(!pf_row_ok(-1, e, E, rows, g.off.data(), g.cnt.data()));                   // negative
    EXPECT(!pf_row_ok(g.off[e] + g.cnt[e], e, E, rows, g.off.data(), g.cnt.data()));  // first row of the next expert
    EXPECT(!pf_row_ok(0, E, E, rows, g.off.data(), g.cnt.data()));                    // expert outside [0, E)
    EXPECT(!pf_row_ok(0, 1 << 30, E, rows, g.off.data(), g.cnt.data()));  // router without a candidate (NaN row)
    std::vector<int32_t> bad = g.cnt;
    bad[0]++;
    EXPECT(!pf_counts_ok(bad.data(), E, rows));  // a count too many: rows would run past the chunk
    bad = g.cnt, bad[1] = -1;
    EXPECT(!pf_counts_ok(bad.data(), E, rows));
}

void test_nan_logit_is_cleaned() {
    const float nan = std::nanf("");
    EXPECT(pf_clean_logit(nan) == -INFINITY);
    EXPECT(pf_clean_logit(-INFINITY) == -INFINITY);
    EXPECT(pf_clean_logit(1.5f) == 1.5f && pf_clean_logit(-0.0f) == 0.0f);
}

}  // namespace

int main() {
    return tona::test::run("pf_check", [] {
        test_scatter_rows_are_inside();
        test_every_row_in_its_expert_range();
        test_violations_are_found();
        test_nan_logit_is_cleaned();
    });
}
