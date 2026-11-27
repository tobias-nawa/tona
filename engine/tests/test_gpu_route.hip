// k_route with NaN, +inf and mixed logit rows: every chosen expert must lie in [0, E), the counts must add up to
// n * ne, and the count of rows with a NaN logit (cnt[route_nan_word]) must be exact. Rows without NaN must choose the
// ne largest logits. Needs the GPU, no model.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "gpu/mega_mailbox_fwd.h"
#include "gpu/prefill_check.h"
#include "gpu/prefill_kernels.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;

namespace {

// E: experts of the run, 256 (the Coder's size class) and 500 and 512 (the wide class, 512 experts and fewer)
int E = 256;
constexpr int NE = 8, N = 512;
constexpr size_t SNE = NE, SN = N;
constexpr size_t kCntBytes = tona::gpu::route_count_words(tona::gpu::MEGA_MAX_EXPERTS_WIDE) * sizeof(int32_t);

int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

// Row t of the input: the kind of damage depends on t % 6
void fill_row(float* row, int kind, std::mt19937& rng) {
    const float nan = std::nanf(""), inf = INFINITY;
    switch (kind) {
        case 0:  // all NaN
            std::fill(row, row + E, nan);
            break;
        case 1:  // a single NaN
            row[rng() % E] = nan;
            break;
        case 2:  // all but a few are NaN
            for (int e = 0; e < E; e++)
                if (e % 37 != 0) row[e] = nan;
            break;
        case 3:  // +inf and -inf
            row[5] = inf, row[9] = -inf;
            break;
        case 4:  // all +inf
            std::fill(row, row + E, inf);
            break;
        default:
            break;
    }
}

// Rows without NaN choose the NE largest logits
void check_clean_rows(const std::vector<float>& logits, const std::vector<int32_t>& sel,
                      const std::vector<bool>& has_nan) {
    for (size_t t = 0; t < SN; t++) {
        if (has_nan[t]) continue;
        const float* row = &logits[t * E];
        std::vector<int> idx(E);
        std::iota(idx.begin(), idx.end(), 0);
        std::stable_sort(idx.begin(), idx.end(), [&](int a, int b) { return row[a] > row[b]; });
        const std::set<int> want(idx.begin(), idx.begin() + NE);
        const std::set<int> got(&sel[t * SNE], &sel[t * SNE] + NE);
        EXPECT(want == got);
    }
}

void run(int experts) {
    E = experts;
    const int max_e = tona::gpu::expert_cap(E), nan_word = tona::gpu::route_nan_word(max_e);
    std::mt19937 rng(72);
    std::normal_distribution<float> g;
    std::vector<float> logits(SN * E);
    std::vector<bool> has_nan(SN);
    for (size_t t = 0; t < SN; t++) {
        for (int e = 0; e < E; e++) logits[t * E + e] = g(rng);
        float* row = &logits[t * E];
        fill_row(row, (int)(t % 6), rng);
        has_nan[t] = std::any_of(row, row + E, [](float v) { return v != v; });
    }
    DeviceBuffer dl(logits.size() * 4), dsel(SN * SNE * 4), dsw(SN * SNE * 4), dcnt(kCntBytes);
    CHECK(hipMemcpy(dl.get(), logits.data(), logits.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(dcnt.get(), 0, kCntBytes));
    tona::gpu::launch_route(
        {dl.get<float>(), dsel.get<int32_t>(), dsw.get<float>(), dcnt.get<int32_t>(), N, E, NE, max_e}, nullptr);
    CHECK(hipDeviceSynchronize());
    std::vector<int32_t> sel(SN * SNE), cnt(kCntBytes / 4);
    CHECK(hipMemcpy(sel.data(), dsel.get(), sel.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(cnt.data(), dcnt.get(), kCntBytes, hipMemcpyDeviceToHost));
    EXPECT(std::all_of(sel.begin(), sel.end(), [](int32_t e) { return e >= 0 && e < E; }));
    EXPECT(std::accumulate(cnt.begin(), cnt.begin() + E, 0) == N * NE);
    EXPECT(cnt[nan_word] == (int)std::count(has_nan.begin(), has_nan.end(), true));
    check_clean_rows(logits, sel, has_nan);
}

}  // namespace

int main() {
    for (const int experts : {256, 500, 512}) run(experts);
    std::printf("test_gpu_route: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
