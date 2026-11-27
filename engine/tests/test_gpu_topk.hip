// k_topk (gpu/prefill_topk.hip) and op_topk of the megakernel (gpu/mega_op_topk.h) against a CPU reference of the
// selection rule: the n_top best pool blocks, ties go to the earlier block, cells ascending, then the tail of the not
// yet pooled positions. cells and ncells must match byte for byte for every input, because the attention reads the
// cells in this order. Needs the GPU, no model.
// Usage: test_gpu_topk [--bench]   (--bench: time per launch of k_topk for 32 tokens
// and of op_topk for 1 and 3 tokens at several positions, random scores)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu/mega_op_topk.h"
#include "gpu/mega_types.h"
#include "gpu/prefill_kernels.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;

namespace {

constexpr int BLK = 4, TOPK = 2048;
constexpr int N_TOP = TOPK / BLK;
constexpr int MAXC = tona::gpu::MEGA_MAX_CELLS;
constexpr int32_t SENTINEL = -7;
constexpr int MEGA_GRID = 32;  // blocks of k_mega

enum class Pattern { Random, Ties, AllEqual, Zeros, FewLevels, Descending };

const char* name_of(Pattern p) {
    switch (p) {
        case Pattern::Random:
            return "random";
        case Pattern::Ties:
            return "ties";
        case Pattern::AllEqual:
            return "all-equal";
        case Pattern::Zeros:
            return "zeros";
        case Pattern::FewLevels:
            return "few-levels";
        case Pattern::Descending:
            return "descending";
    }
    return "?";
}

// Non-negative float bit patterns, as the scoring kernels write them (ReLU sums)
uint32_t score_bits(Pattern p, std::mt19937& rng, int64_t i) {
    float v = 0.0f;
    switch (p) {
        case Pattern::Random:
            v = std::uniform_real_distribution<float>(0.0f, 50.0f)(rng);
            break;
        case Pattern::Ties:
            v = (float)(rng() % 64) * 0.25f;
            break;
        case Pattern::AllEqual:
            v = 1.5f;
            break;
        case Pattern::Zeros:
            v = 0.0f;
            break;
        case Pattern::FewLevels:
            v = rng() % 100 < 3 ? 2.0f : 1.0f;
            break;
        case Pattern::Descending:
            v = 1.0e6f - (float)i;
            break;
    }
    uint32_t b;
    std::memcpy(&b, &v, 4);
    return b;
}

// The rule of op_topk: threshold = n_top-th largest value; everything above it, plus the first ones equal to it
std::vector<int32_t> reference(const uint32_t* sc, int64_t pos) {
    const int64_t n_pool = (pos + 1) / BLK;
    const int n_top = (int)std::min<int64_t>(n_pool, N_TOP);
    std::vector<uint32_t> sorted(sc, sc + n_pool);
    std::nth_element(sorted.begin(), sorted.begin() + (n_top - 1), sorted.end(), std::greater<uint32_t>());
    const uint32_t thr = sorted[n_top - 1];
    int64_t gt = 0;
    for (int64_t i = 0; i < n_pool; i++) gt += sc[i] > thr;
    int64_t take_eq = n_top - gt;
    std::vector<int32_t> cells;
    for (int64_t i = 0; i < n_pool; i++) {
        const bool take = sc[i] > thr || (sc[i] == thr && take_eq-- > 0);
        for (int k = 0; take && k < BLK; k++) cells.push_back((int32_t)(i * BLK + k));
    }
    for (int64_t p = n_pool * BLK; p <= pos; p++) cells.push_back((int32_t)p);
    return cells;
}

struct Case {
    Pattern pattern;
    int64_t pos0;
    int count, t0;
    uint32_t seed;
    bool mega;  // op_topk of the megakernel (1 ... BMAX tokens of one pass, t0 = 0) instead of k_topk
};

// op_topk as k_megas runs it: a block per token of the pass, the grid wider than the pass (idle blocks leave)
__global__ void __launch_bounds__(tona::gpu::THREADS)
    k_mega_topk(tona::gpu::MegaOp op, tona::gpu::MegaParams* params, int nb) {
    __shared__ float sm[tona::gpu::SM_FLOATS];
    tona::gpu::Ctx c{};
    c.pos = params->pos, c.nb = nb, c.lane = threadIdx.x % tona::gpu::WAVE, c.P = params, c.sm = sm;
    tona::gpu::op_topk(op, c);
}

// Launches the selection of the case; the megakernel variant leaves ncells in the n_cells of its parameters
void launch_case(const Case& c, const DeviceBuffer& d_sc, DeviceBuffer& d_cells, DeviceBuffer& d_nc, int64_t sstride) {
    if (!c.mega) {
        tona::gpu::launch_topk(
            {d_sc.get<float>(), d_cells.get<int32_t>(), d_nc.get<int32_t>(), c.t0, c.pos0, BLK, TOPK, sstride}, c.count,
            nullptr);
        return;
    }
    tona::gpu::MegaOp op;
    op.i[0] = op.i[5] = BLK, op.i[1] = op.i[7] = TOPK, op.i[3] = (int32_t)sstride;
    op.p[0] = d_sc.get(), op.p[1] = d_cells.get();
    k_mega_topk<<<MEGA_GRID, tona::gpu::THREADS>>>(op, d_nc.get<tona::gpu::MegaParams>(), c.count);
}

// One launch over `count` tokens at pos0 + t0 ...; scores row tt belongs to token t0 + tt
bool run_case(const Case& c) {
    const int64_t max_pos = c.pos0 + c.t0 + c.count - 1;
    const int64_t sstride = max_pos / BLK + 2;
    std::mt19937 rng(c.seed);
    std::vector<uint32_t> sc((size_t)c.count * sstride, 0);
    for (int tt = 0; tt < c.count; tt++) {
        const int64_t n_pool = (c.pos0 + c.t0 + tt + 1) / BLK;
        for (int64_t i = 0; i < n_pool; i++) sc[(size_t)tt * sstride + i] = score_bits(c.pattern, rng, i);
    }
    const int rows = c.t0 + c.count;
    std::vector<int32_t> cells((size_t)rows * MAXC, SENTINEL), nc(std::max(rows, tona::gpu::MEGA_BMAX), SENTINEL);
    tona::gpu::MegaParams params{};
    params.pos = c.pos0;
    std::fill(params.n_cells, params.n_cells + tona::gpu::MEGA_BMAX, SENTINEL);
    DeviceBuffer d_sc(sc.size() * 4), d_cells(cells.size() * 4), d_nc(c.mega ? sizeof(params) : nc.size() * 4);
    CHECK(hipMemcpy(d_sc.get(), sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_cells.get(), cells.data(), cells.size() * 4, hipMemcpyHostToDevice));
    if (c.mega)
        CHECK(hipMemcpy(d_nc.get(), &params, sizeof(params), hipMemcpyHostToDevice));
    else
        CHECK(hipMemcpy(d_nc.get(), nc.data(), nc.size() * 4, hipMemcpyHostToDevice));
    launch_case(c, d_sc, d_cells, d_nc, sstride);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(cells.data(), d_cells.get(), cells.size() * 4, hipMemcpyDeviceToHost));
    if (c.mega) {
        CHECK(hipMemcpy(&params, d_nc.get(), sizeof(params), hipMemcpyDeviceToHost));
        std::copy(params.n_cells, params.n_cells + tona::gpu::MEGA_BMAX, nc.begin());
    } else {
        CHECK(hipMemcpy(nc.data(), d_nc.get(), nc.size() * 4, hipMemcpyDeviceToHost));
    }
    bool ok = true;
    for (int tt = 0; tt < c.count; tt++) {
        const int t = c.t0 + tt;
        const int64_t pos = c.pos0 + t;
        std::vector<int32_t> want(MAXC, SENTINEL);
        int32_t want_n = SENTINEL;
        if ((pos + 1) / BLK > N_TOP) {  // qsa_sel: the token selects
            const auto ref = reference(&sc[(size_t)tt * sstride], pos);
            std::copy(ref.begin(), ref.end(), want.begin());
            want_n = (int32_t)ref.size();
        }
        if (nc[t] != want_n || std::memcmp(&cells[(size_t)t * MAXC], want.data(), sizeof(int32_t) * MAXC) != 0) {
            std::printf("  MISMATCH token %d (pos %lld): ncells %d, want %d\n", t, (long long)pos, nc[t], want_n);
            ok = false;
            if (tt > 4) break;
        }
    }
    return ok;
}

// Milliseconds per launch (32 tokens) with random scores, median of 20 launches
void bench(int64_t pos0) {
    const int64_t sstride = (pos0 + 32) / BLK + 2;
    std::mt19937 rng(5);
    std::vector<uint32_t> sc((size_t)32 * sstride, 0);
    for (int tt = 0; tt < 32; tt++)
        for (int64_t i = 0; i < (pos0 + tt + 1) / BLK; i++)
            sc[(size_t)tt * sstride + i] = score_bits(Pattern::Random, rng, i);
    DeviceBuffer d_sc(sc.size() * 4), d_cells((size_t)32 * (size_t)MAXC * sizeof(int32_t)), d_nc(32 * sizeof(int32_t));
    CHECK(hipMemcpy(d_sc.get(), sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
    std::vector<double> ms;
    for (int rep = 0; rep < 25; rep++) {
        CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        tona::gpu::launch_topk(
            {d_sc.get<float>(), d_cells.get<int32_t>(), d_nc.get<int32_t>(), 0, pos0, BLK, TOPK, sstride}, 32, nullptr);
        CHECK(hipDeviceSynchronize());
        ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(ms.begin() + 5, ms.end());
    std::printf("k_topk 32 tokens at pos %7lld: %.3f ms per launch (median of 20)\n", (long long)pos0, ms[15]);
}

// Microseconds per op_topk launch (one block per token, the rest of the 32 blocks idle), median of 20; includes the
// launch of a kernel of its own (~10 us), so only differences between variants mean something
void bench_mega(int64_t pos0, int nb) {
    const int64_t sstride = (pos0 + nb) / BLK + 2;
    std::mt19937 rng(5);
    std::vector<uint32_t> sc((size_t)nb * sstride, 0);
    for (int tt = 0; tt < nb; tt++)
        for (int64_t i = 0; i < (pos0 + tt + 1) / BLK; i++)
            sc[(size_t)tt * sstride + i] = score_bits(Pattern::Random, rng, i);
    DeviceBuffer d_sc(sc.size() * 4), d_cells((size_t)nb * MAXC * sizeof(int32_t)), d_p(sizeof(tona::gpu::MegaParams));
    tona::gpu::MegaParams params{};
    params.pos = pos0;
    CHECK(hipMemcpy(d_sc.get(), sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_p.get(), &params, sizeof(params), hipMemcpyHostToDevice));
    const Case c{Pattern::Random, pos0, nb, 0, 5u, true};
    std::vector<double> us;
    for (int rep = 0; rep < 25; rep++) {
        CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        launch_case(c, d_sc, d_cells, d_p, sstride);
        CHECK(hipDeviceSynchronize());
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(us.begin() + 5, us.end());
    std::printf("op_topk %d tokens at pos %7lld: %.0f us per launch (median of 20)\n", nb, (long long)pos0, us[15]);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (const int64_t pos0 : {8192, 16384, 65536, 131072, 262000}) bench(pos0);
        for (const int64_t pos0 : {16384, 131072, 262000})
            for (const int nb : {1, 3}) bench_mega(pos0, nb);
        return 0;
    }
    // pos0 around the first selection (n_pool = 513 at pos 2055), a tail of every length, and up to 256K positions
    const int64_t firsts[] = {2051, 2054, 2055, 2056, 2059, 4099, 20000, 65531, 131069, 262111};
    const Pattern pats[] = {Pattern::Random, Pattern::Ties,      Pattern::AllEqual,
                            Pattern::Zeros,  Pattern::FewLevels, Pattern::Descending};
    int failed = 0, total = 0;
    for (const Pattern p : pats)
        for (const int64_t pos0 : firsts) {
            const Case cases[] = {
                {p, pos0, 32, 0, 11u + (uint32_t)pos0, false}, {p, pos0, 7, 5, 99u + (uint32_t)pos0, false},
                {p, pos0, 1, 0, 3u + (uint32_t)pos0, true},    {p, pos0, 2, 0, 5u + (uint32_t)pos0, true},
                {p, pos0, 3, 0, 7u + (uint32_t)pos0, true},    {p, pos0, 4, 0, 13u + (uint32_t)pos0, true}};
            for (const Case& c : cases) {
                const bool ok = run_case(c);
                std::printf("%-10s %-6s pos0 %7lld count %2d t0 %d: %s\n", name_of(p), c.mega ? "mega" : "prefill",
                            (long long)pos0, c.count, c.t0, ok ? "ok" : "FAILED");
                failed += !ok;
                total++;
            }
        }
    std::printf("%d of %d cases failed\n", failed, total);
    return failed ? 1 : 0;
}
