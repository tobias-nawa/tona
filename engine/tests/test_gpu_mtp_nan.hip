// op_mtp_pick and op_route of the megakernel (gpu/mega_op_mtp.h, gpu/mega_op_moe.h) with NaN and inf logits.
// A row of NaN, one NaN, +inf, -inf and all +inf must give a row / experts inside their range, a finite sum, and a
// NaN count that is exact in the pick buffer; rows without NaN must give what the code before the change gave (the old
// pick is copied below) and what a CPU reference gives. Needs the GPU, no model.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <bit>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <vector>

#include "gpu/mega_op_moe.h"
#include "gpu/mega_op_mtp.h"
#include "gpu/mega_op_wait.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;
namespace g = tona::gpu;

namespace {

int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

constexpr int GRID = 4;  // blocks of the pick
constexpr size_t kW = g::MWAVES, kB = g::MEGA_BMAX, kU = g::MEGA_UMAX;

bool same_bits(float a, float b) { return std::bit_cast<uint32_t>(a) == std::bit_cast<uint32_t>(b); }

// The pick as it was before the change (first step of every block, merge in block 0)
__device__ void old_pick_part(const g::MegaOp& op, const g::Ctx& c, float* part) {
    const int t = threadIdx.x;
    const float* lg = (const float*)op.p[0] + (int64_t)(c.nb - 1) * op.i[1];
    const int32_t* map = (const int32_t*)op.p[1];
    const int n = op.i[0], per = (n + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * per, r1 = min(n, r0 + per);
    float bv = -INFINITY;
    int br = n;
    for (int r = r0 + t; r < r1; r += g::THREADS)
        if ((!map || map[r] >= 0) && lg[r] > bv) bv = lg[r], br = r;
    g::pick_reduce(bv, br, c.sm, (int*)(c.sm + kW));
    float s = 0;
    for (int r = r0 + t; r < r1; r += g::THREADS)
        if (!map || map[r] >= 0) s += __expf(lg[r] - bv);
    s = g::block_sum(s, c.sm + 2 * kW);
    const size_t b3 = 3 * (size_t)blockIdx.x;
    if (t == 0) part[b3] = __int_as_float(br), part[b3 + 1] = bv, part[b3 + 2] = s;
}

__device__ void old_pick_merge(const g::MegaOp& op, const g::Ctx& c, const float* part) {
    const int t = threadIdx.x, G = gridDim.x;
    float bv = -INFINITY;
    int br = 1 << 30;
    const size_t t3 = 3 * (size_t)t;
    if (t < G) bv = part[t3 + 1], br = __float_as_int(part[t3]);
    g::pick_reduce(bv, br, c.sm, (int*)(c.sm + kW));
    float s = t < G && part[t3 + 1] > -INFINITY ? part[t3 + 2] * __expf(part[t3 + 1] - bv) : 0.0f;
    s = g::block_sum(s, c.sm + 2 * kW);
    float* out = (float*)op.p[3];
    if (t == 0) out[0] = __int_as_float(br), out[1] = bv, out[2] = s;
}

template <bool NEW>
__global__ void __launch_bounds__(g::THREADS) k_pick(g::MegaOp op) {
    __shared__ float sm[g::SM_FLOATS];
    g::Ctx c{};
    c.sm = sm, c.nb = 1;
    float* part = (float*)op.p[2];
    if (op.i[2] == 0) {
        if (NEW)
            g::mtp_pick_part(op, c, part);
        else
            old_pick_part(op, c, part);
    } else if (blockIdx.x == 0) {
        if (NEW)
            g::mtp_pick_merge(op, c, part);
        else
            old_pick_merge(op, c, part);
    }
}

struct Pick {
    int row, nan_blocks;
    float max, sum;
};

Pick run_pick(bool use_new, const std::vector<float>& lg, const std::vector<int32_t>* map) {
    const int n = (int)lg.size();
    DeviceBuffer d_lg(lg.size() * sizeof(float)), d_map(map ? map->size() * sizeof(int32_t) : 4), d_part(size_t{768}),
        d_out(size_t{16});
    CHECK(hipMemcpy(d_lg.get(), lg.data(), lg.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(d_out.get(), 0, 16));
    if (map) CHECK(hipMemcpy(d_map.get(), map->data(), map->size() * 4, hipMemcpyHostToDevice));
    g::MegaOp op;
    op.i[0] = n, op.i[1] = n;
    op.p[0] = d_lg.get(), op.p[1] = map ? d_map.get() : nullptr, op.p[2] = d_part.get(), op.p[3] = d_out.get();
    for (int phase = 0; phase < 2; phase++) {
        op.i[2] = phase;
        if (use_new)
            k_pick<true><<<GRID, g::THREADS>>>(op);
        else
            k_pick<false><<<GRID, g::THREADS>>>(op);
        CHECK(hipDeviceSynchronize());
    }
    float out[4];
    CHECK(hipMemcpy(out, d_out.get(), 16, hipMemcpyDeviceToHost));
    Pick p{};
    std::memcpy(&p.row, &out[0], 4);
    p.max = out[1], p.sum = out[2];
    std::memcpy(&p.nan_blocks, &out[3], 4);
    return p;
}

std::vector<float> normal_row(int n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, 3.0f);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

void test_pick_finite_matches_old() {
    for (int n : {1000, 5000, 41000}) {
        std::vector<float> v = normal_row(n, (uint32_t)n);
        v[n / 3] = v[n / 2] = *std::max_element(v.begin(), v.end()) + 1.0f;  // a tie: the smaller row wins
        const Pick a = run_pick(false, v, nullptr), b = run_pick(true, v, nullptr);
        EXPECT(a.row == b.row && a.row == n / 3);
        EXPECT(same_bits(a.max, b.max) && same_bits(a.sum, b.sum));
        EXPECT(b.nan_blocks == 0);
    }
}

void test_pick_with_map_matches_old() {
    const int n = 3000;
    std::vector<float> v = normal_row(n, 7);
    std::vector<int32_t> map(n);
    for (int i = 0; i < n; i++) map[i] = i % 5 == 0 ? -1 : i;
    const Pick a = run_pick(false, v, &map), b = run_pick(true, v, &map);
    EXPECT(a.row == b.row && map[b.row] >= 0 && b.nan_blocks == 0);
    EXPECT(same_bits(a.sum, b.sum));
}

void test_pick_one_nan() {
    const int n = 5000;
    std::vector<float> v = normal_row(n, 3);
    const Pick clean = run_pick(true, v, nullptr);
    v[clean.row == 17 ? 18 : 17] = std::nanf("");  // one NaN: the best of the other rows, the NaN counted
    const Pick p = run_pick(true, v, nullptr);
    EXPECT(p.row == clean.row && p.nan_blocks == 1 && std::isfinite(p.sum) && p.sum > 0);
}

void test_pick_all_nan() {
    const int n = 5000;
    // every row NaN: some row of the head, finite sum, every block with rows counted; the old pick answered row n
    EXPECT(run_pick(false, std::vector<float>(n, std::nanf("")), nullptr).row >= n);
    Pick p = run_pick(true, std::vector<float>(n, std::nanf("")), nullptr);
    EXPECT(p.row >= 0 && p.row < n && std::isfinite(p.sum) && p.sum >= 1 && p.nan_blocks == GRID);
    // every row NaN, the first rows are free: a row with a token
    std::vector<int32_t> map(n, -1);
    for (int i = 700; i < n; i++) map[i] = i;
    p = run_pick(true, std::vector<float>(n, std::nanf("")), &map);
    EXPECT(p.row >= 700 && p.row < n);
}

void test_pick_inf() {
    const int n = 5000;
    const float inf = INFINITY;
    std::vector<float> v = normal_row(n, 4);
    v[4000] = v[100] = inf;  // the first +inf row wins, the sum is a number (1 for the winner)
    Pick p = run_pick(true, v, nullptr);
    EXPECT(p.row == 100 && std::isfinite(p.sum) && p.sum >= 1 && p.nan_blocks == 0);
    p = run_pick(true, std::vector<float>(n, inf), nullptr);
    EXPECT(p.row == 0 && std::isfinite(p.sum) && p.sum >= 1);
    p = run_pick(true, std::vector<float>(n, -inf), nullptr);
    EXPECT(p.row >= 0 && p.row < n && std::isfinite(p.sum) && p.sum >= 1);
}

// op_route

constexpr int E = 64, K = 8;

__global__ void __launch_bounds__(g::THREADS) k_route(g::MegaOp op, g::MegaParams* P, int nb) {
    __shared__ float sm[g::SM_FLOATS];
    g::Ctx c{};
    c.sm = sm, c.nb = nb, c.P = P, c.lane = threadIdx.x % g::WAVE;
    g::op_route(op, c);
}

struct Route {
    int n_u;
    int slot[g::MEGA_UMAX];
    float ew[kB * kU];
    int nan_total;
};

Route run_route(const std::vector<float>& logits, int nb, bool all) {
    DeviceBuffer d_lg(logits.size() * 4), d_flags(kB * 4), d_P(sizeof(g::MegaParams));
    CHECK(hipMemcpy(d_lg.get(), logits.data(), logits.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(d_flags.get(), 0x7f, kB * 4));  // stale counts must be overwritten
    g::MegaParams P{};
    P.nseq = all ? nb : 0;
    CHECK(hipMemcpy(d_P.get(), &P, sizeof P, hipMemcpyHostToDevice));
    g::MegaOp op;
    op.i[0] = E, op.i[1] = K;
    op.p[0] = d_lg.get(), op.p[1] = d_flags.get();
    k_route<<<1, g::THREADS>>>(op, d_P.get<g::MegaParams>(), nb);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&P, d_P.get(), sizeof P, hipMemcpyDeviceToHost));
    Route r{};
    r.n_u = P.n_u;
    std::copy(P.slot, P.slot + g::MEGA_UMAX, r.slot);
    std::copy(P.ew, P.ew + kB * kU, r.ew);
    int flags[g::MEGA_BMAX];
    CHECK(hipMemcpy(flags, d_flags.get(), sizeof flags, hipMemcpyDeviceToHost));
    for (int f : flags) r.nan_total += f;
    return r;
}

bool route_valid(const Route& r) {
    if (r.n_u < 1 || r.n_u > g::MEGA_UMAX) return false;
    for (int u = 0; u < r.n_u; u++)
        if (r.slot[u] < 0 || r.slot[u] >= E) return false;
    return true;
}

void test_route_finite_and_one_nan() {
    const float nan = std::nanf("");
    // finite row: the K largest, weights p / sum p of the chosen
    std::vector<float> v = normal_row(E, 11);
    Route r = run_route(v, 1, false);
    std::vector<int> idx(E);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) { return v[a] > v[b]; });
    const std::set<int> want(idx.begin(), idx.begin() + K), got(r.slot, r.slot + r.n_u);
    EXPECT(route_valid(r) && want == got && r.nan_total == 0);
    double wsum = 0;
    for (int u = 0; u < r.n_u; u++) wsum += r.ew[u];
    EXPECT(std::fabs(wsum - 1.0) < 1e-5);
    // one NaN: never chosen, counted once
    v[idx[0]] = nan;
    r = run_route(v, 1, false);
    EXPECT(route_valid(r) && std::find(r.slot, r.slot + r.n_u, idx[0]) == r.slot + r.n_u && r.nan_total == 1);
}

std::vector<std::vector<float>> bad_rows() {
    const float nan = std::nanf(""), inf = INFINITY;
    std::vector<std::vector<float>> bad = {std::vector<float>(E, nan), std::vector<float>(E, inf),
                                           std::vector<float>(E, -inf)};
    std::vector<float> few(E, nan);  // only K - 1 numbers
    for (size_t i = 0; i + 1 < (size_t)K; i++) few[i * 7] = (float)i;
    bad.push_back(few);
    std::vector<float> mixed = normal_row(E, 12);
    mixed[3] = inf, mixed[9] = -inf;
    bad.push_back(mixed);
    return bad;
}

void test_route_bad_rows() {
    const auto bad = bad_rows();
    for (size_t i = 0; i < bad.size(); i++) {
        const Route r = run_route(bad[i], 1, false);
        bool finite = true;
        for (int u = 0; u < r.n_u; u++) finite = finite && std::isfinite(r.ew[u]);
        std::printf("route case %zu: n_u %d, NaN counted %d\n", i, r.n_u, r.nan_total);
        EXPECT(route_valid(r) && finite);
    }
    EXPECT(run_route(bad[0], 1, false).nan_total == E);
    // several tokens (the MTP of several sequences): one NaN row among them, counted
    std::vector<float> rows;
    for (int b = 0; b < 4; b++) {
        const std::vector<float> w = b == 2 ? std::vector<float>(E, std::nanf("")) : normal_row(E, 20 + b);
        rows.insert(rows.end(), w.begin(), w.end());
    }
    const Route r = run_route(rows, 4, true);
    EXPECT(route_valid(r) && r.nan_total == E);
}

// the copy of the router logits into the mailbox (op_host_wait)

__global__ void __launch_bounds__(g::THREADS) k_copy(const float* src, float* dst, int nb, int E, int* nan_total) {
    const float n = g::copy_logits_clamped<g::MEGA_MAX_EXPERTS>(dst, src, nb, E);
    const float w = g::wave_sum(n);
    if (threadIdx.x % g::WAVE == 0) atomicAdd(nan_total, (int)w);
}

void test_mailbox_copy() {
    const int nb = 3, E = 100;
    const float nan = std::nanf(""), inf = INFINITY;
    std::vector<float> src = normal_row(nb * 256, 31);
    const std::vector<float> orig = src;
    src[5] = nan, src[256 + 99] = nan;  // real logits
    src[256 + 150] = nan;               // padding: not counted
    src[2 * 256 + 7] = inf, src[2 * 256 + 8] = -inf;
    DeviceBuffer d_src(src.size() * 4), d_dst(src.size() * 4), d_n(4);
    CHECK(hipMemcpy(d_src.get(), src.data(), src.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(d_n.get(), 0, 4));
    k_copy<<<1, g::THREADS>>>(d_src.get<float>(), d_dst.get<float>(), nb, E, d_n.get<int>());
    CHECK(hipDeviceSynchronize());
    std::vector<float> dst(src.size());
    int n = -1;
    CHECK(hipMemcpy(dst.data(), d_dst.get(), dst.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(&n, d_n.get(), 4, hipMemcpyDeviceToHost));
    EXPECT(n == 2);
    EXPECT(dst[5] == -FLT_MAX && dst[256 + 99] == -FLT_MAX && dst[256 + 150] == -FLT_MAX);
    EXPECT(dst[2 * 256 + 7] == FLT_MAX && dst[2 * 256 + 8] == -FLT_MAX);
    for (size_t i = 0; i < src.size(); i++)
        if (std::isfinite(orig[i]) && std::isfinite(src[i])) EXPECT(same_bits(dst[i], src[i]));
}

}  // namespace

int main() {
    test_mailbox_copy();
    test_pick_finite_matches_old();
    test_pick_with_map_matches_old();
    test_pick_one_nan();
    test_pick_all_nan();
    test_pick_inf();
    test_route_finite_and_one_nan();
    test_route_bad_rows();
    std::printf("test_gpu_mtp_nan: %s\n", fails ? "FAILED" : "ok");
    return fails ? 1 : 0;
}
