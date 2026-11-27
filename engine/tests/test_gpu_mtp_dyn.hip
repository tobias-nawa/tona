// The steps of the dynamic MTP head rows (gpu/mega_op_mtp_dyn.h) against a CPU reference: projection, scores and
// selection per partition (valid, unique, not excluded rows; every row clearly above the boundary is in) and the
// rows copied for the selected tokens. Needs the GPU, no model.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "gpu/mega_op_mtp.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;
namespace g = tona::gpu;

namespace {

int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

constexpr int GRID = 5;  // fewer blocks than partitions: a block takes several
constexpr int R = 32;    // rank of the test scorer

__global__ void __launch_bounds__(g::THREADS) k_step(g::MegaOp op, int kind) {
    __shared__ float sm[g::SM_FLOATS];
    g::Ctx c{};
    c.sm = sm, c.nb = 1, c.lane = threadIdx.x % g::WAVE;
    c.gw = (int64_t)blockIdx.x * g::MWAVES + threadIdx.x / g::WAVE, c.GW = (int64_t)gridDim.x * g::MWAVES;
    if (kind == 0) g::op_mtp_proj(op, c);
    if (kind == 1) g::op_mtp_dyn(op, c);
}

struct Scorer {
    std::vector<uint8_t> tiles;
    std::vector<float> scale, score;  // score: the reference per row in tile order
    std::vector<int32_t> ids;
    std::vector<uint8_t> excl;
    size_t pad = 0;
};

// n candidates of random int8 rows, tiles as the engine lays them out; excl: every 7th row
Scorer make_scorer(size_t n, const std::vector<float>& p, uint32_t seed) {
    std::mt19937 rng(seed);
    Scorer s;
    s.pad = (n + 31) / 32 * 32;
    s.tiles.assign(s.pad * R, 0), s.scale.assign(s.pad, 0), s.ids.assign(s.pad, -1), s.excl.assign(s.pad, 0);
    s.score.assign(s.pad, -1e30f);
    for (size_t i = 0; i < n; i++) {
        s.ids[i] = (int32_t)(1000 + i), s.scale[i] = 0.01f + 0.001f * (float)(rng() % 10), s.excl[i] = i % 7 == 3;
        double acc = 0;
        for (int k = 0; k < R; k++) {
            const int8_t v = (int8_t)((int)(rng() % 255) - 127);
            s.tiles[(((i / 32) * (R / 16) + k / 16) * 32 + i % 32) * 16 + k % 16] = (uint8_t)v;
            acc += (double)v * p[k];
        }
        s.score[i] = (float)(acc * s.scale[i]);
    }
    return s;
}

std::vector<float> run_proj(const std::vector<float>& bt, const std::vector<float>& x, int r, int d) {
    DeviceBuffer d_b(bt.size() * 4), d_x(x.size() * 4), d_o((size_t)r * 4);
    CHECK(hipMemcpy(d_b.get(), bt.data(), bt.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_x.get(), x.data(), x.size() * 4, hipMemcpyHostToDevice));
    g::MegaOp op;
    op.p[0] = d_b.get(), op.p[1] = d_x.get(), op.p[2] = d_o.get(), op.i[0] = r, op.i[1] = d, op.i[2] = d;
    k_step<<<GRID, g::THREADS>>>(op, 0);
    CHECK(hipDeviceSynchronize());
    std::vector<float> o(r);
    CHECK(hipMemcpy(o.data(), d_o.get(), o.size() * 4, hipMemcpyDeviceToHost));
    return o;
}

void test_proj() {
    const int r = 40, d = 2560;
    std::mt19937 rng(1);
    std::normal_distribution<float> n;
    std::vector<float> bt((size_t)r * d), x(d);
    for (float& v : bt) v = n(rng);
    for (float& v : x) v = n(rng);
    const std::vector<float> o = run_proj(bt, x, r, d);
    for (int j = 0; j < r; j++) {
        double ref = 0;
        for (int k = 0; k < d; k++) ref += (double)bt[(size_t)j * d + k] * x[k];
        EXPECT(std::fabs(ref - o[j]) < 1e-3 * (1 + std::fabs(ref)));
    }
}

constexpr int WORDS = 5;  // words per row of the test head

struct Selected {
    std::vector<int32_t> slots;
    std::vector<uint32_t> rows;
};

// Row t of the test head: word k = t * WORDS + k
Selected run_select(const Scorer& s, const std::vector<float>& p, int m) {
    const size_t nslots = (size_t)g::MTP_PARTS * m, nrows = 1000 + s.pad;
    std::vector<uint32_t> head(nrows * WORDS);
    for (size_t i = 0; i < head.size(); i++) head[i] = (uint32_t)i;
    DeviceBuffer d_t(s.tiles.size()), d_s(s.pad * 4), d_i(s.pad * 4), d_e(s.pad), d_p(p.size() * 4), d_o(nslots * 4),
        d_h(head.size() * 4), d_r(nslots * WORDS * 4);
    CHECK(hipMemcpy(d_t.get(), s.tiles.data(), s.tiles.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_s.get(), s.scale.data(), s.pad * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_i.get(), s.ids.data(), s.pad * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_e.get(), s.excl.data(), s.pad, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_p.get(), p.data(), p.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_h.get(), head.data(), head.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemset(d_r.get(), 0xff, nslots * WORDS * 4));
    g::MegaOp op;
    op.p[0] = d_t.get(), op.p[1] = d_s.get(), op.p[2] = d_i.get(), op.p[3] = d_e.get(), op.p[4] = d_p.get();
    op.p[5] = d_o.get(), op.p[6] = d_h.get(), op.p[7] = d_r.get();
    op.i[0] = (int)(s.pad / 32), op.i[1] = R, op.i[3] = m, op.i[4] = WORDS;
    k_step<<<GRID, g::THREADS>>>(op, 1);
    CHECK(hipDeviceSynchronize());
    Selected out{std::vector<int32_t>(nslots), std::vector<uint32_t>(nslots * WORDS)};
    CHECK(hipMemcpy(out.slots.data(), d_o.get(), nslots * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(out.rows.data(), d_r.get(), nslots * WORDS * 4, hipMemcpyDeviceToHost));
    return out;
}

// Partition of candidate row i: its tile modulo the number of partitions
int part_of(size_t i) { return (int)((i / 32) % g::MTP_PARTS); }

// The row of every selected token is in its slot (a free slot keeps what the buffer held)
void check_rows(const Selected& sel, int m, int part) {
    for (int k = 0; k < m; k++) {
        const size_t slot = (size_t)part * m + k;
        for (int w = 0; w < WORDS; w++)
            EXPECT(sel.rows[slot * WORDS + w] ==
                   (sel.slots[slot] < 0 ? 0xffffffffu : (uint32_t)(sel.slots[slot] * WORDS + w)));
    }
}

// Valid candidates of the partition, best first
std::vector<std::pair<float, size_t>> valid_of(const Scorer& s, int part) {
    std::vector<std::pair<float, size_t>> valid;
    for (size_t i = 0; i < s.pad; i++)
        if (part_of(i) == part && s.ids[i] >= 0 && !s.excl[i]) valid.emplace_back(s.score[i], i);
    std::sort(valid.begin(), valid.end(), [](auto& a, auto& b) { return a.first > b.first; });
    return valid;
}

// The selection ranks by the upper 16 bits of the score (a bin is 1/128 wide) and takes the rows of the boundary bin in
// the order of the threads: rows within two bins of the boundary may go either way
void check_boundary(const Scorer& s, const std::vector<std::pair<float, size_t>>& valid, const std::set<int32_t>& got,
                    size_t want) {
    const float bound = want ? valid[want - 1].first : 0;
    const float eps = 0.016f * std::fabs(bound) + 1e-3f;
    for (const auto& v : valid) {
        const bool in = got.contains(s.ids[v.second]);
        if (v.first > bound + eps) EXPECT(in);
        if (v.first < bound - eps) EXPECT(!in);
    }
}

void check_partition(const Scorer& s, const Selected& sel, int m, int part) {
    const auto valid = valid_of(s, part);
    const size_t want = std::min<size_t>(m, valid.size());
    std::set<int32_t> got;
    for (int k = 0; k < m; k++) {
        const int32_t id = sel.slots[(size_t)part * m + k];
        if (id >= 0) got.insert(id);
    }
    EXPECT(got.size() == want);
    check_boundary(s, valid, got, want);
    for (int32_t id : got) EXPECT(id >= 1000 && !s.excl[(size_t)id - 1000]);
    check_rows(sel, m, part);
}

void test_select() {
    std::mt19937 rng(5);
    std::normal_distribution<float> n;
    std::vector<float> p(R);
    for (float& v : p) v = n(rng);
    for (size_t cand : {size_t{500}, size_t{20000}, size_t{90001}}) {
        const Scorer s = make_scorer(cand, p, (uint32_t)cand);
        for (int m : {16, 128}) {
            const Selected out = run_select(s, p, m);
            for (int part = 0; part < g::MTP_PARTS; part++) check_partition(s, out, m, part);
        }
    }
}

}  // namespace

int main() {
    test_proj();
    test_select();
    if (fails) return 1;
    std::printf("test_gpu_mtp_dyn: ok\n");
    return 0;
}
