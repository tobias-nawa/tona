// k_attn_wmma (gpu/prefill_attn.hip), both gathers (direct and pipelined), against its frozen predecessor
// (tests/attn_wmma_ref.h), byte for byte: the gather must not change a single bit of the output. Random int8 KV cache
// in VRAM or in pinned host memory, selections of every shape (random, drifting like neighbouring tokens, repeated,
// ragged lengths with repeated cells, the dense path below the top-k), positions from 100 to 256K. Needs the GPU, no
// model. Usage: test_gpu_attn_wmma [--bench]   (--bench: time per launch of both kernels, selections that drift slowly,
// fast or not at all, at several positions)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "attn_wmma_ref.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;
namespace gpu = tona::gpu;

namespace {

constexpr int NH = 24, NKV = 2, HD = 256, BLK = 4, TOPK = 2048;
constexpr int ROW = 2 * NKV * HD + NKV * 16 * 4;  // K0 V0 K1 V1 (int8), then 2 x 16 f32 scales
constexpr int MAXC = gpu::MEGA_MAX_CELLS;

enum class Sel { Random, Drift, Repeat, Ragged, Dense, Churn };

const char* name_of(Sel s) {
    static const char* const n[] = {"random", "drift", "repeat", "ragged", "dense", "churn"};
    return n[(int)s];
}

__device__ uint32_t mix(uint32_t x) {
    x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
    return x;
}

// int8 payload from a hash, scales in [0.002, 0.05): the range of the real cache
__global__ void k_fill_kv(uint8_t* kv, int64_t rows, uint32_t seed) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * ROW) return;
    const int64_t r = i / ROW;
    const int col = (int)(i % ROW);
    const uint32_t h = mix((uint32_t)i * 2654435761U ^ seed ^ (uint32_t)(r >> 32));
    if (col < 2 * NKV * HD) {
        kv[i] = (uint8_t)(int8_t)((int)(h % 255) - 127);
    } else if ((col - 2 * NKV * HD) % 4 == 0) {
        reinterpret_cast<float*>(kv + i)[0] = 0.002f + 0.048f * (float)(h & 0xffff) / 65536.0f;
    }
}

// One selection for a token at position pos: ascending cells, blocks of BLK, then the tail not yet pooled
void select_blocks(std::mt19937& rng, int64_t pos, const std::vector<int32_t>& base, std::vector<int32_t>& out,
                   int drop) {
    const int64_t n_pool = (pos + 1) / BLK;
    const int n_top = (int)std::min<int64_t>(n_pool, TOPK / BLK);
    std::set<int32_t> blocks;
    for (int32_t b : base)
        if (b < n_pool && (int)(rng() % 10) >= drop) blocks.insert(b);
    while ((int)blocks.size() < n_top) blocks.insert((int32_t)(rng() % n_pool));
    while ((int)blocks.size() > n_top) blocks.erase(std::next(blocks.begin(), rng() % blocks.size()));
    for (int32_t b : blocks)
        for (int k = 0; k < BLK; k++) out.push_back(b * BLK + k);
    for (int64_t p = n_pool * BLK; p <= pos; p++) out.push_back((int32_t)p);
}

struct Case {
    Sel sel;
    int64_t pos0;
    int tokens;
    bool host_kv;
    uint32_t seed;
};

// cells_all (MAXC per token) and ncells of one case
void build_cells(const Case& c, std::vector<int32_t>& cells, std::vector<int32_t>& nc) {
    std::mt19937 rng(c.seed);
    cells.assign((size_t)c.tokens * MAXC, -7);
    nc.assign(c.tokens, 0);
    std::vector<int32_t> base, cur;
    for (int t = 0; t < c.tokens; t++) {
        const int64_t pos = c.pos0 + t;
        const int64_t n_pool = (pos + 1) / BLK;
        cur.clear();
        if (c.sel == Sel::Repeat && t > 0) {
            cur.assign(cells.begin(), cells.begin() + nc[0]);  // the same cells for every token
        } else if (c.sel == Sel::Ragged) {
            const int n = 1 + (int)(rng() % MAXC);
            for (int i = 0; i < n; i++) cur.push_back((int32_t)(rng() % (pos + 1)));  // repeats allowed
        } else {
            if (c.sel == Sel::Random || base.empty() || (c.sel == Sel::Drift && t % 32 == 0) ||
                (c.sel == Sel::Churn && t % 4 == 0)) {
                base.clear();
                for (int i = 0; i < TOPK / BLK && n_pool > 0; i++) base.push_back((int32_t)(rng() % n_pool));
            }
            select_blocks(rng, pos, base, cur, c.sel == Sel::Churn ? 3 : 1);
        }
        nc[t] = (int)cur.size();
        std::copy(cur.begin(), cur.end(), cells.begin() + (size_t)t * MAXC);
    }
}

struct Data {
    uint8_t* kv = nullptr;
    bool host = false;
    ~Data() {
        if (kv) (void)(host ? hipHostFree(kv) : hipFree(kv));
    }
    void alloc(int64_t rows, bool host_kv, uint32_t seed) {
        host = host_kv;
        if (host_kv) {
            CHECK(hipHostMalloc(reinterpret_cast<void**>(&kv), (size_t)rows * ROW, hipHostMallocDefault));
        } else {
            CHECK(hipMalloc(reinterpret_cast<void**>(&kv), (size_t)rows * ROW));
        }
        uint8_t* dev = kv;
        DeviceBuffer tmp(host_kv ? (size_t)rows * ROW : 1);
        if (host_kv)
            dev = tmp.get<uint8_t>();
        else
            CHECK(hipMemset(kv, 0, (size_t)rows * ROW));
        const int64_t total = rows * ROW;
        k_fill_kv<<<(unsigned)((total + 255) / 256), 256>>>(dev, rows, seed);
        CHECK(hipDeviceSynchronize());
        if (host_kv) CHECK(hipMemcpy(kv, dev, (size_t)rows * ROW, hipMemcpyDeviceToHost));
    }
};

std::vector<float> randoms(size_t n, uint32_t seed, float sd) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(0.0f, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

// Everything one launch needs, uploaded once
struct Inputs {
    DeviceBuffer q, gate, cells, nc, o;
    int tokens;
    Inputs(const Case& c, const std::vector<int32_t>& cells_h, const std::vector<int32_t>& nc_h)
        : q((size_t)c.tokens * NH * HD * 4),
          gate((size_t)c.tokens * NH * HD * 4),
          cells(cells_h.size() * 4),
          nc(nc_h.size() * 4),
          o((size_t)c.tokens * NH * HD * 2),
          tokens(c.tokens) {
        const size_t n = (size_t)c.tokens * NH * HD;
        const auto qh = randoms(n, c.seed + 1, 1.0f), gh = randoms(n, c.seed + 2, 2.0f);
        CHECK(hipMemcpy(q.get(), qh.data(), n * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(gate.get(), gh.data(), n * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(cells.get(), cells_h.data(), cells_h.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(nc.get(), nc_h.data(), nc_h.size() * 4, hipMemcpyHostToDevice));
    }
    gpu::AttnWmmaArgs args(const uint8_t* kv, int64_t pos0, bool kv_host) const {
        return {q.get<float>(),
                kv,
                ROW,
                cells.get<int32_t>(),
                nc.get<int32_t>(),
                gate.get<float>(),
                o.get<_Float16>(),
                pos0,
                NH,
                NKV,
                BLK,
                TOPK,
                1.0f / 16.0f,
                kv_host};
    }
};

// mode: -1 the frozen kernel, 0 ... 4 the gathers of launch_attn_wmma_gather (0, 1, 2 the second kernel, 3 and 4 the
// first one: direct and pipelined), 5 the production choice
std::vector<uint16_t> run(const Inputs& in, const Data& d, int64_t pos0, int mode) {
    CHECK(hipMemset(in.o.get(), 0xAB, (size_t)in.tokens * NH * HD * 2));
    const gpu::AttnWmmaArgs a = in.args(d.kv, pos0, d.host);
    if (mode < 0)
        gpu::ref::launch_attn_wmma_ref(a, in.tokens, nullptr);
    else if (mode == 9)
        gpu::launch_attn_wmma(a, in.tokens, nullptr);
    else
        gpu::launch_attn_wmma_gather(a, in.tokens, nullptr, mode);
    CHECK(hipDeviceSynchronize());
    std::vector<uint16_t> out((size_t)in.tokens * NH * HD);
    CHECK(hipMemcpy(out.data(), in.o.get(), out.size() * 2, hipMemcpyDeviceToHost));
    return out;
}

bool run_case(const Case& c, const Data& d) {
    std::vector<int32_t> cells, nc;
    build_cells(c, cells, nc);
    const Inputs in(c, cells, nc);
    const auto want = run(in, d, c.pos0, -1);
    size_t bad = 0, first = 0;
    int bad_mode = -1;
    std::vector<uint16_t> got;
    for (int mode : {0, 1, 2, 3, 4, 5, 9}) {
        const auto out = run(in, d, c.pos0, mode);
        for (size_t i = 0; i < want.size(); i++)
            if (want[i] != out[i] && bad++ == 0) first = i, bad_mode = mode, got = out;
    }
    int nan = 0;
    for (uint16_t v : want) nan += (v & 0x7c00) == 0x7c00;
    printf("%-7s pos0 %7lld x %3d tokens kv %-4s: %s%s\n", name_of(c.sel), (long long)c.pos0, c.tokens,
           c.host_kv ? "host" : "vram", bad ? "DIFFERENT" : "identical", nan ? " (reference has inf/nan!)" : "");
    if (bad)
        printf("  %zu of %zu halves differ, first at %zu (mode %d): want %04x got %04x\n", bad, want.size() * 7, first,
               bad_mode, want[first], got[first]);
    return bad == 0 && nan == 0;
}

// Median and minimum launch time in ms; mode as in run()
void time_launch(const Inputs& in, const Data& d, int64_t pos0, int mode, double (&res)[2]) {
    std::vector<double> ms;
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    const gpu::AttnWmmaArgs a = in.args(d.kv, pos0, d.host);
    for (int rep = 0; rep < 8; rep++) {
        CHECK(hipEventRecord(e0));
        if (mode < 0)
            gpu::ref::launch_attn_wmma_ref(a, in.tokens, nullptr);
        else
            gpu::launch_attn_wmma_gather(a, in.tokens, nullptr, mode);
        CHECK(hipEventRecord(e1));
        CHECK(hipEventSynchronize(e1));
        float t;
        CHECK(hipEventElapsedTime(&t, e0, e1));
        if (rep > 0) ms.push_back(t);  // first launch warms the caches and the clocks
    }
    std::sort(ms.begin(), ms.end());
    res[0] = ms[ms.size() / 2], res[1] = ms[0];
    (void)hipEventDestroy(e0);
    (void)hipEventDestroy(e1);
}

// us per token-layer (median) of the frozen kernel, the first kernel (direct, pipelined) and the three gathers of the
// second one, alternating twice so that clock drift shows
void bench(Sel sel, int64_t pos0, bool host_kv) {
    const int tokens = 1024;
    const Case c{sel, pos0, tokens, host_kv, 99};
    Data d;
    d.alloc(pos0 + tokens, host_kv, 5);
    std::vector<int32_t> cells, nc;
    build_cells(c, cells, nc);
    const Inputs in(c, cells, nc);
    const int modes[] = {-1, 3, 4, 0, 1, 2, 5};
    double best[7][2];
    for (int round = 0; round < 2; round++)
        for (int k = 0; k < 7; k++) {
            double r[2];
            time_launch(in, d, pos0, modes[k], r);
            if (round == 0 || r[0] < best[k][0]) best[k][0] = r[0], best[k][1] = r[1];
        }
    const double per = 1000.0 / tokens;  // ms per launch -> us per token-layer
    printf(
        "%-6s pos %7lld kv %-4s: frozen %.2f  v1 direct %.2f piped %.2f | v2 late %.2f withK %.2f ahead %.2f Kahead "
        "%.2f (us per token-layer)\n",
        name_of(sel), (long long)pos0, host_kv ? "host" : "vram", best[0][0] * per, best[1][0] * per, best[2][0] * per,
        best[3][0] * per, best[4][0] * per, best[5][0] * per, best[6][0] * per);
}

// The dequantization of the second kernel, fma(x + 128, s, -128 s) rounded to f16, against (float)x * s rounded to f16,
// for every int8 value and scales over the range of the cache (and beyond)
__global__ void k_dequant_check(const float* scales, int n, unsigned* bad) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float s = scales[i];
    for (int x = -128; x < 128; x++) {
        const _Float16 want = (_Float16)((float)x * s);
        const float xf = (float)((uint32_t)(x + 128) & 0xffu);
        const _Float16 got = (_Float16)__builtin_fmaf(xf, s, -128.0f * s);
        if (*reinterpret_cast<const uint16_t*>(&want) != *reinterpret_cast<const uint16_t*>(&got)) atomicAdd(bad, 1u);
    }
}

bool check_dequant() {
    const int n = 1 << 21;
    const size_t bytes = (size_t)n * 4;
    std::vector<float> sc(n);
    std::mt19937 rng(77);
    for (int i = 0; i < n; i++) {
        const uint32_t m = rng() & 0x7fffff, e = 100 + rng() % 30;  // 2^-27 ... 2^2
        const uint32_t bits = (e << 23) | m;
        std::memcpy(&sc[i], &bits, 4);
    }
    DeviceBuffer d(bytes), bad(4);
    CHECK(hipMemcpy(d.get(), sc.data(), bytes, hipMemcpyHostToDevice));
    CHECK(hipMemset(bad.get(), 0, 4));
    k_dequant_check<<<n / 256, 256>>>(d.get<float>(), n, bad.get<unsigned>());
    CHECK(hipDeviceSynchronize());
    unsigned nbad = 0;
    CHECK(hipMemcpy(&nbad, bad.get(), 4, hipMemcpyDeviceToHost));
    printf("dequantization by fma: %u of %lld cases differ\n", nbad, (long long)n * 256);
    return nbad == 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (Sel sel : {Sel::Drift, Sel::Churn, Sel::Random})
            for (int64_t p : {4000, 60000, 128000})
                for (bool h : {false, true}) bench(sel, p, h);
        return 0;
    }
    bool ok = check_dequant();
    const int64_t rows = 262144 + 64;
    for (bool host : {false, true}) {
        Data d;
        d.alloc(rows, host, host ? 11 : 12);
        const Case cases[] = {
            {Sel::Dense, 0, 40, host, 1},       {Sel::Dense, 100, 70, host, 2},      {Sel::Dense, 2000, 40, host, 3},
            {Sel::Random, 2060, 64, host, 4},   {Sel::Drift, 2051, 130, host, 5},    {Sel::Drift, 9000, 100, host, 6},
            {Sel::Repeat, 33000, 40, host, 7},  {Sel::Ragged, 5000, 100, host, 8},   {Sel::Ragged, 120000, 50, host, 9},
            {Sel::Drift, 131000, 96, host, 10}, {Sel::Random, 200000, 64, host, 11}, {Sel::Drift, 262000, 64, host, 12},
        };
        for (const Case& c : cases) ok = run_case(c, d) && ok;
    }
    printf(ok ? "all identical\n" : "FAILED\n");
    return ok ? 0 : 1;
}
