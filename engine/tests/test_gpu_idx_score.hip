// op_idx_score of the megakernel (gpu/mega_op_idx.h) against the loop it replaced (one wave per token and pool block,
// the pool read once per token): every score must match bit for bit, because the top-k reads them as integers.
// Needs the GPU, no model.
// Usage: test_gpu_idx_score [--bench]   (--bench: microseconds per launch of the new step, 3 tokens)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "gpu/mega_op_idx.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;

namespace {

constexpr int HEADS = 4, DIM = 128, BLK = 4, TOPK = 2048;
constexpr int GRID = 32;  // blocks of k_mega
constexpr uint32_t SENTINEL = 0xdeadbeef;

// The step as it was before the pool was read once per pass: a wave per (token, pool block)
__device__ void idx_score_per_token(const tona::gpu::MegaOp& op, const tona::gpu::Ctx& c) {
    const float* q = (const float*)op.p[0];
    const float* pool = (const float*)op.p[1];
    float* score = (float*)op.p[2];
    const int heads = op.i[0], dim = op.i[1], blk = op.i[2], sstride = op.i[3];
    const int64_t per = (c.pos + c.nb) / blk;
    for (int64_t u = c.gw; u < c.nb * per; u += c.GW) {
        const int b = (int)(u / per);
        const int64_t pb = u % per, pos = c.pos + b;
        if (!tona::gpu::qsa_select(op, pos) || pb >= (pos + 1) / blk) continue;
        const float* qb = q + (int64_t)b * heads * dim;
        float total = 0;
        for (int h = 0; h < heads; h++) {
            float dot = 0;
            for (int i = c.lane; i < dim; i += tona::gpu::WAVE) dot += qb[h * dim + i] * pool[pb * dim + i];
            total += fmaxf(tona::gpu::wave_sum(dot), 0.0f);
        }
        if (c.lane == 0) score[(int64_t)b * sstride + pb] = total * rsqrtf((float)dim);
    }
}

template <bool NEW>
__global__ void __launch_bounds__(tona::gpu::THREADS) k_idx(tona::gpu::MegaOp op, int64_t pos, int nb) {
    tona::gpu::Ctx c{};
    c.gw = (int64_t)blockIdx.x * tona::gpu::MWAVES + threadIdx.x / tona::gpu::WAVE;
    c.GW = (int64_t)gridDim.x * tona::gpu::MWAVES;
    c.lane = threadIdx.x % tona::gpu::WAVE;
    c.pos = pos, c.nb = nb;
    if (NEW)
        tona::gpu::op_idx_score(op, c);
    else
        idx_score_per_token(op, c);
}

std::vector<uint32_t> run(bool use_new, const std::vector<float>& q, const std::vector<float>& pool, int64_t pos,
                          int nb, int64_t sstride) {
    DeviceBuffer d_q(q.size() * 4), d_pool(pool.size() * 4), d_sc((size_t)nb * sstride * 4);
    std::vector<uint32_t> sc((size_t)nb * sstride, SENTINEL);
    CHECK(hipMemcpy(d_q.get(), q.data(), q.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_pool.get(), pool.data(), pool.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_sc.get(), sc.data(), sc.size() * 4, hipMemcpyHostToDevice));
    tona::gpu::MegaOp op;
    op.i[0] = HEADS, op.i[1] = DIM, op.i[2] = op.i[5] = BLK, op.i[3] = (int32_t)sstride, op.i[7] = TOPK;
    op.p[0] = d_q.get(), op.p[1] = d_pool.get(), op.p[2] = d_sc.get();
    if (use_new)
        k_idx<true><<<GRID, tona::gpu::THREADS>>>(op, pos, nb);
    else
        k_idx<false><<<GRID, tona::gpu::THREADS>>>(op, pos, nb);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(sc.data(), d_sc.get(), sc.size() * 4, hipMemcpyDeviceToHost));
    return sc;
}

// One pass of nb tokens at pos .. pos + nb - 1 with random q and pool; true if old and new scores are identical
bool check(int64_t pos, int nb, uint32_t seed) {
    const int64_t n_pool = (pos + nb) / BLK, sstride = n_pool + 2;
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> q((size_t)nb * HEADS * DIM), pool((size_t)n_pool * DIM);
    for (float& v : q) v = dist(rng);
    for (float& v : pool) v = dist(rng);
    const auto want = run(false, q, pool, pos, nb, sstride), got = run(true, q, pool, pos, nb, sstride);
    const bool ok = std::memcmp(want.data(), got.data(), want.size() * 4) == 0;
    std::printf("idx_score pos %7lld tokens %d: %s\n", (long long)pos, nb, ok ? "ok" : "FAILED");
    return ok;
}

// Microseconds per launch of the step for 3 tokens, median of 20, against the pool read once: the time the pool of
// 32 MB at 250K needs at the 315 GB/s of the reference platform (docs/backends/reference-platform.md) is ~100 us. The
// launch of a kernel of its own adds ~10 us.
void bench(int64_t pos) {
    const int nb = 3;
    const int64_t n_pool = (pos + nb) / BLK, sstride = n_pool + 2;
    std::mt19937 rng(5);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> q((size_t)nb * HEADS * DIM), pool((size_t)n_pool * DIM);
    for (float& v : q) v = dist(rng);
    for (float& v : pool) v = dist(rng);
    DeviceBuffer d_q(q.size() * 4), d_pool(pool.size() * 4), d_sc((size_t)nb * sstride * 4);
    CHECK(hipMemcpy(d_q.get(), q.data(), q.size() * 4, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_pool.get(), pool.data(), pool.size() * 4, hipMemcpyHostToDevice));
    tona::gpu::MegaOp op;
    op.i[0] = HEADS, op.i[1] = DIM, op.i[2] = op.i[5] = BLK, op.i[3] = (int32_t)sstride, op.i[7] = TOPK;
    op.p[0] = d_q.get(), op.p[1] = d_pool.get(), op.p[2] = d_sc.get();
    std::vector<double> us;
    for (int rep = 0; rep < 25; rep++) {
        CHECK(hipDeviceSynchronize());
        const auto t0 = std::chrono::steady_clock::now();
        k_idx<true><<<GRID, tona::gpu::THREADS>>>(op, pos, nb);
        CHECK(hipDeviceSynchronize());
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(us.begin() + 5, us.end());
    std::printf("op_idx_score 3 tokens at pos %7lld: %.0f us per launch (median of 20), pool %.1f MB\n", (long long)pos,
                us[15], (double)pool.size() * 4 / 1e6);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc > 1 && std::string(argv[1]) == "--bench") {
        for (const int64_t pos : {16384, 131072, 262000}) bench(pos);
        return 0;
    }
    int failed = 0;
    // before the first selection (nothing may be written), around it, and up to 256K positions
    for (const int64_t pos : {100, 2051, 2055, 4099, 20000, 65531, 131069, 262111})
        for (int nb = 1; nb <= 4; nb++) failed += !check(pos, nb, 17u + (uint32_t)pos + (uint32_t)nb);
    std::printf("%d cases failed\n", failed);
    return failed ? 1 : 0;
}
