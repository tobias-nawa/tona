// Prefill attention kernels against each other (test_gpu_fattn --prefill-cmp and --prefill-bench): the wave kernel
// (gpu/fattn_wave.h) and the tile kernels must give the same bits for every group size and ragged token count; the
// bench times them in one process and reports the TFLOPS of the causal pairs against the WMMA peak. Included by
// test_gpu_fattn.hip after randoms, Dev and Shape; not a stand-alone header.
#pragma once

#include <cstring>

namespace {

using gpu::FattnPrefillKernel;

constexpr double WMMA_PEAK_TFLOPS = 93.0;  // measured f16 WMMA peak (docs/backends/reference-platform.md)

// Buffers of one prefill call
struct PrefillBuffers {
    Dev q, gate;
    DeviceBuffer o;
    gpu::FattnPrefillArgs a;
    PrefillBuffers(const Shape& s, const RandomCache& cache, int64_t pos0, int tokens)
        : q(randoms((size_t)tokens * s.os(), 31 + (uint32_t)tokens, 2.0f)),
          gate(randoms((size_t)tokens * s.os(), 32, 2.0f)),
          o((size_t)tokens * s.os() * 2),
          a{q.get<float>(),
            cache.get(),
            cache.row(),
            s.cfg.gate ? gate.get<float>() : nullptr,
            o.get<_Float16>(),
            pos0,
            tokens,
            s.nh,
            s.nkv,
            s.cfg.hd,
            1.0f / std::sqrt((float)s.cfg.hd)} {}
};

// Same bits from the wave kernel and both tile kernels
bool cmp_prefill(const Shape& s, const RandomCache& cache, int64_t pos0, int tokens) {
    PrefillBuffers b(s, cache, pos0, tokens);
    const size_t n = (size_t)tokens * s.os();
    std::vector<_Float16> ref;
    bool same = true;
    for (FattnPrefillKernel k :
         {FattnPrefillKernel::multi_tile, FattnPrefillKernel::single_tile, FattnPrefillKernel::wave}) {
        CHECK(hipMemset(b.o.get(), 0xff, n * 2));
        if (!gpu::launch_fattn_prefill(b.a, nullptr, k)) return false;
        CHECK(hipDeviceSynchronize());
        const auto got = download<_Float16>(b.o, n);
        if (ref.empty()) ref = got;
        same = same && std::memcmp(got.data(), ref.data(), n * 2) == 0;
    }
    printf("prefill bits hd %3d nh %2d nkv %d pos0 %6lld x %4d tokens: wave = tiles: %s\n", s.cfg.hd, s.nh, s.nkv,
           (long long)pos0, tokens, same ? "yes" : "NO");
    return same;
}

bool cmp_all() {
    bool ok = true;
    for (const auto& cfg : {kOrnith, kBolt})
        for (int nh : {2, 4, 8, 16, 32}) {
            const Shape s{cfg, nh, 2};
            RandomCache cache(12000, s.nkv, cfg.hd, 77);
            for (const auto& c :
                 {std::pair<int64_t, int>{0, 1}, {0, 75}, {100, 130}, {5000, 33}, {0, 1000}, {3000, 4097}})
                ok = cmp_prefill(s, cache, c.first, c.second) && ok;
        }
    printf(ok ? "prefill bits all equal\n" : "prefill bits DIFFER\n");
    return ok;
}

// Median milliseconds of `reps` launches of one kernel
double prefill_ms(PrefillBuffers& b, FattnPrefillKernel k, int reps) {
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)), CHECK(hipEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep <= reps; rep++) {
        CHECK(hipEventRecord(e0));
        gpu::launch_fattn_prefill(b.a, nullptr, k);
        CHECK(hipEventRecord(e1)), CHECK(hipEventSynchronize(e1));
        float t = 0;
        CHECK(hipEventElapsedTime(&t, e0, e1));
        if (rep > 0) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return ms[ms.size() / 2];
}

int prefill_bench() {
    for (const auto& sh : {Shape{kOrnith, 16, 2}, Shape{kBolt, 32, 2}})
        for (int tokens : {2048, 16384, 32768}) {
            RandomCache cache(tokens + 64, sh.nkv, sh.cfg.hd, 97);
            PrefillBuffers b(sh, cache, 0, tokens);
            // QK^T and PV: 2 * hd flops each per (query head, key) pair, causal pairs only
            const double flop = 4.0 * sh.cfg.hd * sh.nh * ((double)tokens * (tokens + 1) / 2);
            for (const auto& [name, k] : {std::pair{"multi_tile", FattnPrefillKernel::multi_tile},
                                          std::pair{"wave", FattnPrefillKernel::wave}}) {
                const double ms = prefill_ms(b, k, 5);
                printf(
                    "prefill hd %3d nh %2d %6d tokens %-10s: %9.3f ms, %6.2f us per token, %5.2f TFLOPS (%4.1f %% of "
                    "peak)\n",
                    sh.cfg.hd, sh.nh, tokens, name, ms, 1000.0 * ms / tokens, flop / ms / 1e9,
                    100.0 * flop / ms / 1e9 / WMMA_PEAK_TFLOPS);
            }
        }
    return 0;
}

}  // namespace
