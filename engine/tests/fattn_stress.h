// Repeat-and-scan mode of test_gpu_fattn (--stress N): the same pass N times on the same inputs, every output compared
// bit for bit with the first run and scanned for NaN and infinity. It answers the report of llama.cpp #32178, where the
// HIP flash attention of gfx1200 gave different bits for identical inputs and NaNs where the CPU was finite, in
// shapes with a ragged token count (75, 76), 512 cached positions and 16 query heads per KV head. Included by
// test_gpu_fattn.hip after run_pass and Shape; not a stand-alone header.
#pragma once

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {

struct ScanResult {
    long nan = 0, inf = 0, diff = 0;
    bool bad() const { return nan || inf || diff; }
};

template <class T>
void scan(const std::vector<T>& v, const std::vector<T>& first, ScanResult& r) {
    for (size_t i = 0; i < v.size(); i++) {
        const float x = (float)v[i];
        r.nan += std::isnan(x), r.inf += std::isinf(x);
    }
    r.diff += std::memcmp(v.data(), first.data(), v.size() * sizeof(T)) != 0;
}

// One decode pass of nb tokens at pos0, `reps` times
bool stress_decode(Rig& rig, const Shape& s, RandomCache& cache, int64_t pos0, int nb, int reps) {
    const Inputs in(s, nb, 11 + (uint32_t)pos0);
    const Outputs first = run_pass(rig, s, cache, in, {pos0, 0, nb, 0});
    ScanResult r;
    for (int i = 0; i < reps; i++) {
        const Outputs a = run_pass(rig, s, cache, in, {pos0, 0, nb, 0});
        scan(a.o, first.o, r), scan(a.qn, first.qn, r), scan(a.q8, first.q8, r);
    }
    printf("stress decode  hd %3d nh %2d nkv %d pos0 %5lld x %d: %d runs, NaN %ld, inf %ld, runs differing %ld: %s\n",
           s.cfg.hd, s.nh, s.nkv, (long long)pos0, nb, reps, r.nan, r.inf, r.diff, r.bad() ? "FAILED" : "ok");
    return first.ok && !r.bad();
}

std::vector<_Float16> prefill_once(const Shape& s, const RandomCache& cache, int64_t pos0, int tokens) {
    const size_t n = (size_t)tokens * s.os();
    const auto q = randoms(n, 41 + (uint32_t)pos0, 2.0f), gate = randoms(n, 42, 2.0f);
    const Dev dq(q), dg(gate);
    const DeviceBuffer o(n * 2);
    const gpu::FattnPrefillArgs a{dq.get<float>(),
                                  cache.get(),
                                  cache.row(),
                                  s.cfg.gate ? dg.get<float>() : nullptr,
                                  o.get<_Float16>(),
                                  pos0,
                                  tokens,
                                  s.nh,
                                  s.nkv,
                                  s.cfg.hd,
                                  1.0f / std::sqrt((float)s.cfg.hd)};
    gpu::launch_fattn_prefill(a, nullptr);
    CHECK(hipDeviceSynchronize());
    return download<_Float16>(o, n);
}

bool stress_prefill(const Shape& s, const RandomCache& cache, int64_t pos0, int tokens, int reps) {
    const auto first = prefill_once(s, cache, pos0, tokens);
    ScanResult r;
    for (int i = 0; i < reps; i++) scan(prefill_once(s, cache, pos0, tokens), first, r);
    printf("stress prefill hd %3d nh %2d nkv %d pos0 %5lld x %d: %d runs, NaN %ld, inf %ld, runs differing %ld: %s\n",
           s.cfg.hd, s.nh, s.nkv, (long long)pos0, tokens, reps, r.nan, r.inf, r.diff, r.bad() ? "FAILED" : "ok");
    return !r.bad();
}

// The shapes of the report that our kernel supports (head size 128 and 256, group 16 and 8), ragged token counts
// 75 and 76, 512 positions in total, and a few neighbours
bool stress_all(int reps) {
    bool ok = true;
    const Shape shapes[] = {{kBolt, 32, 2}, {kBolt, 64, 4}, {kOrnith, 16, 2}, {kOrnith, 64, 4}};
    for (const Shape& s : shapes) {
        Rig rig(gpu::probe::fattn_kernels(s.cfg));
        RandomCache cache(1100, s.nkv, s.cfg.hd, 4321);
        for (int nb : {1, 4}) ok = stress_decode(rig, s, cache, 512 - nb, nb, reps) && ok;
        ok = stress_decode(rig, s, cache, 1000, 4, reps) && ok;
        for (int tokens : {75, 76}) ok = stress_prefill(s, cache, 512 - tokens, tokens, reps) && ok;
        ok = stress_prefill(s, cache, 0, 76, reps) && ok;
    }
    printf(ok ? "stress: all ok\n" : "stress: FAILED\n");
    return ok;
}

}  // namespace
