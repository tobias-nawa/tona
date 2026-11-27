// A KV cache split by position (gpu/kv_place.h) must give the bits of the flat one: the decode steps with their K and V
// writes (p[13] / i[7] of the prepare step, p[3] / i[5] of the split step), and the prefill kernel (kv_cold, kv_hot of
// FattnPrefillArgs). The split is at cell T, not a multiple of a tile, and the passes cross it. Included by
// test_gpu_fattn.hip after randoms, Dev, Shape, Inputs and DecodeProgram; not a stand-alone header.
#pragma once

#include <cstring>

namespace {

// The rows [from, to) of the flat cache as a device buffer
Dev rows_of(const HostCache& hc, int row, int64_t from, int64_t to) {
    return Dev(std::vector<uint8_t>(hc.bytes.begin() + from * row, hc.bytes.begin() + to * row));
}

// The decode pass on the flat cache and on the split one: outputs and the rows the pass wrote must be equal
bool split_decode(Rig& rig, const Shape& s, int64_t T, int64_t pos0, int nb) {
    const int row = s.row();
    const int64_t rows = std::max(pos0 + nb + 64, T + 64);
    RandomCache flat(rows, s.nkv, s.cfg.hd, 77);
    const HostCache hc(s, flat);
    const Dev hot = rows_of(hc, row, 0, T), cold = rows_of(hc, row, T, rows);
    const Inputs in(s, nb, 9);
    const Outputs a = run_pass(rig, s, flat, in, {pos0, 0, nb, 0});
    DecodeProgram d(s, flat, in, 0, nb);
    d.prep.p[11] = d.split.p[1] = hot.get(), d.prep.p[13] = d.split.p[3] = cold.get();
    d.prep.i[7] = d.split.i[5] = (int)T;
    const bool ran = rig.run({d.prep, d.split, d.merge}, pass_params(pos0, nb), 0);
    const auto o = download<float>(d.o, (size_t)nb * s.os());
    const auto q8 = download<int8_t>(d.q8, (size_t)nb * s.os());
    bool rows_ok = true;
    for (int t = 0; t < nb; t++) {
        const int64_t p = pos0 + t;
        std::vector<uint8_t> want(row), have(row);
        CHECK(hipMemcpy(want.data(), flat.get() + p * row, row, hipMemcpyDeviceToHost));
        const uint8_t* src = p < T ? hot.get<uint8_t>() + p * row : cold.get<uint8_t>() + (p - T) * row;
        CHECK(hipMemcpy(have.data(), src, row, hipMemcpyDeviceToHost));
        rows_ok = rows_ok && want == have;
    }
    const bool ok = a.ok && ran && o == a.o && q8 == a.q8 && rows_ok;
    printf("split   hd %3d nh %2d at %6lld, pos0 %6lld x %d tokens: outputs and written rows equal: %s\n", s.cfg.hd,
           s.nh, (long long)T, (long long)pos0, nb, ok ? "ok" : "FAILED");
    return ok;
}

// The prefill kernel on the flat cache and on the split one, tokens from pos0
bool split_prefill(const Shape& s, int64_t T, int64_t pos0, int tokens) {
    const int row = s.row();
    const int64_t rows = std::max(pos0 + tokens + 16, T + 16);
    RandomCache flat(rows, s.nkv, s.cfg.hd, 78);
    const HostCache hc(s, flat);
    const Dev hot = rows_of(hc, row, 0, T), cold = rows_of(hc, row, T, rows);
    PrefillBuffers b(s, flat, pos0, tokens);
    if (!gpu::launch_fattn_prefill(b.a, nullptr)) return false;
    CHECK(hipDeviceSynchronize());
    const auto want = download<_Float16>(b.o, (size_t)tokens * s.os());
    gpu::FattnPrefillArgs a = b.a;
    a.kv = hot.get<uint8_t>(), a.kv_cold = cold.get<uint8_t>(), a.kv_hot = T;
    const bool launched = gpu::launch_fattn_prefill(a, nullptr);
    CHECK(hipDeviceSynchronize());
    const auto have = download<_Float16>(b.o, (size_t)tokens * s.os());
    const bool ok = launched && std::memcmp(want.data(), have.data(), want.size() * sizeof(_Float16)) == 0;
    printf("split   hd %3d nh %2d at %6lld, prefill pos0 %6lld x %d tokens: bits equal: %s\n", s.cfg.hd, s.nh,
           (long long)T, (long long)pos0, tokens, ok ? "ok" : "FAILED");
    return ok;
}

bool split_all() {
    bool ok = true;
    for (const Shape& s : {Shape{kOrnith, 16, 2}, Shape{kBolt, 32, 2}}) {
        Rig rig(gpu::probe::fattn_kernels(s.cfg));
        for (const auto& [T, pos0, nb] :
             {std::tuple<int64_t, int64_t, int>{1000, 996, 4}, {1000, 500, 3}, {4093, 9000, 4}, {37, 2, 2}})
            ok = split_decode(rig, s, T, pos0, nb) && ok;
        for (const auto& [T, pos0, tokens] :
             {std::tuple<int64_t, int64_t, int>{1000, 900, 150}, {5000, 100, 70}, {37, 0, 90}})
            ok = split_prefill(s, T, pos0, tokens) && ok;
    }
    return ok;
}

}  // namespace
