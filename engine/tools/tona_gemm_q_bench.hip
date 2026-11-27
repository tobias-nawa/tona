// Times and hashes the expert GEMM that unpacks in the LDS (gpu/gemm_q, k_gemm_q2) per expert format of the model:
// TONA_CHECK_E experts (default 32, the segment limit of one launch) of TONA_CHECK_T tokens each (default 48; the real
// prefill sees about 27 per expert and pass, research/measurements/2026-11-02-prefill-int8.md). The hash of the
// output is equal between two builds exactly when the results are bit-identical.
// Usage: tona-gemm-q-bench MODEL.gguf
#include <hip/hip_runtime.h>

#include <bit>
#include <chrono>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "backend/env.h"
#include "gpu/gemm.h"
#include "model/qwen4exp.h"
#include "tona_hip_check.h"

using namespace tona;
using tona::tools::DeviceBuffer;

namespace {

int experts() {
    static const int e = (int)env::integer("TONA_CHECK_E", 32);
    return e;
}

template <class F>
double seconds_per_call(F&& f, int reps) {
    f();
    CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; rep++) f();
    CHECK(hipDeviceSynchronize());
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

void run_case(const TensorInfo& t, int tokens, std::mt19937& rng) {
    const int in = (int)t.dims[0], rows = (int)t.dims[1];
    const int64_t rb = (int64_t)(t.bytes / t.dims[2] / rows);
    std::normal_distribution<float> G(0, 1);
    std::vector<_Float16> hx((size_t)experts() * tokens * in);
    for (auto& v : hx) v = (_Float16)G(rng);
    const size_t ny = (size_t)experts() * tokens * rows;
    DeviceBuffer dx(hx.size() * 2), dw((size_t)experts() * rows * rb + 64), dy(ny * 4);
    CHECK(hipMemcpy(dx.get(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw.get(), t.data, (size_t)experts() * rows * rb, hipMemcpyHostToDevice));
    std::vector<gpu::GemmQSeg> segs;
    segs.reserve(experts());
    for (int e = 0; e < experts(); e++)
        segs.push_back({dx.get<_Float16>() + (size_t)e * tokens * in, dw.get<uint8_t>() + (size_t)e * rows * rb,
                        dy.get<float>() + (size_t)e * tokens * rows, tokens, rows, in, in, rows, rb});
    gpu::gemm_q(t.type, segs.data(), experts());
    std::vector<float> y(ny);
    CHECK(hipMemcpy(y.data(), dy.get(), ny * 4, hipMemcpyDeviceToHost));
    uint64_t fnv = 1469598103934665603ull;
    for (float v : y) fnv = (fnv ^ std::bit_cast<uint32_t>(v)) * 1099511628211ull;
    const double s = seconds_per_call([&] { gpu::gemm_q(t.type, segs.data(), experts()); }, 10);
    printf("[%016lx] type %2u %dx%d, %d x %d tokens: gemm_q %.3f ms\n", (unsigned long)fnv, t.type, rows, in, experts(),
           tokens, s * 1e3);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-gemm-q-bench MODEL.gguf\n");
        return 2;
    }
    try {
        Qwen4ExpModel m(argv[1]);
        const int tokens = (int)env::integer("TONA_CHECK_T", 48);
        std::map<uint32_t, const TensorInfo*> exps;
        for (const Layer& L : m.layers())
            for (const TensorInfo* t : {L.gate_exps, L.up_exps, L.down_exps})
                if (!exps.contains(t->type)) exps[t->type] = t;
        std::mt19937 rng(5);
        for (const auto& [type, t] : exps) run_case(*t, tokens, rng);
        return 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
