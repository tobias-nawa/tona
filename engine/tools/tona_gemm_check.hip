// Checks the prefill building blocks (gpu/gemm): unpacking to f16 against the CPU reference (model/quant) for every
// format of the model, the GEMM with odd shapes against a sum in double, the GEMM that unpacks in LDS against
// unpack + f16 GEMM, and the fp8 GEMM against the f16 GEMM. Also measures the unpack throughput.
// Usage: tona-gemm-check MODEL.gguf
// TONA_CHECK_T: tokens per expert in the unpack-in-LDS check (default 150).
#include <hip/hip_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <map>
#include <random>
#include <vector>

#include "backend/env.h"
#include "gpu/gemm.h"
#include "model/quant.h"
#include "model/qwen4exp.h"
#include "tona_hip_check.h"

using namespace tona;
using tona::tools::DeviceBuffer;

namespace {

const char* verdict(bool ok) { return ok ? "ok    " : "FAILED"; }

// Seconds per call, after a synchronize
template <class F>
double seconds_per_call(F&& f, int reps) {
    CHECK(hipDeviceSynchronize());
    const auto t0 = std::chrono::steady_clock::now();
    for (int rep = 0; rep < reps; rep++) f();
    CHECK(hipDeviceSynchronize());
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
}

// unpack to f16 against the CPU reference

// One tensor per format (the first), as long as both sides can unpack it
std::map<uint32_t, const TensorInfo*> first_tensor_per_format(const Qwen4ExpModel& m) {
    std::map<uint32_t, const TensorInfo*> by_type;
    for (const auto& t : m.gguf().tensors())
        if (t.dims[0] % 32 == 0 && t.dims.size() >= 2 && dequant_supported(t.type) &&
            gpu::dequant_f16_supported(t.type) && !by_type.contains(t.type))
            by_type[t.type] = &t;
    return by_type;
}

// Up to 256 rows of the tensor; returns whether the error stays below f16 rounding
bool check_dequant(uint32_t type, const TensorInfo& t) {
    const int64_t in = (int64_t)t.dims[0], rows = std::min<int64_t>(256, (int64_t)(t.elements() / in));
    const GgmlTypeInfo* ti = ggml_type_info(type);
    const int64_t rb = in / ti->block_elems * ti->block_bytes;
    DeviceBuffer dw(rows * rb + 64), dout(rows * in * 2);
    CHECK(hipMemcpy(dw.get(), t.data, rows * rb, hipMemcpyHostToDevice));
    gpu::dequant_f16({type, rows, in, in}, dw.get(), dout.get<_Float16>());
    std::vector<_Float16> got(rows * in);
    CHECK(hipMemcpy(got.data(), dout.get(), got.size() * 2, hipMemcpyDeviceToHost));
    std::vector<float> ref(rows * in);
    dequantize_row(type, t.data, ref.data(), rows * in);
    double maxerr = 0, maxv = 0;
    for (size_t i = 0; i < ref.size(); i++) {
        maxv = std::max(maxv, (double)std::fabs(ref[i]));
        // f16 rounds to a relative 2^-11; anything beyond is an error
        const double e = std::fabs((double)got[i] - ref[i]) / (std::fabs(ref[i]) + 1e-6);
        if (std::fabs(ref[i]) > 1e-4) maxerr = std::max(maxerr, e);
    }
    const bool ok = maxerr < 1e-3;
    printf("%s type %2u %-28s %ld x %ld: rel. error %.1e (max |w| %.3g)\n", verdict(ok), type, t.name.c_str(),
           (long)rows, (long)in, maxerr, maxv);
    return ok;
}

// dequant_fp8 gives the same bytes and factors as dequant_f16 followed by to_fp8_rows (the one-pass path of the FP8
// prefill)
bool check_dequant_fp8(uint32_t type, const TensorInfo& t) {
    const int64_t in = (int64_t)t.dims[0], rows = std::min<int64_t>(200, (int64_t)(t.elements() / in));
    if (in % 64 || in > 8192) return true;
    const GgmlTypeInfo* ti = ggml_type_info(type);
    const int64_t rb = in / ti->block_elems * ti->block_bytes;
    DeviceBuffer dw(rows * rb + 64), d16(rows * in * 2), a8(rows * in), b8(rows * in), sa(rows * 4), sb(rows * 4);
    CHECK(hipMemcpy(dw.get(), t.data, rows * rb, hipMemcpyHostToDevice));
    gpu::dequant_f16({type, rows, in, in}, dw.get(), d16.get<_Float16>());
    gpu::to_fp8_rows({d16.get<_Float16>(), in, rows, in, a8.get<uint8_t>(), in, sa.get<float>()});
    gpu::dequant_fp8({type, rows, in, in}, dw.get(), b8.get<uint8_t>(), sb.get<float>());
    std::vector<uint8_t> x(rows * in), y(rows * in);
    std::vector<float> fx(rows), fy(rows);
    CHECK(hipMemcpy(x.data(), a8.get(), x.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(y.data(), b8.get(), y.size(), hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(fx.data(), sa.get(), rows * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(fy.data(), sb.get(), rows * 4, hipMemcpyDeviceToHost));
    const bool ok = x == y && fx == fy;
    printf("%s type %2u dequant_fp8 equals dequant_f16 + to_fp8_rows (%ld x %ld)\n", verdict(ok), type, (long)rows,
           (long)in);
    return ok;
}

// Unpack throughput on the experts of layer 0 (gate, down) and a dense matrix
void measure_dequant_throughput(const Qwen4ExpModel& m) {
    const Layer& L = m.layers()[0];
    for (const TensorInfo* t : {L.gate_exps, L.down_exps, L.qkv}) {
        const int64_t in = (int64_t)t->dims[0], rows = (int64_t)(t->elements() / in);
        DeviceBuffer dw(t->bytes + 64), dout(rows * in * 2);
        CHECK(hipMemcpy(dw.get(), t->data, t->bytes, hipMemcpyHostToDevice));
        auto unpack = [&] { gpu::dequant_f16({t->type, rows, in, in}, dw.get(), dout.get<_Float16>()); };
        unpack();
        const double s = seconds_per_call(unpack, 5);
        printf("unpack %-26s type %2u: %.2f ms, %.0f GB/s read, %.0f GB/s written\n", t->name.c_str(), t->type, s * 1e3,
               t->bytes / s / 1e9, rows * in * 2 / s / 1e9);
    }
}

// GEMM that unpacks in LDS against unpack + f16 GEMM

// 8 experts of one format with `tokens` tokens each, as two segment lists computing the same thing
struct ExpertGemm {
    static constexpr int kExperts = 8;
    ExpertGemm(const TensorInfo& t, int tokens, std::mt19937& rng)
        : type(t.type),
          in((int)t.dims[0]),
          rows((int)t.dims[1]),
          tokens(tokens),
          rb((int64_t)(t.bytes / t.dims[2] / rows)),
          dx((size_t)kExperts * tokens * in * 2),
          dw((size_t)kExperts * rows * rb + 64),
          w16((size_t)kExperts * rows * in * 2),
          y1((size_t)kExperts * tokens * rows * 4),
          y2((size_t)kExperts * tokens * rows * 4) {
        std::uniform_real_distribution<float> U(-1, 1);
        std::vector<_Float16> hx((size_t)kExperts * tokens * in);
        for (auto& v : hx) v = (_Float16)U(rng);
        CHECK(hipMemcpy(dx.get(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dw.get(), t.data, (size_t)kExperts * rows * rb, hipMemcpyHostToDevice));
        for (int e = 0; e < kExperts; e++) {
            unpack(e);
            qs.push_back({dx.get<_Float16>() + (size_t)e * tokens * in, dw.get<uint8_t>() + (size_t)e * rows * rb,
                          y1.get<float>() + (size_t)e * tokens * rows, tokens, rows, in, in, rows, rb});
            fs.push_back({dx.get<_Float16>() + (size_t)e * tokens * in, w16.get<_Float16>() + (size_t)e * rows * in,
                          y2.get<float>() + (size_t)e * tokens * rows, tokens, rows, in, in, in, rows});
        }
    }

    void unpack(int e) {
        gpu::dequant_f16({type, rows, in, in}, dw.get<uint8_t>() + (size_t)e * rows * rb,
                         w16.get<_Float16>() + (size_t)e * rows * in);
    }

    uint32_t type;
    int in, rows, tokens;
    int64_t rb;
    DeviceBuffer dx, dw, w16, y1, y2;
    std::vector<gpu::GemmQSeg> qs;
    std::vector<gpu::GemmSeg> fs;
};

bool check_gemm_q(uint32_t type, const TensorInfo& t, int tokens, std::mt19937& rng) {
    ExpertGemm g(t, tokens, rng);
    const int ne = ExpertGemm::kExperts;
    gpu::gemm_q(type, g.qs.data(), ne);
    gpu::gemm_f16(g.fs.data(), ne);
    std::vector<float> a1((size_t)ne * tokens * g.rows), a2(a1.size());
    CHECK(hipMemcpy(a1.data(), g.y1.get(), a1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(a2.data(), g.y2.get(), a2.size() * 4, hipMemcpyDeviceToHost));
    double err = 0, mx = 0;
    for (size_t i = 0; i < a1.size(); i++)
        err = std::max(err, (double)std::fabs(a1[i] - a2[i])), mx = std::max(mx, (double)std::fabs(a2[i]));
    const bool ok = err <= 1e-4 * mx;
    const double sq = seconds_per_call([&] { gpu::gemm_q(type, g.qs.data(), ne); }, 5);
    const double sf = seconds_per_call(
        [&] {
            for (int e = 0; e < ne; e++) g.unpack(e);
            gpu::gemm_f16(g.fs.data(), ne);
        },
        5);
    printf(
        "%s gemm_q type %2u %dx%d, 8 experts of %d tokens: deviation %.1e of max %.2g; %.2f ms (unpack + f16: %.2f "
        "ms)\n",
        verdict(ok), type, g.rows, g.in, tokens, err, mx, sq * 1e3, sf * 1e3);
    return ok;
}

// fp8 GEMM against f16 GEMM

struct Fp8Shape {
    const char* name;
    int n, k;
};

// Prefill shapes at 2,048 tokens; deviation relative to the mean magnitude of Y
bool check_fp8(const Fp8Shape& sh, std::mt19937& rng) {
    std::normal_distribution<float> G(0, 1);
    const int T = 2048, N = sh.n, K = sh.k;
    std::vector<_Float16> hx((size_t)T * K), hw((size_t)N * K);
    for (auto& v : hx) v = (_Float16)G(rng);
    for (auto& v : hw) v = (_Float16)(G(rng) * 0.02f);
    for (int t = 0; t < T; t += 97) hx[(size_t)t * K + 5] = (_Float16)40.0f;  // outliers as in activations
    DeviceBuffer dx(hx.size() * 2), dw(hw.size() * 2), x8(hx.size()), w8(hw.size()), sx((size_t)T * 4),
        sw((size_t)N * 4), y1((size_t)T * N * 4), y2((size_t)T * N * 4);
    CHECK(hipMemcpy(dx.get(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw.get(), hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    gpu::to_fp8_rows({dx.get<_Float16>(), K, T, K, x8.get<uint8_t>(), K, sx.get<float>()});
    gpu::to_fp8_rows({dw.get<_Float16>(), K, N, K, w8.get<uint8_t>(), K, sw.get<float>()});
    const gpu::Gemm8Seg s8{
        x8.get<uint8_t>(), sx.get<float>(), w8.get<uint8_t>(), sw.get<float>(), y1.get<float>(), T, N, K, K, K, N};
    gpu::gemm_fp8(&s8, 1);
    gpu::gemm_f16(dx.get<_Float16>(), dw.get<_Float16>(), y2.get<float>(), {T, N, K});
    std::vector<float> a1((size_t)T * N), a2(a1.size());
    CHECK(hipMemcpy(a1.data(), y1.get(), a1.size() * 4, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(a2.data(), y2.get(), a2.size() * 4, hipMemcpyDeviceToHost));
    double se = 0, sa = 0;
    for (size_t i = 0; i < a1.size(); i++) se += std::fabs(a1[i] - a2[i]), sa += std::fabs(a2[i]);
    const double rel = se / sa;
    const bool ok = rel < 0.08;
    const double t8 = seconds_per_call([&] { gpu::gemm_fp8(&s8, 1); }, 10);
    const double tc = seconds_per_call(
        [&] { gpu::to_fp8_rows({dx.get<_Float16>(), K, T, K, x8.get<uint8_t>(), K, sx.get<float>()}); }, 10);
    const double t16 = seconds_per_call(
        [&] { gpu::gemm_f16(dx.get<_Float16>(), dw.get<_Float16>(), y2.get<float>(), {T, N, K}); }, 10);
    const double fl = 2.0 * T * N * K;
    printf(
        "%s fp8 %s, T %d: mean dev. %.2f %%; fp8 %.3f ms = %.1f TFLOPS (+ converting X %.3f ms), f16 %.3f ms = %.1f "
        "TFLOPS\n",
        verdict(ok), sh.name, T, rel * 100, t8 * 1e3, fl / t8 / 1e12, tc * 1e3, t16 * 1e3, fl / t16 / 1e12);
    return ok;
}

// f16 GEMM with odd shapes, two parts in one launch

struct OddShape {
    int t, n, k;
};

double worst_relative_error(const OddShape& f, const std::vector<_Float16>& hx, const std::vector<_Float16>& hw,
                            const std::vector<float>& hy) {
    double maxerr = 0;
    for (int t = 0; t < f.t; t++)
        for (int n = 0; n < f.n; n++) {
            double ref = 0;
            for (int k = 0; k < f.k; k++)
                ref += (double)(float)hx[(size_t)t * f.k + k] * (float)hw[(size_t)n * f.k + k];
            for (int part = 0; part < 2; part++)
                maxerr = std::max(
                    maxerr, std::fabs(ref - hy[(size_t)part * f.t * f.n + (size_t)t * f.n + n]) / (std::fabs(ref) + 1));
        }
    return maxerr;
}

bool check_odd_shape(const OddShape& f, std::mt19937& rng) {
    std::uniform_real_distribution<float> U(-1, 1);
    std::vector<_Float16> hx((size_t)f.t * f.k), hw((size_t)f.n * f.k);
    for (auto& v : hx) v = (_Float16)U(rng);
    for (auto& v : hw) v = (_Float16)U(rng);
    DeviceBuffer dx(hx.size() * 2), dw(hw.size() * 2), dy((size_t)2 * f.t * f.n * 4);
    CHECK(hipMemcpy(dx.get(), hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dw.get(), hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
    // Part 2: the same computation into the second half of y
    const _Float16 *x = dx.get<_Float16>(), *w = dw.get<_Float16>();
    float* y = dy.get<float>();
    const gpu::GemmSeg segs[2] = {{x, w, y, f.t, f.n, f.k, f.k, f.k, f.n},
                                  {x, w, y + (size_t)f.t * f.n, f.t, f.n, f.k, f.k, f.k, f.n}};
    gpu::gemm_f16(segs, 2);
    std::vector<float> hy((size_t)2 * f.t * f.n);
    CHECK(hipMemcpy(hy.data(), dy.get(), hy.size() * 4, hipMemcpyDeviceToHost));
    const double maxerr = worst_relative_error(f, hx, hw, hy);
    // WMMA sums in f32 but not in the CPU's order: the error grows with K and stays below the rounding of the f16
    // inputs (2^-11 = 4.9e-4 per value)
    const bool ok = maxerr < 1e-3;
    printf("%s GEMM T=%d N=%d K=%d (2 parts): rel. error %.1e\n", verdict(ok), f.t, f.n, f.k, maxerr);
    return ok;
}

int run_checks(const char* model_path) {
    int fails = 0;
    Qwen4ExpModel m(model_path);
    for (const auto& [type, t] : first_tensor_per_format(m))
        fails += !check_dequant(type, *t) + !check_dequant_fp8(type, *t);
    measure_dequant_throughput(m);

    // per expert format 8 experts of TONA_CHECK_T tokens each
    const int tokens = (int)env::integer("TONA_CHECK_T", 150);
    std::mt19937 rq(2);
    std::map<uint32_t, const TensorInfo*> exps;
    for (const Layer& L : m.layers())
        for (const TensorInfo* t : {L.gate_exps, L.up_exps, L.down_exps})
            if (!exps.contains(t->type)) exps[t->type] = t;
    for (const auto& [type, t] : exps) fails += !check_gemm_q(type, *t, tokens, rq);

    std::mt19937 r8(3);
    for (const Fp8Shape& sh : {Fp8Shape{"qkv 2560->10240", 10240, 2560}, Fp8Shape{"q 2560->12288", 12288, 2560},
                               Fp8Shape{"ssm_out 6144->2560", 2560, 6144}, Fp8Shape{"odd 1000->333", 333, 1024}})
        fails += !check_fp8(sh, r8);

    std::mt19937 rng(1);
    for (const OddShape& f : {OddShape{200, 300, 320}, OddShape{1, 2560, 640}, OddShape{1000, 4, 10240},
                              OddShape{129, 129, 64}, OddShape{300, 4200, 192}, OddShape{70, 5000, 96}})
        fails += !check_odd_shape(f, rng);
    return fails;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-gemm-check MODEL.gguf\n");
        return 2;
    }
    try {
        const int fails = run_checks(argv[1]);
        printf(fails ? "%d failures\n" : "all ok\n", fails);
        return fails ? 1 : 0;
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
