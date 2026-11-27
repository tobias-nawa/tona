// NVFP4 (type 40) and IQ2_XXS (type 16) on the GPU against the CPU dequantization (model/quant.cpp), no other kernel
// involved: (1) the int8 matvec of the real k_mega with a test-only profile (gpu/probe/quant_matvec_kernels.hip) at 1,
// 4 and 8 tokens and with 8, 16 and 32 lanes per row, and its wide (WMMA) pass; (2) dequant_f16 bit for bit; (3) gemm_q
// against double. Random blocks always; with TONA_MODEL=<Bolt GGUF> also the first rows of the file's own tensors of
// both types. TONA_BOLT_SPEED=1 adds the time of a decode matvec against the VRAM bandwidth (prints GB/s).
// Bounds (stated before the first run): int8 path against the dequantized weights times the dequantized int8 x,
// relative to the sum of the absolute terms of the row: 1e-5 (float summation only); IQ2_XXS bit-equal to the CPU
// experts' matvec_q8 (8 lanes); wide pass 3e-3 (f16 weights and x); dequant_f16 equal in every bit; gemm_q 1e-4.
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "backend/env.h"
#include "cpu/experts.h"
#include "gpu/gemm.h"
#include "gpu/matvec_dev.h"
#include "gpu/probe/probe_kernels.h"
#include "mega_rig.h"
#include "model/gguf.h"
#include "model/quant.h"

namespace {

using namespace rig;
using tona::gpu::MegaOp;

int fails = 0;
#define EXPECT(c, ...)                                                                                                 \
    do {                                                                                                               \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s ", __LINE__, #c), std::fprintf(stderr, __VA_ARGS__), fails++; \
    } while (0)

constexpr int NBMAX = 8;

struct Weights {
    std::string label;
    uint32_t type;
    int64_t in, rows;
    std::vector<uint8_t> bytes;
    int64_t row_bytes() const { return tona::gpu::row_bytes_of(type, in); }
};

// Random blocks: valid for both formats (every grid index, sign index and code is legal); NVFP4 scale bytes in a
// moderate range, IQ2_XXS d as a small half
Weights random_weights(uint32_t type, int64_t in, int64_t rows, uint32_t seed) {
    std::mt19937 rng(seed);
    Weights w{type == 16 ? "random IQ2_XXS" : "random NVFP4", type, in, rows, {}};
    w.bytes.resize((size_t)(rows * w.row_bytes()));
    for (auto& b : w.bytes) b = (uint8_t)rng();
    const int blk = type == 16 ? 66 : 36;
    if (type != 16 && type != 40) return w;
    for (size_t o = 0; o + blk <= w.bytes.size(); o += blk) {
        if (type == 16) {
            const _Float16 d = (_Float16)(0.002f + 0.01f * (float)(rng() % 100) / 100.0f);
            std::memcpy(&w.bytes[o], &d, 2);
        } else {
            for (int s = 0; s < 4; s++) w.bytes[o + s] = (uint8_t)(0x20 + rng() % 0x30);
        }
    }
    return w;
}

// Rows of the file's own tensors: the first of the type, and for IQ2_XXS also the first shared-expert one
std::vector<Weights> real_weights(const tona::Gguf& g, int64_t max_rows) {
    std::vector<Weights> out;
    for (uint32_t type : {40u, 16u})
        for (const char* key : {type == 40 ? "ssm_in" : "_exps", type == 40 ? "ssm_out" : "shexp"}) {
            for (const auto& t : g.tensors()) {
                if (t.type != type || !t.name.contains(key)) continue;
                Weights w{t.name, type, (int64_t)t.dims[0], std::min<int64_t>(max_rows, (int64_t)t.dims[1]), {}};
                w.bytes.assign(t.data, t.data + w.rows * w.row_bytes());
                out.push_back(std::move(w));
                break;
            }
        }
    return out;
}

struct Activations {
    std::vector<float> x;  // [NBMAX][in]
    std::vector<int8_t> q;
    std::vector<tona::cpu::Q8Meta> m;
    std::vector<float> dq;  // the int8 values times their scales
    explicit Activations(int64_t in, uint32_t seed) : x(NBMAX * in), q(NBMAX * in), m(NBMAX * in / 32), dq(NBMAX * in) {
        std::mt19937 rng(seed);
        std::normal_distribution<float> n(0.0f, 1.0f);
        for (auto& v : x) v = n(rng);
        for (int b = 0; b < NBMAX; b++) tona::cpu::quantize_q8(&x[b * in], in, &q[b * in], &m[b * in / 32]);
        for (int64_t i = 0; i < NBMAX * in; i++) dq[i] = q[i] * m[i / 32].d;
    }
};

class Dev : public DeviceBuffer {
public:
    template <class T>
    explicit Dev(const std::vector<T>& v) : DeviceBuffer(std::max<size_t>(v.size() * sizeof(T), 16) + 64) {
        CHECK(hipMemcpy(get(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
};

template <class T>
std::vector<T> download(const DeviceBuffer& b, size_t n) {
    std::vector<T> v(n);
    CHECK(hipMemcpy(v.data(), b.get(), n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

std::vector<float> dequantized(const Weights& w) {
    std::vector<float> d((size_t)(w.rows * w.in));
    for (int64_t r = 0; r < w.rows; r++)
        tona::dequantize_row(w.type, &w.bytes[(size_t)(r * w.row_bytes())], &d[(size_t)(r * w.in)], w.in);
    return d;
}

// Rows of the oracle: sum of d[r][i] * v[i] in double and the sum of the absolute terms
struct Oracle {
    std::vector<double> y, mag;
    Oracle(const std::vector<float>& d, int64_t in, int64_t rows, const float* v) : y(rows), mag(rows) {
        for (int64_t r = 0; r < rows; r++)
            for (int64_t i = 0; i < in; i++) {
                const double t = (double)d[(size_t)(r * in + i)] * v[i];
                y[r] += t, mag[r] += std::fabs(t);
            }
    }
};

struct Run {
    int nb, lanes;
    bool wide;
};

// One OP_MATVEC step over the weights; y [nb][rows]
std::vector<float> run_matvec(Rig& rig, const Weights& w, const Activations& a, const Run& cfg) {
    const Dev dw(w.bytes), dx(a.x), dq(a.q), dm(a.m);
    const DeviceBuffer dy((size_t)cfg.nb * w.rows * 4);
    CHECK(hipMemset(dy.get(), 0xff, (size_t)cfg.nb * w.rows * 4));  // NaN where nothing is written
    const int per = 32 / cfg.lanes;
    tona::gpu::DevTask t{dw.get<uint8_t>(),
                         dx.get<float>(),
                         dy.get<float>(),
                         dq.get<int8_t>(),
                         dm.get<tona::gpu::Q8Meta>(),
                         w.rows,
                         (w.rows + per - 1) / per,
                         w.row_bytes(),
                         0,
                         w.type,
                         (int32_t)w.in,
                         cfg.lanes};
    t.xb = w.in, t.yb = w.rows, t.wide = cfg.wide ? 1 : 0;
    const Dev tasks(std::vector<tona::gpu::DevTask>{t});
    rig.set_tasks(tasks.get());
    MegaOp op;
    op.kind = tona::gpu::OP_MATVEC, op.a = 0, op.b = 1;
    tona::gpu::MegaParams p{};
    p.nb = cfg.nb, p.commit = cfg.nb, p.n_u = -1, p.wide_min = cfg.wide ? 1 : 0;
    EXPECT(rig.run({op}, p), "aborted\n");
    return download<float>(dy, (size_t)cfg.nb * w.rows);
}

// Worst |got - ref| / mag over the rows of the tokens
double worst_rel(const std::vector<float>& got, const Oracle* ref, int nb, int64_t rows) {
    double worst = 0;
    for (int b = 0; b < nb; b++)
        for (int64_t r = 0; r < rows; r++) {
            const double g = got[(size_t)(b * rows + r)];
            worst =
                std::max(worst, std::isfinite(g) ? std::fabs(g - ref[b].y[r]) / std::max(ref[b].mag[r], 1e-20) : 1e9);
        }
    return worst;
}

void check_int8(Rig& rig, const Weights& w, const Activations& a, const std::vector<float>& deq) {
    std::vector<Oracle> ref, ref_float;
    ref.reserve(NBMAX), ref_float.reserve(NBMAX);
    for (int b = 0; b < NBMAX; b++) ref.emplace_back(deq, w.in, w.rows, &a.dq[(size_t)(b * w.in)]);
    for (int b = 0; b < NBMAX; b++) ref_float.emplace_back(deq, w.in, w.rows, &a.x[(size_t)(b * w.in)]);
    double act = 0;
    for (int nb : {1, 4, 8})
        for (int lanes : {8, 16, 32}) {
            if ((w.in / 32) % lanes) continue;
            const auto y = run_matvec(rig, w, a, {nb, lanes, false});
            const double e = worst_rel(y, ref.data(), nb, w.rows);
            EXPECT(e <= 1e-5, "%s nb %d lanes %d: worst %.2e\n", w.label.c_str(), nb, lanes, e);
            act = std::max(act, worst_rel(y, ref_float.data(), nb, w.rows));
            if (nb == 8 && lanes == 8) std::printf("%-34s int8 path: worst %.2e (bound 1e-5)\n", w.label.c_str(), e);
        }
    std::printf("%-34s int8 activations against float x: worst %.2e of the sum of absolute terms\n", w.label.c_str(),
                act);
    const auto yw = run_matvec(rig, w, a, {8, 8, true});
    std::vector<Oracle> refw;
    refw.reserve(NBMAX);
    for (int b = 0; b < NBMAX; b++) refw.emplace_back(deq, w.in, w.rows, &a.x[(size_t)(b * w.in)]);
    const double ew = worst_rel(yw, refw.data(), 8, w.rows);
    EXPECT(ew <= 3e-3, "%s wide: worst %.2e\n", w.label.c_str(), ew);
    std::printf("%-34s wide pass: worst %.2e against float x (bound 3e-3)\n", w.label.c_str(), ew);
}

// The CPU experts' row function is the bit-identical twin of 8 lanes per row (cpu/experts.h)
void check_cpu_twin(Rig& rig, const Weights& w, const Activations& a) {
    if (!tona::cpu::expert_type_supported(w.type) || w.in % 256) return;
    const auto y = run_matvec(rig, w, a, {4, 8, false});
    std::vector<float> c((size_t)(4 * w.rows));
    tona::cpu::matvec_q8({w.type, w.bytes.data(), w.row_bytes(), w.in, 0, w.rows, a.q.data(), a.m.data(), w.in, 4, 0xf,
                          c.data(), w.rows});
    EXPECT(y == c, "%s: GPU and CPU rows differ\n", w.label.c_str());
    std::printf("%-34s GPU against CPU experts: %s\n", w.label.c_str(), y == c ? "bit-equal" : "DIFFERENT");
}

void check_dequant(const Weights& w, const std::vector<float>& deq) {
    const tona::gpu::DequantShape sh{w.type, w.rows, w.in, w.in};
    const Dev dw(w.bytes);
    const DeviceBuffer out((size_t)(w.rows * w.in) * 2);
    tona::gpu::dequant_f16(sh, dw.get(), out.get<_Float16>());
    CHECK(hipDeviceSynchronize());
    const auto h = download<_Float16>(out, (size_t)(w.rows * w.in));
    size_t bad = 0;
    for (size_t i = 0; i < h.size(); i++) {
        const _Float16 want = (_Float16)deq[i];
        bad += std::bit_cast<uint16_t>(h[i]) != std::bit_cast<uint16_t>(want);
    }
    EXPECT(bad == 0, "%s: dequant_f16 differs in %zu of %zu values\n", w.label.c_str(), bad, h.size());
    std::printf("%-34s dequant_f16: %zu differing values of %zu\n", w.label.c_str(), bad, h.size());
}

void check_gemm(const Weights& w, const std::vector<float>& deq) {
    constexpr int T = 100;
    std::mt19937 rng(5);
    std::normal_distribution<float> n(0.0f, 1.0f);
    std::vector<_Float16> x((size_t)(T * w.in));
    for (auto& v : x) v = (_Float16)n(rng);
    const Dev dx(x), dw(w.bytes);
    const DeviceBuffer dy((size_t)T * w.rows * 4);
    const tona::gpu::GemmQSeg seg{dx.get<_Float16>(), dw.get<uint8_t>(), dy.get<float>(), T,
                                  (int32_t)w.rows,    (int32_t)w.in,     (int32_t)w.in,   (int32_t)w.rows,
                                  w.row_bytes()};
    tona::gpu::gemm_q(w.type, &seg, 1);
    CHECK(hipDeviceSynchronize());
    const auto y = download<float>(dy, (size_t)T * w.rows);
    double worst = 0;
    for (int t = 0; t < T; t += 7)
        for (int64_t r = 0; r < w.rows; r += 3) {
            double ref = 0, mag = 0;
            for (int64_t i = 0; i < w.in; i++) {
                const double v =
                    (double)(float)(_Float16)deq[(size_t)(r * w.in + i)] * (float)x[(size_t)(t * w.in + i)];
                ref += v, mag += std::fabs(v);
            }
            worst = std::max(worst, std::fabs(y[(size_t)(t * w.rows + r)] - ref) / std::max(mag, 1e-20));
        }
    EXPECT(worst <= 1e-4, "%s: gemm_q worst %.2e\n", w.label.c_str(), worst);
    std::printf("%-34s gemm_q: worst %.2e (bound 1e-4)\n", w.label.c_str(), worst);
}

// Decode matvec time on a matrix larger than the cache, against the VRAM bandwidth; 8 lanes per row (the expert
// groups) and the lanes of a dense task (lanes_per_row of the row length)
void bench(Rig& rig, uint32_t type, int64_t in, int64_t rows) {
    Weights w = random_weights(type, in, rows, 9);
    const Activations a(in, 3);
    const Dev dw(w.bytes), dx(a.x), dq(a.q), dm(a.m);
    const DeviceBuffer dy((size_t)NBMAX * rows * 4);
    const int64_t subs = in / 32;
    // TONA_BOLT_SPEED_ONE=1: the decode case of the expert groups only (8 lanes, 1 token), for a profiler run
    static const bool one = tona::env::flag("TONA_BOLT_SPEED_ONE");
    for (const int lanes : {8, (int)std::min<int64_t>(32, subs & -subs)}) {
        if (one && lanes != 8) continue;
        tona::gpu::DevTask t{dw.get<uint8_t>(),
                             dx.get<float>(),
                             dy.get<float>(),
                             dq.get<int8_t>(),
                             dm.get<tona::gpu::Q8Meta>(),
                             rows,
                             (rows * lanes + 31) / 32,
                             w.row_bytes(),
                             0,
                             type,
                             (int32_t)in,
                             lanes};
        t.xb = in, t.yb = rows;
        const Dev tasks(std::vector<tona::gpu::DevTask>{t});
        rig.set_tasks(tasks.get());
        MegaOp op;
        op.kind = tona::gpu::OP_MATVEC, op.a = 0, op.b = 1;
        for (int nb : {1, 4, 8}) {
            if (one && nb != 1) continue;
            tona::gpu::MegaParams p{};
            p.nb = nb, p.commit = nb, p.n_u = -1;
            const double us = rig.time_us({op}, p, 50);
            std::printf("speed type %u %lld x %lld (%.1f MB), %d lanes, %d tokens: %.1f us, %.1f GB/s of weights\n",
                        type, (long long)rows, (long long)in, rows * w.row_bytes() / 1e6, lanes, nb, us,
                        rows * w.row_bytes() / us / 1e3);
        }
    }
}

// Matrices of 70 MB and more, above the 32 MB of cache; 22, 18 and 8 are the yardsticks of the existing loops
// TONA_BOLT_SPEED_TYPE=T: only that format
void bench_all(Rig& rig) {
    const int64_t only = tona::env::integer("TONA_BOLT_SPEED_TYPE", 0);
    if (!only || only == 40) bench(rig, 40, 4096, 32768);
    if (!only || only == 16) bench(rig, 16, 1024, 262144);
    if (!only || only == 22) bench(rig, 22, 1024, 262144);
    if (!only || only == 18) bench(rig, 18, 1024, 262144);
    if (!only || only == 8) bench(rig, 8, 4096, 16384);
}

void check_all(Rig& rig, const Weights& w) {
    const Activations a(w.in, 11);
    const auto deq = dequantized(w);
    check_int8(rig, w, a, deq);
    check_cpu_twin(rig, w, a);
    check_dequant(w, deq);
    check_gemm(w, deq);
}

}  // namespace

int main() {
    Rig rig(tona::gpu::probe::quant_matvec_kernels());
    std::vector<Weights> all;
    all.push_back(random_weights(40, 4096, 517, 1));
    all.push_back(random_weights(16, 1024, 517, 2));
    all.push_back(random_weights(16, 2560, 301, 3));
    std::unique_ptr<tona::Gguf> g;
    if (const char* path = tona::env::text("TONA_MODEL")) {
        g = std::make_unique<tona::Gguf>(path);
        for (auto& w : real_weights(*g, 2048)) all.push_back(std::move(w));
    }
    if (!tona::env::flag("TONA_BOLT_SPEED_ONE"))
        for (const Weights& w : all) check_all(rig, w);
    if (tona::env::flag("TONA_BOLT_SPEED")) bench_all(rig);
    std::printf(fails ? "FAILED: %d\n" : "all passed\n", fails);
    return fails != 0;
}
