// Checks and measures gpu::matvec on real weights: per format all tensors with the most common row length merged
// into one large matrix (up to --cap MB, so the 32 MiB Infinity Cache does not distort the timing), result against
// the CPU reference (dequantization bit-exact, dot product in double).
// Usage: tona-gpu-matvec MODEL.gguf [--types 22,18,...] [--cap 512] [--runs 10]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "gpu/matvec.h"
#include "model/gguf.h"
#include "model/quant.h"
#include "tona_args.h"
#include "tona_hip_check.h"

using tona::tools::DeviceBuffer;

namespace {

constexpr double kErrorLimit = 1e-4;
constexpr double kPeakGBs = 315.0;  // measured VRAM read bandwidth (research/measurements/2026-10-22-micro.md)

struct Options {
    std::vector<uint32_t> only;
    int64_t cap = 512ll << 20;
    int runs = 10;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 2; i + 1 < argc; i++) {
        if (!std::strcmp(argv[i], "--types")) {
            std::stringstream ss(argv[i + 1]);
            for (std::string t; std::getline(ss, t, ',');)
                o.only.push_back((uint32_t)tona::tools::parse_int(t.c_str()));
        } else if (!std::strcmp(argv[i], "--cap")) {
            o.cap = tona::tools::parse_int64(argv[i + 1]) << 20;
        } else if (!std::strcmp(argv[i], "--runs")) {
            o.runs = tona::tools::parse_int(argv[i + 1]);
        }
    }
    return o;
}

// Rows of one type with one row length, collected up to the cap
struct Matrix {
    uint32_t type;
    const tona::GgmlTypeInfo* info;
    int64_t in;
    int64_t row_bytes;
    int64_t rows;
    std::vector<uint8_t> host;
};

using TensorsByLength = std::map<uint64_t, std::vector<const tona::TensorInfo*>>;

uint64_t total_bytes(const std::vector<const tona::TensorInfo*>& v) {
    uint64_t s = 0;
    for (auto* t : v) s += t->bytes;
    return s;
}

// The row length with the most bytes
Matrix collect_matrix(uint32_t type, const TensorsByLength& lens, int64_t cap) {
    auto best = std::max_element(lens.begin(), lens.end(), [](const auto& a, const auto& b) {
        return total_bytes(a.second) < total_bytes(b.second);
    });
    Matrix m;
    m.type = type;
    m.info = tona::ggml_type_info(type);
    m.in = (int64_t)best->first;
    m.row_bytes = m.in / m.info->block_elems * m.info->block_bytes;
    for (auto* t : best->second) {
        const int64_t take = std::min<int64_t>((int64_t)t->bytes, cap - (int64_t)m.host.size());
        if (take <= 0) break;
        const int64_t n = take / m.row_bytes * m.row_bytes;
        m.host.insert(m.host.end(), t->data, t->data + n);
    }
    m.rows = (int64_t)m.host.size() / m.row_bytes;
    return m;
}

// Matrix, x and result on the device; the int8 path needs x rounded to int8
struct Device {
    explicit Device(const Matrix& m)
        : w(m.host.size() + tona::gpu::kMatvecReadPad),
          x(m.in * 4),
          y(m.rows * 4),
          q(m.in),
          meta(m.in / 32 * sizeof(tona::gpu::Q8Meta)) {}
    DeviceBuffer w, x, y, q, meta;
    tona::gpu::Q8Vec vec() const { return {q.get<int8_t>(), meta.get<tona::gpu::Q8Meta>()}; }
};

// x after rounding to int8, as the int8 path sees it
std::vector<float> rounded_x(const Matrix& m, const Device& dev) {
    tona::gpu::quantize_q8(dev.x.get<float>(), m.in, dev.vec());
    std::vector<int8_t> hq(m.in);
    std::vector<tona::gpu::Q8Meta> hm(m.in / 32);
    CHECK(hipMemcpy(hq.data(), dev.q.get(), m.in, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(hm.data(), dev.meta.get(), hm.size() * sizeof(tona::gpu::Q8Meta), hipMemcpyDeviceToHost));
    std::vector<float> x8(m.in);
    for (int64_t i = 0; i < m.in; i++) x8[i] = hq[i] * hm[i / 32].d;
    return x8;
}

template <class Run>
std::vector<float> result_of(Run run, const Matrix& m, const Device& dev) {
    run();
    CHECK(hipDeviceSynchronize());
    std::vector<float> y(m.rows);
    CHECK(hipMemcpy(y.data(), dev.y.get(), m.rows * 4, hipMemcpyDeviceToHost));
    return y;
}

struct Errors {
    double f = 0;   // float path against the exact result
    double q = 0;   // int8 path against the same rounded x (checks the kernel)
    double qx = 0;  // int8 path against the real x (shows the rounding of x)
};

double relative_error(double got, double want, double scale) {
    return scale > 0 ? std::fabs(got - want) / scale : std::fabs(got - want);
}

// Reference in double; error relative to sum |w*x| of the row (the measure for rounding when summing in float)
Errors reference_errors(const Matrix& m, const std::vector<float>& x, const std::vector<float>& x8,
                        const std::vector<float>& yf, const std::vector<float>& yq) {
    Errors err;
    const bool q8 = !yq.empty();
#pragma omp parallel
    {
        std::vector<float> wr(m.in);
        Errors local;
#pragma omp for schedule(static)
        for (int64_t r = 0; r < m.rows; r++) {
            tona::dequantize_row(m.type, m.host.data() + r * m.row_bytes, wr.data(), m.in);
            double s = 0, a = 0, s8 = 0, a8 = 0;
            for (int64_t i = 0; i < m.in; i++) {
                s += (double)wr[i] * x[i], a += std::fabs((double)wr[i] * x[i]);
                s8 += (double)wr[i] * x8[i], a8 += std::fabs((double)wr[i] * x8[i]);
            }
            local.f = std::max(local.f, relative_error(yf[r], s, a));
            if (q8)
                local.q = std::max(local.q, relative_error(yq[r], s8, a8)),
                local.qx = std::max(local.qx, relative_error(yq[r], s, a));
        }
#pragma omp critical
        err.f = std::max(err.f, local.f), err.q = std::max(err.q, local.q), err.qx = std::max(err.qx, local.qx);
    }
    return err;
}

// Sorted times in us per call: `runs` measurements of 10 calls each
template <class Run>
std::vector<double> time_calls(Run run, int runs) {
    hipEvent_t a, b;
    CHECK(hipEventCreate(&a));
    CHECK(hipEventCreate(&b));
    for (int i = 0; i < 5; i++) run();
    std::vector<double> us;
    for (int k = 0; k < runs; k++) {
        CHECK(hipEventRecord(a));
        for (int i = 0; i < 10; i++) run();
        CHECK(hipEventRecord(b));
        CHECK(hipEventSynchronize(b));
        float ms;
        CHECK(hipEventElapsedTime(&ms, a, b));
        us.push_back(ms * 100.0);  // ms per 10 calls -> us per call
    }
    CHECK(hipEventDestroy(a));
    CHECK(hipEventDestroy(b));
    std::sort(us.begin(), us.end());
    return us;
}

std::string timing_columns(const std::vector<double>& us, size_t bytes) {
    const double med = us[us.size() / 2], gbs = bytes / (med * 1e3);
    char buf[160];
    snprintf(buf, sizeof buf, "%.0f (%.0f-%.0f) | %.0f | %.0f %%", med, us.front(), us.back(), gbs,
             gbs / kPeakGBs * 100);
    return buf;
}

void print_row(const Matrix& m, const Errors& err, bool q8, const std::string& float_timing,
               const std::string& int8_timing) {
    std::string tq = "- | - | - | - | -";
    if (q8) {
        char buf[64];
        snprintf(buf, sizeof buf, "%.1e%s | %.1e | ", err.q, err.q > kErrorLimit ? " **WRONG**" : "", err.qx);
        tq = buf + int8_timing;
    }
    printf("| %s | %lld | %lld | %.0f | %.1e%s | %s | %s |\n", m.info->name, (long long)m.in, (long long)m.rows,
           m.host.size() / 1e6, err.f, err.f > kErrorLimit ? " **WRONG**" : "", float_timing.c_str(), tq.c_str());
    fflush(stdout);
}

// Check and measure one format; skipped if no rows fit
void measure_type(uint32_t type, const TensorsByLength& lens, const Options& o, std::mt19937& rng) {
    const Matrix m = collect_matrix(type, lens, o.cap);
    if (m.rows == 0) return;
    std::vector<float> x(m.in);
    std::normal_distribution<float> nd(0, 1);
    for (auto& v : x) v = nd(rng);
    Device dev(m);
    CHECK(hipMemcpy(dev.w.get(), m.host.data(), m.host.size(), hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.x.get(), x.data(), m.in * 4, hipMemcpyHostToDevice));
    const bool q8 = tona::gpu::matvec_q8_supported(type);
    const std::vector<float> x8 = q8 ? rounded_x(m, dev) : std::vector<float>(m.in);
    auto run_f = [&] {
        tona::gpu::matvec(
            tona::gpu::MatvecTask{type, m.in, m.rows, dev.w.get(), dev.x.get<float>(), dev.y.get<float>()});
    };
    auto run_q = [&] {
        tona::gpu::matvec_q8(
            tona::gpu::MatvecTask{type, m.in, m.rows, dev.w.get(), nullptr, dev.y.get<float>(), dev.vec()});
    };
    const auto yf = result_of(run_f, m, dev);
    const auto yq = q8 ? result_of(run_q, m, dev) : std::vector<float>();
    const Errors err = reference_errors(m, x, x8, yf, yq);
    const std::string tf = timing_columns(time_calls(run_f, o.runs), m.host.size());
    const std::string tq = q8 ? timing_columns(time_calls(run_q, o.runs), m.host.size()) : "";
    print_row(m, err, q8, tf, tq);
}

}  // namespace

namespace {

int run(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-gpu-matvec MODEL.gguf [--types 22,18] [--cap MB] [--runs N]\n");
        return 2;
    }
    const Options o = parse_options(argc, argv);
    tona::Gguf g(argv[1]);

    // Per type by row length; the PLE table is left out (no matvec)
    std::map<uint32_t, TensorsByLength> by;
    for (const auto& t : g.tensors())
        if (t.name != "per_layer_token_embd.weight" && t.dims.size() >= 2 && t.dims[0] >= 256 && t.dims[0] % 64 == 0)
            by[t.type][t.dims[0]].push_back(&t);

    printf(
        "| Type | Row | Rows | MB | float: error | us (min-max) | GB/s | %% of 315 "
        "| int8: error | rounding of x | us (min-max) | GB/s | %% of 315 |\n");
    printf("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    std::mt19937 rng(1);
    for (auto& [type, lens] : by) {
        if (!o.only.empty() && std::find(o.only.begin(), o.only.end(), type) == o.only.end()) continue;
        if (!tona::gpu::matvec_supported(type)) continue;
        measure_type(type, lens, o, rng);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(argc, argv);
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
}
