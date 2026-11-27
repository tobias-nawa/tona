// Wide verify passes: time of a dense matvec with nb = 1 ... 8 tokens on real weights, the megakernel's int8 path
// (row_q8, x in LDS, 32 blocks x 16 waves like k_mega) against f16 WMMA with unpacking in LDS (gemm_q of the
// prefill) and the k_wide candidate. Per format the tensors with row length d are joined into one matrix up to
// --cap MB (above the Infinity Cache).
// Usage: tona-wide-probe MODEL.gguf [--types 14,12] [--cap 256] [--runs 20] [--hc | --exps [--phase N]]
//   --hc: hyper-connection matvecs in Q8_0, 320 x 10240 and 10240 x 320, random weights in 64 copies in turn (cold
//     as in a pass), 32 blocks x 16 waves, nb = 1 and 4, us per step (including ~2 us kernel launch)
//   --tile: Q6_K layer matrices in GGUF order against TONA_Q6_K_T (tile order), row kernel and k_wide, W = 1..8,
//   outputs
//     compared bit for bit (tona_wide_probe_tile.h)
//   --abl: row loop of the expert matrices with parts left out (tona_wide_probe_abl.h)
//   --exps: expert matrices per (type, row length), int8 path with L lanes per row, once large (throughput) and once
//     as large as a phase in the megakernel (--phase experts, 32 blocks as k_mega, 50 launches in a row)
// TONA_PROBE_NBS: widths of --exps (TONA_PROBE_LANES, TONA_PROBE_PHASE: see run_exps_type). TONA_HC_NC: weight copies
// for --hc (default 64; 1 = warm, always the same weights). TONA_HC_MULT: row multiplier.
#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <random>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "backend/env.h"
#include "gpu/gemm.h"
#include "gpu/matvec_dev.h"
#include "model/gguf.h"
#include "model/quant.h"
#include "tona_args.h"
#include "tona_hip_check.h"
#include "tona_wide_probe_abl.h"
#include "tona_wide_probe_kernels.h"
#include "tona_wide_probe_tile.h"

using namespace tona;
using namespace tona::gpu;
using tona::tools::DeviceBuffer;

namespace {

struct Options {
    std::vector<uint32_t> types = {14, 12};
    int64_t cap = 256ll << 20;
    int runs = 20;
    bool hc = false;
    bool exps = false;
    bool tile = false;
    bool abl = false;
    int phase = 10;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; i++) {
        if (!std::strcmp(argv[i], "--hc")) o.hc = true;
        if (!std::strcmp(argv[i], "--exps")) o.exps = true;
        if (!std::strcmp(argv[i], "--tile")) o.tile = true;
        if (!std::strcmp(argv[i], "--abl")) o.abl = true;
        if (i + 1 >= argc) break;
        if (!std::strcmp(argv[i], "--types")) {
            o.types.clear();
            std::stringstream ss(argv[i + 1]);
            for (std::string s; std::getline(ss, s, ',');)
                o.types.push_back((uint32_t)tona::tools::parse_int64(s.c_str()));
        } else if (!std::strcmp(argv[i], "--cap")) {
            o.cap = tona::tools::parse_int64(argv[i + 1]) << 20;
        } else if (!std::strcmp(argv[i], "--runs")) {
            o.runs = tona::tools::parse_int(argv[i + 1]);
        } else if (!std::strcmp(argv[i], "--phase")) {
            o.phase = tona::tools::parse_int(argv[i + 1]);
        }
    }
    return o;
}

// Weights of one matrix and x of 8 tokens (float, f16 and int8) on the device
struct ProbeData {
    ProbeData(const std::vector<uint8_t>& host, int64_t in, int64_t rows, std::mt19937& rng)
        : in(in),
          dw(host.size() + kMatvecReadPad),
          dx(8 * in * 4),
          dx16(8 * in * 2),
          dy(8 * rows * 4),
          q(8 * in),
          meta(8 * in / 32 * sizeof(Q8Meta)) {
        CHECK(hipMemcpy(dw.get(), host.data(), host.size(), hipMemcpyHostToDevice));
        std::normal_distribution<float> nd(0, 1);
        std::vector<float> x(8 * in);
        std::vector<_Float16> x16(8 * in);
        for (size_t i = 0; i < x.size(); i++) x[i] = nd(rng), x16[i] = (_Float16)x[i];
        CHECK(hipMemcpy(dx.get(), x.data(), x.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dx16.get(), x16.data(), x16.size() * 2, hipMemcpyHostToDevice));
        for (int b = 0; b < 8; b++)
            quantize_q8(dx.get<float>() + b * in, in,
                        Q8Vec{q.get<int8_t>() + b * in, meta.get<Q8Meta>() + b * in / 32});
    }
    int8_t* xq() const { return q.get<int8_t>(); }
    Q8Meta* xm() const { return meta.get<Q8Meta>(); }

    int64_t in;
    DeviceBuffer dw, dx, dx16, dy, q, meta;
};

void launch_probe(int nb, int blocks, const DevTask* dt, const ProbeData& d) {
    if (nb == 1)
        hipLaunchKernelGGL(k_probe<1>, dim3(blocks), dim3(PT), 0, 0, dt, d.xq(), d.xm(), nb, d.in);
    else if (nb <= 4)
        hipLaunchKernelGGL(k_probe<4>, dim3(blocks), dim3(PT), 0, 0, dt, d.xq(), d.xm(), nb, d.in);
    else
        hipLaunchKernelGGL(k_probe<8>, dim3(blocks), dim3(PT), 0, 0, dt, d.xq(), d.xm(), nb, d.in);
}

// Median of `runs` timed calls of go(), divided by `per_call` launches inside it; us
template <class Go>
double median_us(Go go, int runs, int per_call) {
    go();
    CHECK(hipDeviceSynchronize());
    std::vector<double> us;
    for (int r = 0; r < runs; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        go();
        CHECK(hipDeviceSynchronize());
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count() /
                     per_call);
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

DevTask make_task(const ProbeData& d, uint32_t type, int64_t rows, int64_t row_bytes, int lanes, int64_t xb_rows) {
    DevTask t{d.dw.get<uint8_t>(),
              nullptr,
              d.dy.get<float>(),
              d.xq(),
              d.xm(),
              rows,
              (rows + WAVE / lanes - 1) / (WAVE / lanes),
              row_bytes,
              0,
              type,
              (int32_t)d.in,
              lanes};
    t.xb = d.in, t.yb = xb_rows;
    return t;
}

// --hc

void run_hc_matrix(int64_t rows, int64_t in, int copies, std::mt19937& rng, int cus) {
    const int64_t rb = in / 32 * 34, bytes = rows * rb;
    std::vector<uint8_t> host(bytes * copies);
    for (int64_t k = 0; k < (int64_t)host.size(); k++) host[k] = (uint8_t)rng();
    for (int64_t r = 0; r < rows * copies; r++)  // scales as small f16
        for (int64_t b = 0; b < in / 32; b++) {
            const _Float16 h = (_Float16)0.01f;
            std::memcpy(&host[r * rb + b * 34], &h, 2);
        }
    ProbeData d(host, in, rows, rng);
    const int lanes = std::min<int>(32, lanes_per_row(in));
    std::vector<DevTask> tasks;
    for (int k = 0; k < copies; k++)
        tasks.push_back(make_task(d, 8u, rows, rb, lanes, rows)), tasks.back().w = d.dw.get<uint8_t>() + k * rows * rb;
    DeviceBuffer dt(copies * sizeof(DevTask));
    CHECK(hipMemcpy(dt.get(), tasks.data(), copies * sizeof(DevTask), hipMemcpyHostToDevice));
    for (int nb : {1, 4}) {
        auto go = [&] {
            for (int k = 0; k < 64; k++) launch_probe(nb, cus, dt.get<DevTask>() + k % copies, d);
        };
        const double us = median_us(go, 9, 64);
        std::vector<float> yv(4 * rows);
        CHECK(hipMemcpy(yv.data(), d.dy.get(), yv.size() * 4, hipMemcpyDeviceToHost));
        double h = 0;
        for (float v : yv) h = h * 1.000001 + v;
        printf("%lld x %lld, L %d, nb %d, %.1f us per step (%.0f GB/s), checksum %.9g\n", (long long)rows,
               (long long)in, lanes, nb, us, rows * rb / us / 1e3, h);
    }
}

int run_hc(int cus) {
    const int copies = (int)env::integer("TONA_HC_NC", 64);
    const int64_t mult = env::integer("TONA_HC_MULT", 1);
    std::mt19937 rng(7);
    for (auto [rows0, in] : {std::pair<int64_t, int64_t>{320, 10240}, {10240, 320}})
        run_hc_matrix(rows0 * mult, in, copies, rng, cus);
    return 0;
}

// --exps

// Rows of the tensors collected up to the cap
std::vector<uint8_t> collect_rows(const std::vector<const TensorInfo*>& ts, int64_t rb, int64_t cap) {
    std::vector<uint8_t> host;
    for (const TensorInfo* t : ts) {
        const int64_t take = std::min<int64_t>((int64_t)t->bytes, cap - (int64_t)host.size());
        if (take <= 0) break;
        host.insert(host.end(), t->data, t->data + take / rb * rb);
    }
    return host;
}

struct LaunchSpec {
    uint32_t type;
    int64_t rows;  // rows computed by the launch
    int64_t row_bytes;
    int lanes;
    int nb;
    int reps;  // launches in a row
    int blocks;
};

// us per launch of the task, `reps` launches in a row
double expert_launch_us(const ProbeData& d, DevTask* dt, const LaunchSpec& l, int64_t total_rows) {
    const DevTask task = make_task(d, l.type, l.rows, l.row_bytes, l.lanes, total_rows);
    CHECK(hipMemcpy(dt, &task, sizeof task, hipMemcpyHostToDevice));
    auto go = [&] {
        for (int k = 0; k < l.reps; k++) launch_probe(l.nb, l.blocks, dt, d);
    };
    return median_us(go, 7, l.reps);
}

// TONA_PROBE_NBS=1,2,4,8: widths of --exps (default 1 and 4)
std::vector<int> probe_nbs() {
    std::vector<int> v;
    const char* list = env::text("TONA_PROBE_NBS");
    for (const char* p = list ? list : "1,4"; *p;) {
        char* end = nullptr;
        const long n = std::strtol(p, &end, 10);
        if (end == p) break;
        if (n >= 1 && n <= 8) v.push_back((int)n);
        p = *end == ',' ? end + 1 : end;
    }
    return v;
}

void run_exps_type(uint32_t type, uint64_t in, const std::vector<const TensorInfo*>& ts, const Options& o, int cus) {
    const GgmlTypeInfo* ti = ggml_type_info(type);
    const int64_t rb = (int64_t)in / ti->block_elems * ti->block_bytes, rpe = (int64_t)ts[0]->dims[1];
    const std::vector<uint8_t> host = collect_rows(ts, rb, o.cap);
    const int64_t rows = (int64_t)host.size() / rb;
    std::mt19937 rng(1);
    ProbeData d(host, (int64_t)in, rows, rng);
    DeviceBuffer dt(sizeof(DevTask));
    // TONA_PROBE_LANES=8: one L; TONA_PROBE_PHASE=0: no phase launches (for a counter run: fewer dispatches)
    const int only_l = (int)env::integer("TONA_PROBE_LANES", 0);
    const bool phase = env::integer("TONA_PROBE_PHASE", 1) != 0;
    for (int lanes : {4, 8, 16, 32}) {
        if ((int64_t)in / 32 < lanes || (only_l && lanes != only_l)) continue;
        for (int nb : probe_nbs()) {
            const double tb = expert_launch_us(d, dt.get<DevTask>(), {type, rows, rb, lanes, nb, 3, cus}, rows);
            const int64_t pr = std::min<int64_t>(rows, (int64_t)o.phase * rpe);
            const double tp =
                phase ? expert_launch_us(d, dt.get<DevTask>(), {type, pr, rb, lanes, nb, 50, cus}, rows) : 0;
            const double mb = rows * rb / 1e6, pmb = pr * rb / 1e6;
            printf("| %u | %llu | %lld | %d | %d | %.0f | %.1f | %.0f |\n", type, (unsigned long long)in,
                   (long long)rpe, lanes, nb, mb / tb * 1e3, tp, pmb / tp * 1e3);
        }
    }
}

int run_exps(const Gguf& g, const Options& o, int cus) {
    std::map<std::pair<uint32_t, uint64_t>, std::vector<const TensorInfo*>> by;
    for (const auto& t : g.tensors())
        if (t.name.contains("_exps") && t.dims.size() == 3) by[{t.type, t.dims[0]}].push_back(&t);
    printf(
        "| Type | Row length | Rows per expert | L | nb | large GB/s | Phase (%d exp.) us | GB/s |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|\n",
        o.phase);
    for (const auto& [key, ts] : by) run_exps_type(key.first, key.second, ts, o, cus);
    return 0;
}

// --abl: ablation of the row loop per format (tona_wide_probe_abl.h)

struct AblRun {
    const ProbeData* d;
    int64_t rows, rb;
    int blocks;
};

template <uint32_t T, int MODE, int NB>
void launch_abl(const AblRun& r) {
    const AblArgs a{r.d->dw.get<uint8_t>(), r.rows, r.rb, (int)r.d->in, r.d->xq(), r.d->xm(), r.d->dy.get<float>()};
    hipLaunchKernelGGL((k_abl<T, MODE, NB>), dim3(r.blocks), dim3(PT), 0, 0, a);
}

template <uint32_t T, int MODE>
void launch_abl_nb(int nb, const AblRun& r) {
    if (nb == 1) launch_abl<T, MODE, 1>(r);
    if (nb == 4) launch_abl<T, MODE, 4>(r);
    if (nb == 8) launch_abl<T, MODE, 8>(r);
}

template <uint32_t T>
void launch_abl_mode(int mode, int nb, const AblRun& r) {
    if (mode == 0) launch_abl_nb<T, 0>(nb, r);
    if (mode == 1) launch_abl_nb<T, 1>(nb, r);
    if (mode == 2) launch_abl_nb<T, 2>(nb, r);
    if (mode == 3) launch_abl_nb<T, 3>(nb, r);
    if (mode == 4) launch_abl_nb<T, 4>(nb, r);
}

void launch_abl_type(uint32_t type, int mode, int nb, const AblRun& r) {
    switch (type) {
        case 18:
            return launch_abl_mode<18>(mode, nb, r);
        case 20:
            return launch_abl_mode<20>(mode, nb, r);
        case 21:
            return launch_abl_mode<21>(mode, nb, r);
        case 22:
            return launch_abl_mode<22>(mode, nb, r);
        case 23:
            return launch_abl_mode<23>(mode, nb, r);
        case 42:
            return launch_abl_mode<42>(mode, nb, r);
        default:
            return;
    }
}

int run_abl(const Gguf& g, const Options& o, int blocks) {
    std::map<std::pair<uint32_t, uint64_t>, std::vector<const TensorInfo*>> by;
    for (const auto& t : g.tensors())
        if (t.name.contains("_exps") && t.dims.size() == 3) by[{t.type, t.dims[0]}].push_back(&t);
    printf(
        "| Type | nb | loads only GB/s | decode GB/s | decode + dot, x in registers GB/s | row_q8 GB/s | decode, sign "
        "masks by arithmetic GB/s |\n"
        "|---|---:|---:|---:|---:|---:|---:|\n");
    for (const auto& [key, ts] : by) {
        const GgmlTypeInfo* ti = ggml_type_info(key.first);
        const int64_t rb = (int64_t)key.second / ti->block_elems * ti->block_bytes;
        const std::vector<uint8_t> host = collect_rows(ts, rb, o.cap);
        const int64_t rows = (int64_t)host.size() / rb;
        std::mt19937 rng(1);
        ProbeData d(host, (int64_t)key.second, rows, rng);
        for (int nb : probe_nbs()) {
            if (nb == 2 || nb == 3) continue;  // 1, 4 and 8 are compiled
            printf("| %u | %d |", key.first, nb);
            for (int mode = 0; mode < 5; mode++) {
                const double us =
                    median_us([&] { launch_abl_type(key.first, mode, nb, AblRun{&d, rows, rb, blocks}); }, o.runs, 1);
                printf(" %.0f |", rows * rb / us / 1e3);
            }
            printf("\n");
        }
    }
    return 0;
}

// default: int8 path against gemm_q and k_wide

void launch_wide(uint32_t type, int blocks, const WideArgs& a) {
    switch (type) {
        case 14:
            hipLaunchKernelGGL(k_wide<14>, dim3(blocks), dim3(PT), 0, 0, a);
            break;
        case 12:
            hipLaunchKernelGGL(k_wide<12>, dim3(blocks), dim3(PT), 0, 0, a);
            break;
        case 13:
            hipLaunchKernelGGL(k_wide<13>, dim3(blocks), dim3(PT), 0, 0, a);
            break;
        case 8:
            hipLaunchKernelGGL(k_wide<8>, dim3(blocks), dim3(PT), 0, 0, a);
            break;
        default:
            throw std::invalid_argument("k_wide: unsupported type");
    }
}

struct FormatRun {
    uint32_t type;
    int64_t rows;
    int64_t row_bytes;
    size_t bytes;
    int blocks;
};

void run_format_width(const FormatRun& f, const ProbeData& d, DevTask* dt, int nb, const Options& o) {
    const uint32_t type = f.type;
    const int64_t rows = f.rows, rb = f.row_bytes;
    const size_t bytes = f.bytes;
    const int blocks = f.blocks;
    const int64_t in = d.in;
    const double ti8 = median_us([&] { launch_probe(nb, blocks, dt, d); }, o.runs, 1);
    const GemmQSeg s{d.dx16.get<_Float16>(),
                     d.dw.get<uint8_t>(),
                     d.dy.get<float>(),
                     nb,
                     (int32_t)rows,
                     (int32_t)in,
                     (int32_t)in,
                     (int32_t)rows,
                     rb};
    const double tw = median_us([&] { gemm_q(type, &s, 1, nullptr); }, o.runs, 1);
    std::vector<float> yq((size_t)nb * rows), yw((size_t)nb * rows);
    CHECK(hipMemcpy(yq.data(), d.dy.get(), yq.size() * 4, hipMemcpyDeviceToHost));  // gemm_q: y[t * rows + n]
    const WideArgs wide{d.dw.get<uint8_t>(), rb, d.dx16.get<_Float16>(), nb, (int)in, rows, d.dy.get<float>()};
    const double tk = median_us([&] { launch_wide(type, blocks, wide); }, o.runs, 1);
    CHECK(hipMemcpy(yw.data(), d.dy.get(), yw.size() * 4, hipMemcpyDeviceToHost));
    double md = 0, mr = 0;
    for (size_t i = 0; i < yq.size(); i++)
        md = std::max(md, (double)std::fabs(yw[i] - yq[i])), mr = std::max(mr, (double)std::fabs(yq[i]));
    const double mb = bytes / 1e6;
    printf("| %u | %lld | %.0f | %d | %.0f | %.0f | %.0f | %.0f | %.0f | %.0f | %.1e |\n", type, (long long)rows, mb,
           nb, ti8, mb / ti8 * 1e3, tw, mb / tw * 1e3, tk, mb / tk * 1e3, md / mr);
}

void run_format(uint32_t type, const Gguf& g, const Options& o, int blocks) {
    constexpr int64_t in = 2560;
    const GgmlTypeInfo* ti = ggml_type_info(type);
    const int64_t rb = in / ti->block_elems * ti->block_bytes;
    std::vector<const TensorInfo*> same_length;
    for (const auto& t : g.tensors())
        if (t.type == type && t.dims[0] == (uint64_t)in && t.dims.size() >= 2) same_length.push_back(&t);
    const std::vector<uint8_t> host = collect_rows(same_length, rb, o.cap);
    const int64_t rows = (int64_t)host.size() / rb;
    if (rows == 0) return;
    std::mt19937 rng(1);
    ProbeData d(host, in, rows, rng);
    const int lanes = lanes_per_row(in);
    DeviceBuffer dt(sizeof(DevTask));
    const DevTask task = make_task(d, type, rows, rb, lanes, rows);
    CHECK(hipMemcpy(dt.get(), &task, sizeof task, hipMemcpyHostToDevice));
    for (int nb : {1, 2, 4, 6, 8}) run_format_width({type, rows, rb, host.size(), blocks}, d, dt.get<DevTask>(), nb, o);
}

int run_formats(const Gguf& g, const Options& o, int cus) {
    printf(
        "| Type | Rows | MB | nb | int8 us | GB/s | gemm_q us | GB/s | k_wide us | GB/s | k_wide error |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n");
    for (uint32_t type : o.types) run_format(type, g, o, 2 * cus);
    return 0;
}

}  // namespace

namespace {

int run(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-wide-probe MODEL.gguf [--types 14,12] [--cap 256] [--runs 20]\n");
        return 2;
    }
    const Options o = parse_options(argc, argv);
    Gguf g(argv[1]);
    upload_tables();
    int cus = 0;
    CHECK(hipDeviceGetAttribute(&cus, hipDeviceAttributeMultiprocessorCount, 0));
    if (o.hc) return run_hc(cus);
    if (o.exps) return run_exps(g, o, cus * (int)env::integer("TONA_PROBE_BLOCKS_PER_CU", 2));  // 32 blocks as k_mega
    if (o.abl) return run_abl(g, o, cus * (int)env::integer("TONA_PROBE_BLOCKS_PER_CU", 2));
    if (o.tile) return run_tile(g, o.cap, o.runs, cus * (int)env::integer("TONA_PROBE_BLOCKS_PER_CU", 2));
    return run_formats(g, o, cus);
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
