// Megakernel experiment: all matvecs of one token (real weights, 10 experts per layer) in three ways: one kernel
// per matvec, one kernel per phase, one persistent kernel for everything (own barrier with a time limit). Small
// steps (norms, DeltaNet, attention) are missing; the phases follow the dependencies of the real forward pass.
// Usage: tona-gpu-chain MODEL.gguf [--reps N] [--runs N] [--q8]
//   --q8: int8 path for all formats that have it (x quantized once, all tasks read it)
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <format>
#include <functional>
#include <random>
#include <stdexcept>
#include <vector>

#include "gpu/matvec.h"
#include "model/qwen4exp.h"
#include "tona_args.h"

void hip_check(hipError_t err, const char* what) {
    if (err != hipSuccess) throw std::runtime_error(std::format("{}: {}", what, hipGetErrorString(err)));
}
#define HIPCHECK(x) hip_check((x), #x)

namespace {

using tona::TensorInfo;
using tona::gpu::MatvecTask;

struct Options {
    int reps = 10;
    int runs = 7;
    bool q8 = false;
};

Options parse_options(int argc, char** argv) {
    Options o;
    for (int i = 2; i < argc; i++)
        if (!std::strcmp(argv[i], "--q8")) o.q8 = true;
    for (int i = 2; i + 1 < argc; i++) {
        if (!std::strcmp(argv[i], "--reps"))
            o.reps = tona::tools::parse_int(argv[i + 1]);
        else if (!std::strcmp(argv[i], "--runs"))
            o.runs = tona::tools::parse_int(argv[i + 1]);
    }
    return o;
}

// Collects the matvec tasks of one token in phases
struct Builder {
    std::vector<MatvecTask> tasks;
    std::vector<int> phase_start{0};
    double bytes = 0;  // weight bytes per token
    int64_t out_rows = 0;
    const float* x = nullptr;

    const uint8_t* upload(const uint8_t* src, size_t n) {
        void* p;
        HIPCHECK(hipMalloc(&p, n + tona::gpu::kMatvecReadPad));
        HIPCHECK(hipMemcpy(p, src, n, hipMemcpyHostToDevice));
        return static_cast<const uint8_t*>(p);
    }
    void add(uint32_t type, int64_t in, int64_t rows, const void* w, size_t nbytes) {
        tasks.push_back({type, in, rows, w, x, nullptr});
        out_rows += rows;
        bytes += (double)nbytes;
    }
    void dense(const TensorInfo* t) {
        if (!t) return;
        const int64_t in = (int64_t)t->dims[0];
        add(t->type, in, (int64_t)t->elements() / in, upload(t->data, t->bytes), t->bytes);
    }
    void phase() {
        if ((int)tasks.size() > phase_start.back()) phase_start.push_back((int)tasks.size());
    }
};

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// Random x for the longest input row, optionally quantized to int8
tona::gpu::Q8Vec make_input(const tona::Qwen4ExpModel& m, Builder& B, bool q8) {
    int64_t max_in = 0;
    for (const auto& L : m.layers())
        for (const TensorInfo* t : {L.gr_attn.up, L.gr_attn.down, L.qkv, L.ssm_out, L.o, L.q, L.down_exps})
            if (t) max_in = std::max<int64_t>(max_in, (int64_t)t->dims[0]);
    std::vector<float> xh((size_t)max_in);
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1, 1);
    for (auto& v : xh) v = U(rng);
    void* xd;
    HIPCHECK(hipMalloc(&xd, xh.size() * sizeof(float)));
    HIPCHECK(hipMemcpy(xd, xh.data(), xh.size() * sizeof(float), hipMemcpyHostToDevice));
    B.x = static_cast<const float*>(xd);
    tona::gpu::Q8Vec xq{};
    if (q8) {
        HIPCHECK(hipMalloc(&xq.q, max_in));
        HIPCHECK(hipMalloc(&xq.meta, max_in / 32 * sizeof(tona::gpu::Q8Meta)));
        tona::gpu::quantize_q8(B.x, max_in, xq);
        HIPCHECK(hipDeviceSynchronize());
    }
    return xq;
}

// 10 experts (here 0..9, contiguous in the tensor): gate+up, then down with the shared down
void add_experts(Builder& B, const tona::Layer& L, int n_exp) {
    const TensorInfo* parts[3] = {L.gate_exps, L.up_exps, L.down_exps};
    const uint8_t* dev[3];
    size_t per[3];
    for (int k = 0; k < 3; k++) {
        per[k] = parts[k]->bytes / parts[k]->dims[2];
        dev[k] = B.upload(parts[k]->data, per[k] * n_exp);
    }
    for (int e = 0; e < n_exp; e++)
        for (int k = 0; k < 2; k++)
            B.add(parts[k]->type, (int64_t)parts[k]->dims[0], (int64_t)parts[k]->dims[1], dev[k] + e * per[k], per[k]);
    B.phase();
    for (int e = 0; e < n_exp; e++)
        B.add(parts[2]->type, (int64_t)parts[2]->dims[0], (int64_t)parts[2]->dims[1], dev[2] + e * per[2], per[2]);
    B.dense(L.shexp_down), B.phase();
}

void add_layer(Builder& B, const tona::Qwen4ExpModel& m, uint32_t l) {
    const auto& h = m.hp();
    const auto& L = m.layers()[l];
    // hyper-connection before attention: down and inject on xn, then up
    B.dense(L.gr_attn.down), B.dense(L.gr_attn.inject), B.phase();
    B.dense(L.gr_attn.up), B.phase();
    if (h.is_qsa(l)) {
        B.dense(L.idx_k), B.dense(L.q), B.dense(L.k), B.dense(L.v), B.phase();
        B.dense(L.o), B.phase();
    } else {
        B.dense(L.qkv), B.dense(L.gate), B.dense(L.beta), B.dense(L.alpha), B.phase();
        B.dense(L.ssm_out), B.phase();
    }
    B.dense(L.gr_ffn.down), B.dense(L.gr_ffn.inject), B.phase();
    B.dense(L.gr_ffn.up), B.phase();
    // router and shared expert depend only on the mixed stream
    B.dense(L.router), B.dense(L.shexp_gate), B.dense(L.shexp_up), B.dense(L.shexp_gate_inp), B.phase();
    add_experts(B, L, (int)h.n_expert_used);
}

void build_plan(Builder& B, const tona::Qwen4ExpModel& m) {
    for (uint32_t l = 0; l < m.hp().n_layer; l++) add_layer(B, m, l);
    B.dense(m.gr_head().down), B.phase();
    B.dense(m.gr_head().up), B.phase();
    B.dense(m.output()), B.phase();
}

struct Variant {
    std::string name;
    std::function<void()> run;
};

// Every task writes into its own region, so the variants can be compared
class Bench {
public:
    Bench(Builder& B, tona::gpu::MatvecChain& chain, void* yd) : B_(B), chain_(chain), yd_(yd) {}

    void result(const std::function<void()>& f, std::vector<float>& out) const {
        HIPCHECK(hipMemset(yd_, 0, B_.out_rows * sizeof(float)));
        f();
        HIPCHECK(hipDeviceSynchronize());
        HIPCHECK(hipMemcpy(out.data(), yd_, B_.out_rows * sizeof(float), hipMemcpyDeviceToHost));
    }

    // ms per token for each of `runs` runs of `reps` calls
    std::vector<double> time(const Variant& v, const Options& o) const {
        hipEvent_t e0, e1;
        HIPCHECK(hipEventCreate(&e0));
        HIPCHECK(hipEventCreate(&e1));
        for (int i = 0; i < 3; i++) v.run();
        std::vector<double> ms;
        for (int r = 0; r < o.runs; r++) {
            HIPCHECK(hipEventRecord(e0));
            for (int i = 0; i < o.reps; i++) v.run();
            HIPCHECK(hipEventRecord(e1));
            HIPCHECK(hipEventSynchronize(e1));
            float t;
            HIPCHECK(hipEventElapsedTime(&t, e0, e1));
            ms.push_back(t / o.reps);
        }
        HIPCHECK(hipEventDestroy(e0));
        HIPCHECK(hipEventDestroy(e1));
        return ms;
    }

    void report(const Variant& v, const std::vector<float>& ref, const Options& o) {
        std::vector<float> got(ref.size());
        chain_.reset_diag();
        result(v.run, got);
        const bool same = std::memcmp(ref.data(), got.data(), ref.size() * sizeof(float)) == 0;
        const std::vector<double> ms = time(v, o);
        const int kernels = v.name[0] == 'A' ? (int)B_.tasks.size() : v.name[0] == 'B' ? chain_.phases() : 1;
        const double med = median(ms);
        std::string d = "- | - | -";
        if (v.name[0] == 'C') {
            const auto g = chain_.diag();
            d = std::format("{} | {:.1f} | {:.1f}", g.aborts, g.start_skew_us, g.max_wait_us);
        }
        printf("| %s | %d | %.2f | %.2f-%.2f | %.0f | %s | %s |\n", v.name.c_str(), kernels, med,
               *std::min_element(ms.begin(), ms.end()), *std::max_element(ms.begin(), ms.end()),
               B_.bytes / (med / 1e3) / 1e9, same ? "yes" : "NO", d.c_str());
    }

private:
    Builder& B_;
    tona::gpu::MatvecChain& chain_;
    void* yd_;
};

std::vector<Variant> make_variants(tona::gpu::MatvecChain& chain) {
    const int mb = chain.max_blocks();
    std::vector<Variant> vs = {
        {"A: one kernel per matvec", [&] { chain.run_single(); }},
        {"B: one kernel per phase", [&] { chain.run_grouped(); }},
        {std::format("C: persistent, {} blocks", mb), [&] { chain.run_persistent(mb); }},
    };
    for (int b : {64, 48, 32})
        if (b < mb) vs.push_back({std::format("C: persistent, {} blocks", b), [&, b] { chain.run_persistent(b); }});
    return vs;
}

void run(const char* model_path, const Options& o) {
    tona::Qwen4ExpModel m(model_path);
    Builder B;
    const tona::gpu::Q8Vec xq = make_input(m, B, o.q8);
    build_plan(B, m);
    void* yd;
    HIPCHECK(hipMalloc(&yd, B.out_rows * sizeof(float)));
    float* y = static_cast<float*>(yd);
    for (auto& t : B.tasks) t.y = y, y += t.rows, t.xq = xq;
    tona::gpu::MatvecChain chain(B.tasks, B.phase_start);
    printf("%s: %zu matvecs in %d phases, %.3f GB weights per token (lower bound at 315 GB/s: %.2f ms)\n",
           o.q8 ? "int8" : "float", B.tasks.size(), chain.phases(), B.bytes / 1e9, B.bytes / 315e9 * 1e3);
    printf("persistent kernel: at most %d blocks of 256 threads at once\n\n", chain.max_blocks());

    // Equality: all variants against run_single
    Bench bench(B, chain, yd);
    std::vector<float> ref(B.out_rows);
    bench.result([&] { chain.run_single(); }, ref);
    printf(
        "| Variant | Kernels per token | ms per token (median) | min-max | GB/s | bit-exact | Aborts | Start skew us | "
        "Wait max us |\n"
        "|---|---:|---:|---:|---:|---|---:|---:|---:|\n");
    for (const Variant& v : make_variants(chain)) bench.report(v, ref, o);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-gpu-chain MODEL.gguf [--reps N] [--runs N] [--q8]\n");
        return 2;
    }
    try {
        run(argv[1], parse_options(argc, argv));
    } catch (const std::exception& e) {
        fprintf(stderr, "Error: %s\n", e.what());
        return 1;
    }
    return 0;
}
