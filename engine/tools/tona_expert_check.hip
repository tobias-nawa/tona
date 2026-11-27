// Bit-exactness CPU vs GPU for the experts: per expert format one matrix of a real expert through
// gpu::matvec_q8 (8 lanes per row like the expert groups in the megakernel) and cpu::matvec_q8; also det::silu and
// the int8 rounding on both sides. Reports the first deviating place. Usage: tona-expert-check MODEL.gguf
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <set>
#include <vector>

#include "cpu/detmath.h"
#include "cpu/experts.h"
#include "gpu/matvec.h"
#include "model/gguf.h"
#include "tona_hip_check.h"

using tona::tools::DeviceBuffer;

__global__ void k_silu(const float* x, float* y, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = tona::det::silu(x[i]);
}

namespace {

uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

// silu over a wide range of inputs; returns 1 if GPU and CPU differ anywhere
int check_silu() {
    const int n = 1 << 20;
    std::vector<float> x(n), yg(n);
    for (int i = 0; i < n; i++) x[i] = (i % 2 ? 1 : -1) * std::ldexp((float)(i % 4096) / 4096.0f + 0.5f, i % 15 - 7);
    x[0] = 0, x[1] = 100, x[2] = -100, x[3] = 87.9f, x[4] = -87.9f;
    const size_t bytes = n * sizeof(float);
    DeviceBuffer dx(bytes), dy(bytes);
    CHECK(hipMemcpy(dx.get(), x.data(), bytes, hipMemcpyHostToDevice));
    k_silu<<<n / 256, 256>>>(dx.get<float>(), dy.get<float>(), n);
    CHECK(hipMemcpy(yg.data(), dy.get(), bytes, hipMemcpyDeviceToHost));
    int diff = 0;
    for (int i = 0; i < n; i++)
        if (bits(yg[i]) != bits(tona::det::silu(x[i])) && diff++ < 3)
            printf("silu(%a): GPU %a, CPU %a\n", x[i], yg[i], tona::det::silu(x[i]));
    printf("silu: %d of %d differ\n", diff, n);
    return diff > 0;
}

struct Q8Device {
    explicit Q8Device(int64_t in) : x(in * 4), q(in), meta(in / 32 * sizeof(tona::gpu::Q8Meta)) {}
    DeviceBuffer x, q, meta;
    tona::gpu::Q8Vec vec() const { return {q.get<int8_t>(), meta.get<tona::gpu::Q8Meta>()}; }
};

// int8 rounding of x: GPU against CPU; returns the number of differing values and scales
int compare_rounding(const std::vector<float>& x, int64_t in, const Q8Device& dev, std::vector<int8_t>& qc,
                     std::vector<tona::cpu::Q8Meta>& mc) {
    std::vector<int8_t> qg(in);
    std::vector<tona::gpu::Q8Meta> mg(in / 32);
    tona::cpu::quantize_q8(x.data(), in, qc.data(), mc.data());
    CHECK(hipMemcpy(dev.x.get(), x.data(), in * 4, hipMemcpyHostToDevice));
    tona::gpu::quantize_q8(dev.x.get<float>(), in, dev.vec());
    CHECK(hipMemcpy(qg.data(), dev.q.get(), in, hipMemcpyDeviceToHost));
    CHECK(hipMemcpy(mg.data(), dev.meta.get(), in / 32 * sizeof(tona::gpu::Q8Meta), hipMemcpyDeviceToHost));
    int diff = 0;
    for (int64_t i = 0; i < in; i++) diff += qc[i] != qg[i];
    for (int64_t i = 0; i < in / 32; i++)
        diff += bits(mc[i].d) != bits(mg[i].d) || mc[i].s0 != mg[i].s0 || mc[i].s1 != mg[i].s1;
    return diff;
}

struct ExpertMatrix {
    const tona::TensorInfo& tensor;
    const uint8_t* weights;
    int64_t row_bytes, in, rows;
};

// Matvec with the CPU's rounded x on both sides; returns the number of differing rows
int compare_matvec(const ExpertMatrix& mat, const Q8Device& dev, const std::vector<int8_t>& qc,
                   const std::vector<tona::cpu::Q8Meta>& mc) {
    const int64_t in = mat.in, rows = mat.rows, rb = mat.row_bytes;
    DeviceBuffer dy(rows * 4), dw(rows * rb + tona::gpu::kMatvecReadPad);
    CHECK(hipMemcpy(dw.get(), mat.weights, rows * rb, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.q.get(), qc.data(), in, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dev.meta.get(), mc.data(), in / 32 * sizeof(tona::gpu::Q8Meta), hipMemcpyHostToDevice));
    const uint32_t type = mat.tensor.type;
    tona::gpu::matvec_q8(tona::gpu::MatvecTask{type, in, rows, dw.get(), nullptr, dy.get<float>(), dev.vec()});
    std::vector<float> yg(rows), yc(rows);
    CHECK(hipMemcpy(yg.data(), dy.get(), rows * 4, hipMemcpyDeviceToHost));
    tona::cpu::matvec_q8({type, mat.weights, rb, in, 0, rows, qc.data(), mc.data(), in, 1, 1, yc.data(), rows});
    int diff = 0;
    for (int64_t r = 0; r < rows; r++)
        if (bits(yg[r]) != bits(yc[r]) && diff++ < 2) printf("  row %ld: GPU %a, CPU %a\n", (long)r, yg[r], yc[r]);
    return diff;
}

// One matrix of a random expert of this tensor; returns 1 on any difference
int check_expert_type(const tona::TensorInfo& t, std::mt19937& rng, std::normal_distribution<float>& nd) {
    const int64_t in = (int64_t)t.dims[0], rows = (int64_t)t.dims[1];
    const int64_t rb = tona::cpu::row_bytes(t.type, in);
    const uint8_t* w = t.data + (size_t)(rng() % t.dims[2]) * rows * rb;
    std::vector<float> x(in);
    for (auto& v : x) v = nd(rng) * 3.0f;
    std::vector<int8_t> qc(in);
    std::vector<tona::cpu::Q8Meta> mc(in / 32);
    Q8Device dev(in);
    const int rounding_diff = compare_rounding(x, in, dev, qc, mc);
    const int matvec_diff = compare_matvec({t, w, rb, in, rows}, dev, qc, mc);
    printf("%-28s type %2u: int8 rounding %d differ, matvec %d of %ld rows differ\n", t.name.c_str(), t.type,
           rounding_diff, matvec_diff, (long)rows);
    return rounding_diff > 0 || matvec_diff > 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: tona-expert-check MODEL.gguf\n");
        return 2;
    }
    setenv("TONA_Q8_LANES", "8", 1);
    tona::Gguf g(argv[1]);
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    int bad = check_silu();
    std::set<uint32_t> seen;
    for (const auto& t : g.tensors()) {
        if (!t.name.contains("_exps") || seen.contains(t.type)) continue;
        seen.insert(t.type);
        bad += check_expert_type(t, rng, nd);
    }
    printf(bad ? "DEVIATION\n" : "bit-exact\n");
    return bad ? 1 : 0;
}
