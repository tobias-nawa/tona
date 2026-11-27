// cpu/experts against a reference: rows of real expert tensors (TONA_MODEL=<part 1 of the GGUF>) unpacked with
// dequantize_row and multiplied in double with the dequantized int8 x. Also the time for a whole expert.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <format>
#include <random>
#include <set>
#include <vector>

#include "backend/env.h"
#include "cpu/experts.h"
#include "model/gguf.h"
#include "model/quant.h"

namespace {

constexpr int NB = 4;

struct Rng {
    std::mt19937 gen{7};
    std::normal_distribution<float> normal{0.0f, 1.0f};
};

int64_t row_bytes_of(const tona::TensorInfo* t) {
    const tona::GgmlTypeInfo* ti = tona::ggml_type_info(t->type);
    return (int64_t)t->dims[0] / ti->block_elems * ti->block_bytes;
}

// NB random vectors of length n, quantized to int8 with metadata
struct QuantX {
    std::vector<int8_t> q;
    std::vector<tona::cpu::Q8Meta> m;
    std::vector<float> dq;  // dequantized
};

QuantX make_x(Rng& rng, int64_t n) {
    std::vector<float> x(NB * n);
    for (auto& v : x) v = rng.normal(rng.gen);
    QuantX r{std::vector<int8_t>(NB * n), std::vector<tona::cpu::Q8Meta>(NB * n / 32), std::vector<float>(NB * n)};
    for (int b = 0; b < NB; b++)
        tona::cpu::quantize_q8(x.data() + b * n, n, r.q.data() + b * n, r.m.data() + b * n / 32);
    for (int64_t i = 0; i < NB * n; i++) r.dq[i] = r.q[i] * r.m[i / 32].d;
    return r;
}

// Runs matvec_q8 for one mask and row range; returns the worst relative deviation and counts failures
struct MatvecCase {
    const tona::TensorInfo& t;
    const uint8_t* w;
    int64_t rb;
    const QuantX& x;
    const std::vector<std::vector<float>>& deq;
};

struct Totals {
    double worst = 0;
    int checked = 0, fail = 0;
};

void check_mask(const MatvecCase& c, uint32_t mask, std::pair<int64_t, int64_t> rows, Totals& tot) {
    const auto [r0, r1] = rows;
    const int64_t in = (int64_t)c.t.dims[0], out = (int64_t)c.t.dims[1];
    std::vector<float> y(NB * out, -7.0f);
    tona::cpu::matvec_q8({c.t.type, c.w, c.rb, in, r0, r1, c.x.q.data(), c.x.m.data(), in, NB, mask, y.data(), out});
    for (int64_t r = 0; r < out; r++)
        for (int b = 0; b < NB; b++) {
            if (!(mask >> b & 1) || r < r0 || r >= r1) {
                if (y[b * out + r] != -7.0f) tot.fail++;  // nothing may be written outside
                continue;
            }
            double ref = 0, mag = 0;
            for (int64_t i = 0; i < in; i++)
                ref += (double)c.deq[r][i] * c.x.dq[b * in + i], mag += std::fabs(c.deq[r][i] * c.x.dq[b * in + i]);
            tot.worst = std::max(tot.worst, std::fabs(y[b * out + r] - ref) / std::max(mag, 1e-20));
            tot.checked++;
        }
}

// One expert tensor type: matvec against the reference for several masks and row ranges, then the timing
void check_matvec_type(const tona::TensorInfo& t, Rng& rng, int& checked, int& fail) {
    const int64_t in = (int64_t)t.dims[0], out = (int64_t)t.dims[1];
    const int64_t rb = row_bytes_of(&t);
    const uint8_t* w = t.data + (size_t)(rng.gen() % t.dims[2]) * out * rb;  // a random expert
    const QuantX x = make_x(rng, in);
    // Masks: contiguous (NB = 1 ... 4 in the kernel), with a gap, single; row ranges with a remainder (not a multiple
    // of 4)
    std::vector<std::vector<float>> deq(out, std::vector<float>(in));
    for (int64_t r = 0; r < out; r++) tona::dequantize_row(t.type, w + r * rb, deq[r].data(), in);
    const MatvecCase c{t, w, rb, x, deq};
    Totals tot;
    for (const uint32_t mask : {0b1011u, 0b1111u, 0b0111u, 0b1110u, 0b0011u, 0b0001u, 0b1000u})
        for (const auto& rows : {std::pair<int64_t, int64_t>{0, out}, {3, out - 2}}) check_mask(c, mask, rows, tot);
    // Time: all rows, 1 token, one thread
    std::vector<float> y(NB * out);
    auto t0 = std::chrono::steady_clock::now();
    const int reps = 20;
    for (int k = 0; k < reps; k++)
        tona::cpu::matvec_q8({t.type, w, rb, in, 0, out, x.q.data(), x.m.data(), in, 1, 1, y.data(), out});
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    printf("%-22s type %2u: worst relative deviation %.2e, 1 thread %.0f us per matrix (%.1f GB/s)\n", t.name.c_str(),
           t.type, tot.worst, s * 1e6, out * rb / s / 1e9);
    checked += tot.checked;
    fail += tot.fail + (tot.worst > 1e-5 ? 1 : 0);
}

int check_matvec(const tona::Gguf& g, Rng& rng) {
    int fail = 0, checked = 0;
    std::set<uint32_t> seen;
    for (const auto& t : g.tensors()) {
        if (!t.name.contains("_exps") || seen.contains(t.type)) continue;
        seen.insert(t.type);
        if (!tona::cpu::expert_type_supported(t.type)) {
            fprintf(stderr, "not supported: %s (%u)\n", t.name.c_str(), t.type);
            fail++;
            continue;
        }
        check_matvec_type(t, rng, checked, fail);
    }
    printf("%d values compared\n", checked);
    return fail;
}

struct LayerTensors {
    const tona::TensorInfo *gate, *up, *down;
};

// Reference for whole experts in double; the activation is rounded to int8 like in the kernel
std::vector<double> reference_ffn(const LayerTensors& L, const std::vector<tona::cpu::ExpertW>& E, const QuantX& x,
                                  const std::vector<float>& w, int64_t d, int64_t ff) {
    const int ne = (int)E.size();
    std::vector<double> ref(NB * d, 0.0);
    std::vector<float> row(d), a(ff), aq_d(ff);
    std::vector<int8_t> aq(ff);
    std::vector<tona::cpu::Q8Meta> am(ff / 32);
    auto dot_x = [&](const uint8_t* wrow, const tona::TensorInfo* t, int b) {
        tona::dequantize_row(t->type, wrow, row.data(), d);
        double acc = 0;
        for (int64_t i = 0; i < d; i++) acc += (double)row[i] * x.q[b * d + i] * x.m[(b * d + i) / 32].d;
        return acc;
    };
    for (int b = 0; b < NB; b++)
        for (int e = 0; e < ne; e++) {
            if (w[b * ne + e] == 0) continue;
            for (int64_t r = 0; r < ff; r++) {
                const double gg = dot_x(E[e].gate + r * row_bytes_of(L.gate), L.gate, b);
                const double uu = dot_x(E[e].up + r * row_bytes_of(L.up), L.up, b);
                a[r] = (float)(gg / (1 + std::exp(-gg)) * uu);
            }
            tona::cpu::quantize_q8(a.data(), ff, aq.data(), am.data());
            for (int64_t i = 0; i < ff; i++) aq_d[i] = aq[i] * am[i / 32].d;
            for (int64_t r = 0; r < d; r++) {
                tona::dequantize_row(L.down->type, E[e].down + r * row_bytes_of(L.down), row.data(), ff);
                double acc = 0;
                for (int64_t i = 0; i < ff; i++) acc += (double)row[i] * aq_d[i];
                ref[b * d + r] += w[b * ne + e] * acc;
            }
        }
    return ref;
}

// Each expert at 1 to 4 tokens (the misses of a verify pass; with 4 tokens also three or all)
std::vector<float> make_weights(int ne) {
    std::vector<float> w((size_t)NB * ne, 0.0f);
    for (int e = 0; e < ne; e++) {
        const int k = 1 + e % NB;  // this many tokens, starting at token e % NB
        for (int j = 0; j < k; j++) w[((e + j) % NB) * ne + e] = 0.1f + 0.05f * e - 0.01f * j;
    }
    return w;
}

// Whole experts of one layer (experts_ffn) against the reference, then the time with all threads
int check_layer(const tona::Gguf& g, Rng& rng, int layer, int ne) {
    auto T = [&](const char* k) { return g.find(std::format("blk.{}.ffn_{}_exps.weight", layer, k)); };
    const LayerTensors L{T("gate"), T("up"), T("down")};
    const int64_t d = (int64_t)L.gate->dims[0], ff = (int64_t)L.gate->dims[1];  // [d, ff, experts]
    std::vector<tona::cpu::ExpertW> E;
    for (int e = 0; e < ne; e++) {
        const int id = (int)(rng.gen() % L.gate->dims[2]);
        E.push_back({L.gate->type, L.up->type, L.down->type, L.gate->data + (size_t)id * ff * row_bytes_of(L.gate),
                     L.up->data + (size_t)id * ff * row_bytes_of(L.up),
                     L.down->data + (size_t)id * d * row_bytes_of(L.down)});
    }
    const std::vector<float> w = make_weights(ne);
    const QuantX x = make_x(rng, d);
    std::vector<float> y(NB * d);
    tona::cpu::ExpertScratch sc;
    const tona::cpu::FfnArgs args{E, d, ff, x.q.data(), x.m.data(), NB, w.data()};
    tona::cpu::experts_ffn(args, y.data(), sc);
    const std::vector<double> ref = reference_ffn(L, E, x, w, d, ff);
    double num = 0, den = 0;
    for (int64_t i = 0; i < NB * d; i++) num += (y[i] - ref[i]) * (y[i] - ref[i]), den += ref[i] * ref[i];
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    const int reps = 50;
    tona::cpu::experts_ffn(args, y.data(), sc);
    auto t0 = std::chrono::steady_clock::now();
    for (int k = 0; k < reps; k++) tona::cpu::experts_ffn(args, y.data(), sc);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() / reps;
    const double bytes = ne * (ff * (row_bytes_of(L.gate) + row_bytes_of(L.up)) + d * row_bytes_of(L.down));
    printf("layer %2d, %2d experts, 4 tokens: relative error %.2e, %.0f us (%.0f us per expert, %.1f GB/s)\n", layer,
           ne, rel, s * 1e6, s * 1e6 / ne, bytes / s / 1e9);
    return rel > 1e-3 ? 1 : 0;
}

// IQ2_XXS rows from random blocks (every grid index, sign index and scale is legal), needs no model: an odd row count,
// the two row lengths of the Bolt experts (up 1024 wide, down up to 2560)
int check_random_iq2xxs(Rng& rng) {
    int fail = 0, checked = 0;
    for (const int64_t in : {1024, 2560}) {
        const int64_t out = 77, rb = in / 256 * 66;
        std::vector<uint8_t> bytes((size_t)(out * rb));
        for (auto& b : bytes) b = (uint8_t)rng.gen();
        for (size_t o = 0; o < bytes.size(); o += 66) {
            const _Float16 d = (_Float16)(0.002f + 0.0001f * (float)(rng.gen() % 100));
            std::memcpy(&bytes[o], &d, 2);
        }
        tona::TensorInfo t;
        t.name = std::format("random IQ2_XXS {}", in), t.type = 16, t.dims = {(uint64_t)in, (uint64_t)out, 1};
        t.data = bytes.data();
        check_matvec_type(t, rng, checked, fail);
    }
    return fail;
}

// Random IQ2_XXS blocks (any bytes are legal, only the scale is set to a sane value)
std::vector<uint8_t> random_iq2xxs(Rng& rng, int64_t rows, int64_t in) {
    std::vector<uint8_t> bytes((size_t)(rows * (in / 256 * 66)));
    for (auto& b : bytes) b = (uint8_t)rng.gen();
    for (size_t o = 0; o < bytes.size(); o += 66) {
        const _Float16 d = (_Float16)(0.002f + 0.0001f * (float)(rng.gen() % 100));
        std::memcpy(&bytes[o], &d, 2);
    }
    return bytes;
}

struct Width {
    int64_t d, ff;
};

// One expert of one token added to ref: up in double, relu(u)^2 rounded to int8 like the kernel, down in double
void ungated_reference(const tona::cpu::ExpertW& E, const QuantX& x, int b, float wgt, const Width& wd,
                       std::vector<double>& ref) {
    const int64_t d = wd.d, ff = wd.ff;
    const int64_t rb_up = d / 256 * 66, rb_down = ff / 256 * 66;
    std::vector<float> row(std::max(d, ff)), a(ff), aq_d(ff);
    std::vector<int8_t> aq(ff);
    std::vector<tona::cpu::Q8Meta> am(ff / 32);
    for (int64_t r = 0; r < ff; r++) {
        tona::dequantize_row(16, E.up + r * rb_up, row.data(), d);
        double u = 0;
        for (int64_t i = 0; i < d; i++) u += (double)row[i] * x.q[b * d + i] * x.m[(b * d + i) / 32].d;
        a[r] = u > 0 ? (float)(u * u) : 0.0f;
    }
    tona::cpu::quantize_q8(a.data(), ff, aq.data(), am.data());
    for (int64_t i = 0; i < ff; i++) aq_d[i] = aq[i] * am[i / 32].d;
    for (int64_t r = 0; r < d; r++) {
        tona::dequantize_row(16, E.down + r * rb_down, row.data(), ff);
        double acc = 0;
        for (int64_t i = 0; i < ff; i++) acc += (double)row[i] * aq_d[i];
        ref[r] += wgt * acc;
    }
}

// The ungated experts of Bolt (up, relu(u)^2, down; latent width in and out) through experts_ffn against a double
// reference that rounds the activation to int8 like the kernel; needs no model
int check_random_ungated(Rng& rng) {
    const int64_t d = 1024, ff = 1536;
    const int ne = 9;
    std::vector<std::vector<uint8_t>> up, down;
    std::vector<tona::cpu::ExpertW> E;
    for (int e = 0; e < ne; e++) {
        up.push_back(random_iq2xxs(rng, ff, d)), down.push_back(random_iq2xxs(rng, d, ff));
        E.push_back({0, 16, 16, nullptr, up.back().data(), down.back().data()});
    }
    const std::vector<float> w = make_weights(ne);
    const QuantX x = make_x(rng, d);
    std::vector<float> y(NB * d);
    tona::cpu::ExpertScratch sc;
    tona::cpu::experts_ffn({E, d, ff, x.q.data(), x.m.data(), NB, w.data()}, y.data(), sc);
    double num = 0, den = 0;
    for (int b = 0; b < NB; b++) {
        std::vector<double> ref(d, 0.0);
        for (int e = 0; e < ne; e++)
            if (w[b * ne + e] != 0) ungated_reference(E[e], x, b, w[b * ne + e], {d, ff}, ref);
        for (int64_t r = 0; r < d; r++)
            num += (y[b * d + r] - ref[r]) * (y[b * d + r] - ref[r]), den += ref[r] * ref[r];
    }
    const double rel = std::sqrt(num / std::max(den, 1e-30));
    printf("ungated IQ2_XXS experts, %d experts, 4 tokens: relative error %.2e\n", ne, rel);
    return rel > 1e-3 ? 1 : 0;
}

int run() {
    Rng rng;
    int fail = check_random_iq2xxs(rng);
    fail += check_random_ungated(rng);
    const char* path = tona::env::text("TONA_MODEL");
    if (!path) {
        printf("TONA_MODEL not set: only the random IQ2_XXS rows were checked\n");
        return fail;
    }
    tona::Gguf g(path);
    fail += check_matvec(g, rng);
    int n_layers = 0;
    for (const auto& t : g.tensors()) n_layers += t.name.contains("ffn_gate_exps");
    // Whole experts need gate, up and down (the Bolt experts have no gate)
    if (n_layers > 0)
        for (int layer : {0, 1, 17, n_layers - 1})
            for (int ne : {6, 12}) fail += check_layer(g, rng, layer, ne);
    if (fail) {
        fprintf(stderr, "%d failures\n", fail);
        return 1;
    }
    return 0;
}

}  // namespace

int main() {
    try {
        return run();
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
