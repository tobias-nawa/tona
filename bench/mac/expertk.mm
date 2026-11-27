// Time of the expert mat-vecs of one layer (10 experts) for the Coder's expert types at B = 1 to 4 tokens, per type
// and for a whole token with the Coder's per-layer type mix. Weights are synthetic blocks in the real formats;
// outputs are checked against the CPU reference (engine/model/quant.cpp) before timing.
// Plan and evaluation: research/measurements/2026-11-11-mac-expertk.md
// Build: clang++ -std=c++20 -O2 -fobjc-arc -Iengine bench/mac/expertk.mm engine/model/quant.cpp
//        -framework Metal -framework Foundation -o work/build/mac/expertk
// Usage: expertk [repo root, default .]
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "model/quant.h"

namespace {

struct Type {
    const char* name;  // kernel suffix
    uint32_t ggml;     // ggml type id for dequantize_row
    int vals, bytes;   // values and bytes per block
};
const Type kTypes[] = {{"IQ2S", 22, 256, 82},   {"IQ3XXS", 18, 256, 98}, {"IQ3S", 21, 256, 110},
                       {"IQ4XS", 23, 256, 136}, {"IQ4NL", 20, 32, 18},   {"Q20", 42, 64, 18}};
enum { IQ2S, IQ3XXS, IQ3S, IQ4XS, IQ4NL, Q20 };

constexpr int kHidden = 2560, kFf = 640, kPool = 64, kSel = 10, kLayers = 48, kNsg = 4;
int g_rows = 1;  // rows per simdgroup (kernel variant r1 or r4)

// gate/up type and down type per layer of the pinned Coder file (GGUF header, revision 5348543e)
// (2 = IQ2_S, X = IQ3_XXS, S = IQ3_S, 4 = IQ4_XS; N = IQ4_NL, Q = Q2_0)
const char* kMix = "XN 2Q 2N 2N 2N 2N 2N XN 2Q 2N 2Q 2N XQ 2Q XN 2N XN SN XN 2N XQ SQ XN 2N "
                   "2N SN 2N SN SN SN XQ XN 2N XN 2N SN 2Q XN XN 2N SN XN XN XN SN SN XN 4N";

struct Params {
    uint32_t rows, k, row_bytes, n_sel, x_stride;
};

struct Mat {  // kPool experts of one matrix
    int type, rows, k;
    uint32_t row_bytes;
    id<MTLBuffer> buf;
};

id<MTLDevice> dev;
id<MTLCommandQueue> queue;
id<MTLLibrary> lib;

std::string slurp(const std::string& p) {
    std::ifstream f(p);
    if (!f) throw std::runtime_error("cannot read " + p);
    std::stringstream s;
    s << f.rdbuf();
    return s.str();
}

// quant_tables.h (C++) to MSL constants
std::string tables_msl(const std::string& root) {
    std::string t = slurp(root + "/engine/model/quant_tables.h");
    t = std::regex_replace(
        t, std::regex(R"(#pragma once|#include <cstdint>|namespace tona::tables \{|\}  // namespace.*)"), "");
    t = std::regex_replace(t, std::regex("inline constexpr uint64_t"), "constant ulong");
    t = std::regex_replace(t, std::regex("inline constexpr uint32_t"), "constant uint");
    t = std::regex_replace(t, std::regex("inline constexpr uint8_t"), "constant uchar");
    t = std::regex_replace(t, std::regex("inline constexpr int8_t"), "constant char");
    return "#include <metal_stdlib>\nusing namespace metal;\n" + t;
}

Mat make_mat(int type, int rows, int k, std::mt19937_64& rng) {
    const Type& T = kTypes[type];
    Mat m{type, rows, k, uint32_t(k / T.vals * T.bytes), nil};
    size_t n = size_t(kPool) * rows * m.row_bytes;
    m.buf = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    auto* p = static_cast<uint8_t*>(m.buf.contents);
    for (size_t i = 0; i + 8 <= n; i += 8) {
        uint64_t v = rng();
        std::memcpy(p + i, &v, 8);
    }
    const _Float16 d = 0.002f;  // block scale: keeps the sums in a sane range
    for (size_t b = 0; b < n / T.bytes; ++b) std::memcpy(p + b * T.bytes, &d, 2);
    return m;
}

id<MTLComputePipelineState> pipe(const std::string& name) {
    NSError* err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!f) throw std::runtime_error("no kernel " + name);
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) throw std::runtime_error(std::string("pipeline ") + name);
    return p;
}

id<MTLBuffer> buf_floats(size_t n) { return [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared]; }

// input, output and the expert ids from ids_off on
struct Io {
    id<MTLBuffer> x, out, ids;
    size_t ids_off;
};

// one gate_up (gu != nullptr) or down dispatch over kSel experts starting at ids[ids_off]
void encode(id<MTLComputeCommandEncoder> enc, const Mat& w, const Mat* up, int B, const Io& io) {
    id<MTLBuffer> x = io.x, out = io.out, ids = io.ids;
    const size_t ids_off = io.ids_off;
    std::string name = std::string(up ? "gate_up_" : "down_") + kTypes[w.type].name + "_" + std::to_string(B) + "_r" +
                       std::to_string(g_rows);
    static std::vector<std::pair<std::string, id<MTLComputePipelineState>>> cache;
    auto it = std::find_if(cache.begin(), cache.end(), [&](auto& e) { return e.first == name; });
    if (it == cache.end()) it = cache.insert(cache.end(), {name, pipe(name)});
    Params P{uint32_t(w.rows), uint32_t(w.k), w.row_bytes, kSel, up ? 0u : uint32_t(B * w.k)};
    [enc setComputePipelineState:it->second];
    [enc setBuffer:w.buf offset:0 atIndex:0];
    if (up) [enc setBuffer:up->buf offset:0 atIndex:1];
    [enc setBuffer:x offset:0 atIndex:2];
    [enc setBuffer:out offset:0 atIndex:3];
    [enc setBuffer:ids offset:ids_off * 4 atIndex:4];
    [enc setBytes:&P length:sizeof P atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(w.rows / (kNsg * g_rows), kSel, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * kNsg, 1, 1)];
}

double run_cb(void (^body)(id<MTLComputeCommandEncoder>)) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial];
    body(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(cb.error.description.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
}

// max |gpu - ref| / max |ref| of row r of expert e
double row_ref(const Mat& w, int e, int r, const float* x) {
    std::vector<float> row(w.k);
    tona::dequantize_row(kTypes[w.type].ggml,
                         static_cast<uint8_t*>(w.buf.contents) + (size_t(e) * w.rows + r) * w.row_bytes, row.data(),
                         w.k);
    double s = 0;
    for (int i = 0; i < w.k; ++i) s += double(row[i]) * x[i];
    return s;
}

// checks gate_up and down outputs of two experts against the CPU reference; returns the worst relative error
// b: input x, hidden h, output y, expert ids
double check(const Mat& g, const Mat& u, const Mat& dn, int B, const std::array<id<MTLBuffer>, 4>& b) {
    id<MTLBuffer> x = b[0], h = b[1], y = b[2], ids = b[3];
    uint32_t* idp = static_cast<uint32_t*>(ids.contents);
    for (int i = 0; i < kSel; ++i) idp[i] = uint32_t(i * 5 + 3);
    run_cb(^(id<MTLComputeCommandEncoder> enc) {
      encode(enc, g, &u, B, {x, h, ids, 0});
      encode(enc, dn, nullptr, B, {h, y, ids, 0});
    });
    const float* xp = static_cast<float*>(x.contents);
    const float* hp = static_cast<float*>(h.contents);
    const float* yp = static_cast<float*>(y.contents);
    double err = 0, mag = 0;
    for (int s : {0, kSel - 1})
        for (int b = 0; b < B; ++b) {
            for (int r = 0; r < kFf; r += 37) {
                double a = row_ref(g, idp[s], r, xp + b * kHidden), c = row_ref(u, idp[s], r, xp + b * kHidden);
                double ref = a / (1 + std::exp(-a)) * c;
                err = std::max(err, std::fabs(ref - hp[(s * B + b) * kFf + r])), mag = std::max(mag, std::fabs(ref));
            }
            for (int r = 0; r < kHidden; r += 131) {
                double ref = row_ref(dn, idp[s], r, hp + (s * B + b) * kFf);
                err = std::max(err, std::fabs(ref - yp[(s * B + b) * kHidden + r])),
                mag = std::max(mag, std::fabs(ref));
            }
        }
    return mag > 0 ? err / mag : 1;
}

void fill_ids(id<MTLBuffer> ids, int layers, std::mt19937_64& rng) {
    uint32_t* p = static_cast<uint32_t*>(ids.contents);
    std::vector<uint32_t> all(kPool);
    for (int i = 0; i < kPool; ++i) all[i] = uint32_t(i);
    for (int l = 0; l < layers; ++l) {
        std::shuffle(all.begin(), all.end(), rng);
        std::copy(all.begin(), all.begin() + kSel, p + l * kSel);
    }
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

struct Pool {
    Mat gate[4], up[4], down[2];  // gate/up: IQ2S, IQ3XXS, IQ3S, IQ4XS; down: IQ4NL, Q20
    id<MTLBuffer> x, h, y, ids;
};

int gate_slot(char c) { return c == '2' ? 0 : c == 'X' ? 1 : c == 'S' ? 2 : 3; }

void run_checks(Pool& P) {
    const int combos[][2] = {{0, 0}, {1, 0}, {2, 0}, {3, 0}, {0, 1}, {1, 1}};
    for (auto& c : combos)
        for (int B = 1; B <= 4; ++B) {
            double e = check(P.gate[c[0]], P.up[c[0]], P.down[c[1]], B, {P.x, P.h, P.y, P.ids});
            printf("{\"kind\": \"check\", \"rows\": %d, \"gate\": \"%s\", \"down\": \"%s\", \"B\": %d, \"rel_err\": "
                   "%.2e}\n",
                   g_rows, kTypes[P.gate[c[0]].type].name, kTypes[P.down[c[1]].type].name, B, e);
            if (!(e < 1e-3)) throw std::runtime_error("check failed");
        }
}

// per kernel type: 48 layers of 10 experts in one command buffer, 10 command buffers per point
void run_types(Pool& P, std::mt19937_64& rng, bool print) {
    for (int B = 1; B <= 4; ++B)
        for (int m = 0; m < 6; ++m) {
            bool gu = m < 4;
            const Mat& w = gu ? P.gate[m] : P.down[m - 4];
            std::vector<double> t;
            for (int rep = 0; rep < 10; ++rep) {
                fill_ids(P.ids, kLayers, rng);
                t.push_back(run_cb(^(id<MTLComputeCommandEncoder> enc) {
                              for (int l = 0; l < kLayers; ++l)
                                  encode(enc, w, gu ? &P.up[m] : nullptr, B,
                                         {gu ? P.x : P.h, gu ? P.h : P.y, P.ids, size_t(l) * kSel});
                            }) /
                            kLayers);
            }
            double bytes = double(kSel) * (gu ? 2 : 1) * w.rows * w.row_bytes;
            if (print)
                printf("{\"kind\": \"type\", \"rows\": %d, \"kernel\": \"%s_%s\", \"B\": %d, \"us_per_layer_median\": "
                       "%.1f, "
                       "\"us_min\": %.1f, \"us_max\": %.1f, \"gbs\": %.1f}\n",
                       g_rows, gu ? "gate_up" : "down", kTypes[w.type].name, B, median(t),
                       *std::min_element(t.begin(), t.end()), *std::max_element(t.begin(), t.end()),
                       bytes / median(t) / 1e3);
        }
}

// whole token: the Coder's 48 layers with their types, gate_up + down per layer, 20 command buffers per point
void run_token(Pool& P, std::mt19937_64& rng) {
    std::vector<std::pair<int, int>> mix;
    for (const char* c = kMix; *c; ++c)
        if (*c != ' ') mix.push_back({gate_slot(c[0]), c[1] == 'Q' ? 1 : 0}), ++c;
    double bytes = 0;
    for (auto [g, d] : mix) bytes += double(kSel) * (2.0 * kFf * P.gate[g].row_bytes + kHidden * P.down[d].row_bytes);
    for (int B = 1; B <= 4; ++B) {
        std::vector<double> t;
        for (int rep = 0; rep < 20; ++rep) {
            fill_ids(P.ids, kLayers, rng);
            t.push_back(run_cb(^(id<MTLComputeCommandEncoder> enc) {
              for (int l = 0; l < kLayers; ++l) {
                  encode(enc, P.gate[mix[l].first], &P.up[mix[l].first], B, {P.x, P.h, P.ids, size_t(l) * kSel});
                  encode(enc, P.down[mix[l].second], nullptr, B, {P.h, P.y, P.ids, size_t(l) * kSel});
              }
            }));
        }
        printf("{\"kind\": \"token\", \"rows\": %d, \"layers\": %zu, \"B\": %d, \"ms_median\": %.3f, \"ms_min\": %.3f, "
               "\"ms_max\": %.3f, \"expert_bytes\": %.0f, \"gbs\": %.1f}\n",
               g_rows, mix.size(), B, median(t) / 1e3, *std::min_element(t.begin(), t.end()) / 1e3,
               *std::max_element(t.begin(), t.end()) / 1e3, bytes, bytes / median(t) / 1e3);
    }
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string root = argc > 1 ? argv[1] : ".";
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        std::string src = tables_msl(root) + slurp(root + "/engine/backend/metal/kernels/decode.metal") +
                          slurp(root + "/bench/mac/expertk.metal");
        NSError* err = nil;
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "%s\n", err.description.UTF8String);
            return 1;
        }
        std::mt19937_64 rng(20261111);
        Pool P;
        for (int i = 0; i < 4; ++i)
            P.gate[i] = make_mat(i, kFf, kHidden, rng), P.up[i] = make_mat(i, kFf, kHidden, rng);
        P.down[0] = make_mat(IQ4NL, kHidden, kFf, rng), P.down[1] = make_mat(Q20, kHidden, kFf, rng);
        P.x = buf_floats(4 * kHidden), P.h = buf_floats(kSel * 4 * kFf), P.y = buf_floats(kSel * 4 * kHidden);
        P.ids = [dev newBufferWithLength:kLayers * kSel * 4 options:MTLResourceStorageModeShared];
        std::normal_distribution<float> nd(0.f, 1.f);
        for (int i = 0; i < 4 * kHidden; ++i) static_cast<float*>(P.x.contents)[i] = nd(rng);
        try {
            for (int rows : {1, 4}) {
                g_rows = rows;
                run_checks(P);
                run_types(P, rng, false);  // warm-up: pipelines, clocks
                run_types(P, rng, true);
                run_token(P, rng);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
    return 0;
}
