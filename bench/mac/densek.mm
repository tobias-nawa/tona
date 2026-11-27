// Time of the dense part of one Coder pass on the M1 Max GPU at B = 1 to 4 tokens: every non-expert weight matrix
// of the pinned Coder file (737 mat-vecs in file order, real types and shapes, synthetic blocks) and the DeltaNet
// state step of the 36 DeltaNet layers. Outputs are checked against engine/model/quant.cpp (mat-vecs) and the
// recurrence of engine/cpu/ref.cpp (state step) before timing.
// Plan and evaluation: research/measurements/2026-11-11-mac-densek.md
// Build: clang++ -std=c++20 -O2 -fobjc-arc -Iengine bench/mac/densek.mm engine/model/quant.cpp
//        -framework Metal -framework Foundation -o work/build/mac/densek
// Usage: densek [repo root, default .] [tensor list, default work/gguf-header/coder-dense.tsv]
//        (tensor list: name, ggml type, K, N per line, from the GGUF header of the pinned Coder file)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <random>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "model/quant.h"

namespace {

struct Type {
    const char* name;
    int vals, bytes;
    std::vector<int> scale_offsets;  // fp16 scales per block that need sane values
};
const std::map<uint32_t, Type> kTypes = {{12, {"Q4K", 256, 144, {0, 2}}}, {13, {"Q5K", 256, 176, {0, 2}}},
                                         {14, {"Q6K", 256, 210, {208}}},  {8, {"Q80", 32, 34, {0}}},
                                         {23, {"IQ4XS", 256, 136, {0}}},  {20, {"IQ4NL", 32, 18, {0}}},
                                         {30, {"BF16", 1, 2, {}}}};

constexpr int kNsg = 4, kNk = 16, kNv = 48, kDnLayers = 36, kHead = 128;

struct Tensor {
    std::string name;
    uint32_t type;
    int k, n;
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

id<MTLComputePipelineState> pipe(const std::string& name) {
    static std::map<std::string, id<MTLComputePipelineState>> cache;
    auto it = cache.find(name);
    if (it != cache.end()) return it->second;
    NSError* err = nil;
    id<MTLFunction> f = [lib newFunctionWithName:[NSString stringWithUTF8String:name.c_str()]];
    if (!f) throw std::runtime_error("no kernel " + name);
    id<MTLComputePipelineState> p = [dev newComputePipelineStateWithFunction:f error:&err];
    if (!p) throw std::runtime_error("pipeline " + name);
    return cache[name] = p;
}

id<MTLBuffer> buf_floats(size_t n) { return [dev newBufferWithLength:n * 4 options:MTLResourceStorageModeShared]; }

double run_cb(void (^body)(id<MTLComputeCommandEncoder>), bool concurrent = false) {
    id<MTLCommandBuffer> cb = [queue commandBuffer];
    id<MTLComputeCommandEncoder> enc =
        [cb computeCommandEncoderWithDispatchType:concurrent ? MTLDispatchTypeConcurrent : MTLDispatchTypeSerial];
    body(enc);
    [enc endEncoding];
    [cb commit];
    [cb waitUntilCompleted];
    if (cb.error) throw std::runtime_error(cb.error.description.UTF8String);
    return (cb.GPUEndTime - cb.GPUStartTime) * 1e6;
}

double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}

// random blocks; fp16 scales 0.002 (min 0.001); BF16 values of magnitude 2^-7 to 2^-3
void fill(Tensor& t, std::mt19937_64& rng) {
    const Type& T = kTypes.at(t.type);
    size_t n = size_t(t.n) * t.row_bytes;
    t.buf = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
    auto* p = static_cast<uint8_t*>(t.buf.contents);
    for (size_t i = 0; i + 8 <= n; i += 8) {
        uint64_t v = rng();
        std::memcpy(p + i, &v, 8);
    }
    if (t.type == 30) {
        auto* h = reinterpret_cast<uint16_t*>(p);
        for (size_t i = 0; i < n / 2; ++i) h[i] = uint16_t((h[i] & 0x807f) | ((120 + (h[i] >> 8) % 5) << 7));
        return;
    }
    const _Float16 d = 0.002f, m = 0.001f;
    for (size_t b = 0; b < n / T.bytes; ++b)
        for (size_t s = 0; s < T.scale_offsets.size(); ++s)
            std::memcpy(p + b * T.bytes + T.scale_offsets[s], s == 0 ? &d : &m, 2);
}

std::vector<Tensor> load_list(const std::string& path, std::mt19937_64& rng) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot read " + path);
    std::vector<Tensor> out;
    std::string name;
    uint32_t type;
    int k, n;
    while (f >> name >> type >> k >> n) {
        if (!kTypes.count(type)) throw std::runtime_error("no kernel for type of " + name);
        const Type& T = kTypes.at(type);
        out.push_back({name, type, k, n, uint32_t(k / T.vals * T.bytes), nil});
        fill(out.back(), rng);
    }
    return out;
}

int g_split_below = 0;  // matrices with fewer rows use the split-K kernel (0: never)

int g_rows = 4;  // rows per simdgroup of the wide kernel (densek ROWS=... for experiments)

int g_nsg = kNsg;
bool g_xtg = false;
int g_q6_loads = 0;  // densek Q6LOADS=1: Q6_K timed with loads only, 2: aligned loads only (diagnostic, no checks)  //
                     // densek XTG=1: x staged in threadgroup memory where it fits  // simdgroups per threadgroup of the
                     // wide kernel (densek NSG=... for experiments)

int rows_per_sg(const Tensor& t) { return t.n % (g_nsg * g_rows) == 0 ? g_rows : t.n % (g_nsg * 4) == 0 ? 4 : 1; }

void encode_mv(id<MTLComputeCommandEncoder> enc, const Tensor& t, int B, id<MTLBuffer> x, id<MTLBuffer> y,
               NSUInteger y_off = 0) {
    bool split = t.n < g_split_below;
    bool staged = !split && g_xtg && B * t.k <= 8192;  // x in threadgroup memory (32 KB)
    int R = split ? (t.n % 4 == 0 ? 4 : 1) : rows_per_sg(t);
    std::string tn =
        t.type == 14 && g_q6_loads && !split && !staged ? (g_q6_loads == 2 ? "Q6KA" : "Q6KL") : kTypes.at(t.type).name;
    std::string name = std::string(split    ? "mvs_"
                                   : staged ? "mvt_"
                                            : "mv_") +
                       tn + "_" + std::to_string(B) + "_r" + std::to_string(R);
    uint32_t P[3] = {uint32_t(t.n), uint32_t(t.k), t.row_bytes};
    [enc setComputePipelineState:pipe(name)];
    [enc setBuffer:t.buf offset:0 atIndex:0];
    [enc setBuffer:x offset:0 atIndex:1];
    [enc setBuffer:y offset:y_off atIndex:2];
    [enc setBytes:P length:sizeof P atIndex:3];
    if (split) {
        [enc dispatchThreadgroups:MTLSizeMake(t.n / R, 1, 1) threadsPerThreadgroup:MTLSizeMake(32 * 8, 1, 1)];
        return;
    }
    if (staged) [enc setThreadgroupMemoryLength:NSUInteger(B) * t.k * 4 atIndex:0];
    int per_tg = g_nsg * R;
    [enc dispatchThreadgroups:MTLSizeMake((t.n + per_tg - 1) / per_tg, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * g_nsg, 1, 1)];
}

// worst |gpu - ref| / max |ref| over sampled rows of every type's first tensor, B = 1 to 4
void check_mv(const std::vector<Tensor>& ts, id<MTLBuffer> x, id<MTLBuffer> y) {
    std::map<uint32_t, const Tensor*> first;
    for (const Tensor& t : ts) first.emplace(t.type, &t);
    const float* xp = static_cast<float*>(x.contents);
    for (auto& entry : first)
        for (int B = 1; B <= 4; ++B) {
            const uint32_t type = entry.first;
            const Tensor* t = entry.second;
            run_cb(^(id<MTLComputeCommandEncoder> enc) {
              encode_mv(enc, *t, B, x, y);
            });
            const float* yp = static_cast<float*>(y.contents);
            std::vector<float> row(t->k);
            double err = 0, mag = 0;
            for (int r = 0; r < t->n; r += std::max(1, t->n / 61)) {
                tona::dequantize_row(type, static_cast<uint8_t*>(t->buf.contents) + size_t(r) * t->row_bytes,
                                     row.data(), t->k);
                for (int b = 0; b < B; ++b) {
                    double s = 0;
                    for (int i = 0; i < t->k; ++i) s += double(row[i]) * xp[b * t->k + i];
                    err = std::max(err, std::fabs(s - yp[b * t->n + r])), mag = std::max(mag, std::fabs(s));
                }
            }
            double rel = mag > 0 ? err / mag : 1;
            printf("{\"kind\": \"check\", \"split_below\": %d, \"what\": \"mv\", \"tensor\": \"%s\", \"type\": \"%s\", "
                   "\"B\": %d, "
                   "\"rel_err\": %.2e}\n",
                   g_split_below, t->name.c_str(), kTypes.at(type).name, B, rel);
            if (!(rel < 1e-3)) throw std::runtime_error("mv check failed");
        }
}

struct Dn {
    id<MTLBuffer> state, q, k, v, alpha, beta, o;
};

void encode_dn(id<MTLComputeCommandEncoder> enc, const Dn& d, int layer, int B) {
    struct {
        uint32_t nk, nv;
        float qscale;
    } P{kNk, kNv, 1.f / std::sqrt(float(kHead))};
    [enc setComputePipelineState:pipe("dn_step_" + std::to_string(B))];
    [enc setBuffer:d.state offset:size_t(layer) * kNv * kHead * kHead * 4 atIndex:0];
    id<MTLBuffer> in[] = {d.q, d.k, d.v, d.alpha, d.beta, d.o};
    for (int i = 0; i < 6; ++i) [enc setBuffer:in[i] offset:0 atIndex:i + 1];
    [enc setBytes:&P length:sizeof P atIndex:7];
    [enc dispatchThreadgroups:MTLSizeMake(kHead / (kNsg * 8), kNv, 1)
        threadsPerThreadgroup:MTLSizeMake(32 * kNsg, 1, 1)];
}

Dn make_dn(std::mt19937_64& rng) {
    Dn d{buf_floats(size_t(kDnLayers) * kNv * kHead * kHead),
         buf_floats(4 * kNk * kHead),
         buf_floats(4 * kNk * kHead),
         buf_floats(4 * kNv * kHead),
         buf_floats(4 * kNv),
         buf_floats(4 * kNv),
         buf_floats(4 * kNv * kHead)};
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.f, 1.f);
    auto put = [&](id<MTLBuffer> b, size_t n, float scale, bool unit) {
        for (size_t i = 0; i < n; ++i) static_cast<float*>(b.contents)[i] = unit ? ud(rng) : nd(rng) * scale;
    };
    put(d.state, size_t(kDnLayers) * kNv * kHead * kHead, 0.05f, false);
    put(d.q, 4 * kNk * kHead, 0.09f, false), put(d.k, 4 * kNk * kHead, 0.09f, false);  // ~unit L2 norm per head
    put(d.v, 4 * kNv * kHead, 1.f, false), put(d.alpha, 4 * kNv, 1.f, true), put(d.beta, 4 * kNv, 1.f, true);
    return d;
}

// layer 0, B tokens against the loops of ref.cpp Forward::deltanet (float state, double sums)
void check_dn(Dn& d) {
    const size_t ns = size_t(kNv) * kHead * kHead;
    auto* st = static_cast<float*>(d.state.contents);
    std::vector<float> saved(st, st + ns);
    for (int B = 1; B <= 4; ++B) {
        std::vector<float> ref(saved), o(size_t(B) * kNv * kHead);
        const float *q = static_cast<float*>(d.q.contents), *k = static_cast<float*>(d.k.contents);
        const float *v = static_cast<float*>(d.v.contents), *al = static_cast<float*>(d.alpha.contents);
        const float* be = static_cast<float*>(d.beta.contents);
        for (int b = 0; b < B; ++b)
            for (int j = 0; j < kNv; ++j) {
                const float *kj = k + (b * kNk + j % kNk) * kHead, *qj = q + (b * kNk + j % kNk) * kHead;
                float* s = ref.data() + size_t(j) * kHead * kHead;
                for (int i = 0; i < kHead * kHead; ++i) s[i] *= al[b * kNv + j];
                for (int vi = 0; vi < kHead; ++vi) {
                    float* row = s + vi * kHead;
                    double sk = 0, ov = 0;
                    for (int ki = 0; ki < kHead; ++ki) sk += double(row[ki]) * kj[ki];
                    const float dl = (v[(b * kNv + j) * kHead + vi] - float(sk)) * be[b * kNv + j];
                    for (int ki = 0; ki < kHead; ++ki) row[ki] += kj[ki] * dl, ov += double(row[ki]) * qj[ki];
                    o[(size_t(b) * kNv + j) * kHead + vi] = float(ov) / std::sqrt(float(kHead));
                }
            }
        std::copy(saved.begin(), saved.end(), st);
        run_cb(^(id<MTLComputeCommandEncoder> enc) {
          encode_dn(enc, d, 0, B);
        });
        const float* og = static_cast<float*>(d.o.contents);
        double err = 0, mag = 0;
        for (size_t i = 0; i < o.size(); ++i)
            err = std::max(err, double(std::fabs(o[i] - og[i]))), mag = std::max(mag, double(std::fabs(o[i])));
        for (size_t i = 0; i < ns; ++i) err = std::max(err, double(std::fabs(ref[i] - st[i])));
        printf("{\"kind\": \"check\", \"what\": \"dn_step\", \"B\": %d, \"rel_err\": %.2e}\n", B, err / mag);
        if (!(err / mag < 1e-3)) throw std::runtime_error("dn check failed");
    }
    std::copy(saved.begin(), saved.end(), st);
}

void time_all(const std::vector<Tensor>& ts, Dn& d, id<MTLBuffer> x, id<MTLBuffer> y) {
    double bytes = 0;
    for (const Tensor& t : ts) bytes += double(t.n) * t.row_bytes;
    for (int B = 1; B <= 4; ++B) {
        std::vector<double> mv, dn;
        for (int rep = 0; rep < 20; ++rep) {
            mv.push_back(run_cb(^(id<MTLComputeCommandEncoder> enc) {
              for (const Tensor& t : ts) encode_mv(enc, t, B, x, y);
            }));
            dn.push_back(run_cb(^(id<MTLComputeCommandEncoder> enc) {
              for (int l = 0; l < kDnLayers; ++l) encode_dn(enc, d, l, B);
            }));
        }
        printf("{\"kind\": \"token\", \"split_below\": %d, \"B\": %d, \"mv_ms_median\": %.3f, \"mv_ms_min\": %.3f, "
               "\"mv_ms_max\": %.3f, "
               "\"mv_bytes\": %.0f, \"mv_gbs\": %.1f, \"n_mv\": %zu, \"dn_ms_median\": %.3f, \"dn_ms_min\": %.3f, "
               "\"dn_ms_max\": %.3f}\n",
               g_split_below, B, median(mv) / 1e3, *std::min_element(mv.begin(), mv.end()) / 1e3,
               *std::max_element(mv.begin(), mv.end()) / 1e3, bytes, bytes / median(mv) / 1e3, ts.size(),
               median(dn) / 1e3, *std::min_element(dn.begin(), dn.end()) / 1e3,
               *std::max_element(dn.begin(), dn.end()) / 1e3);
    }
}

// Steps of a pass in the order of the reference's program (engine/gpu/mega_build_layer.cpp): matrices that read the
// same input form one step; key (layer, step), the head after the last layer
std::pair<int, int> phase_of(const std::string& name) {
    static const std::vector<std::pair<std::string, int>> steps = {
        {"hc_attn_down", 0},   {"hc_attn_inject", 0}, {"hc_attn_up", 1},   {"attn_qkv", 2},       {"attn_gate", 2},
        {"ssm_alpha", 2},      {"ssm_beta", 2},       {"attn_q.", 2},      {"attn_k.", 2},        {"attn_v.", 2},
        {"indexer.k_proj", 2}, {"indexer.q_proj", 2}, {"ssm_out", 3},      {"attn_output", 3},    {"hc_ffn_down", 4},
        {"hc_ffn_inject", 4},  {"hc_ffn_up", 5},      {"ffn_gate_inp", 6}, {"ffn_gate_shexp", 6}, {"ffn_up_shexp", 6},
        {"ffn_down_shexp", 7}, {"output_hc_down", 0}, {"output_hc_up", 1}, {"output.", 2},        {"ple_", 3}};
    std::smatch m;
    int layer = std::regex_search(name, m, std::regex(R"(^blk\.(\d+)\.)")) ? std::stoi(m[1]) : 1000;
    for (const auto& [key, step] : steps)
        if (name.find(key) != std::string::npos) return {layer, step};
    throw std::runtime_error("no step for " + name);
}

// All mat-vecs of a token in steps: concurrent dispatches within a step, a buffer barrier between steps
void time_phased(const std::vector<Tensor>& ts, id<MTLBuffer> x, id<MTLBuffer> y) {
    std::map<std::pair<int, int>, std::vector<const Tensor*>> steps;
    for (const Tensor& t : ts) steps[phase_of(t.name)].push_back(&t);
    double bytes = 0;
    for (const Tensor& t : ts) bytes += double(t.n) * t.row_bytes;
    for (int B = 1; B <= 4; ++B) {
        std::vector<double> mv;
        for (int rep = 0; rep < 20; ++rep)
            mv.push_back(run_cb(
                ^(id<MTLComputeCommandEncoder> enc) {
                  for (const auto& entry : steps) {
                      NSUInteger off = 0;  // own output range per matrix: no write overlap inside a step
                      for (const Tensor* t : entry.second) {
                          encode_mv(enc, *t, B, x, y, off);
                          off += (NSUInteger(B) * t->n * 4 + 255) / 256 * 256;
                      }
                      [enc memoryBarrierWithScope:MTLBarrierScopeBuffers];
                  }
                },
                true));
        printf("{\"kind\": \"phased\", \"split_below\": %d, \"B\": %d, \"steps\": %zu, \"mv_ms_median\": %.3f, "
               "\"mv_ms_min\": %.3f, \"mv_ms_max\": %.3f, \"mv_gbs\": %.1f}\n",
               g_split_below, B, steps.size(), median(mv) / 1e3, *std::min_element(mv.begin(), mv.end()) / 1e3,
               *std::max_element(mv.begin(), mv.end()) / 1e3, bytes / median(mv) / 1e3);
    }
}

// per role (name without block number) at B = 1: all its matrices in one command buffer, median of 10
void time_roles(const std::vector<Tensor>& ts, id<MTLBuffer> x, id<MTLBuffer> y) {
    std::map<std::string, std::vector<const Tensor*>> roles;
    for (const Tensor& t : ts) roles[std::regex_replace(t.name, std::regex(R"(^blk\.\d+\.)"), "")].push_back(&t);
    for (auto& entry : roles) {
        const std::string& role = entry.first;
        const std::vector<const Tensor*>& list = entry.second;
        std::vector<double> t;
        double bytes = 0;
        for (const Tensor* p : list) bytes += double(p->n) * p->row_bytes;
        for (int rep = 0; rep < 10; ++rep)
            t.push_back(run_cb(^(id<MTLComputeCommandEncoder> enc) {
              for (const Tensor* p : list) encode_mv(enc, *p, 1, x, y);
            }));
        printf("{\"kind\": \"role\", \"split_below\": %d, \"role\": \"%s\", \"count\": %zu, \"B\": 1, \"us_median\": "
               "%.1f, \"gbs\": %.1f}\n",
               g_split_below, role.c_str(), list.size(), median(t), bytes / median(t) / 1e3);
    }
}

}  // namespace

int main(int argc, char** argv) {
    @autoreleasepool {
        std::string root = argc > 1 ? argv[1] : ".";
        std::string list = argc > 2 ? argv[2] : root + "/work/gguf-header/coder-dense.tsv";
        if (const char* r = getenv("ROWS")) g_rows = atoi(r);
        if (const char* r = getenv("NSG")) g_nsg = atoi(r);
        g_xtg = getenv("XTG") && atoi(getenv("XTG")) == 1;
        g_q6_loads = getenv("Q6LOADS") ? atoi(getenv("Q6LOADS")) : 0;
        dev = MTLCreateSystemDefaultDevice();
        queue = [dev newCommandQueue];
        std::string src = tables_msl(root) + slurp(root + "/engine/backend/metal/kernels/decode.metal") +
                          slurp(root + "/bench/mac/densek.metal");
        NSError* err = nil;
        lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:nil error:&err];
        if (!lib) {
            fprintf(stderr, "%s\n", err.description.UTF8String);
            return 1;
        }
        try {
            std::mt19937_64 rng(20261111);
            std::vector<Tensor> ts = load_list(list, rng);
            id<MTLBuffer> x = buf_floats(4 * 10240), y = buf_floats(4 * 248320);
            std::normal_distribution<float> nd(0.f, 1.f);
            for (int i = 0; i < 4 * 10240; ++i) static_cast<float*>(x.contents)[i] = nd(rng);
            Dn d = make_dn(rng);
            check_dn(d);
            for (int split : {0, 2048}) {  // first-cut layout, then split-K below 2048 rows
                g_split_below = split;
                if (!g_q6_loads) check_mv(ts, x, y);
                time_roles(ts, x, y);  // also the warm-up of all pipelines
                time_all(ts, d, x, y);
                time_phased(ts, x, y);
            }
        } catch (const std::exception& e) {
            fprintf(stderr, "%s\n", e.what());
            return 1;
        }
    }
    return 0;
}
