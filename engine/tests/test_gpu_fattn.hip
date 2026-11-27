// Full-context attention (gpu/mega_op_fattn.h, gpu/prefill_fattn.hip) against the CPU primitives of the oracle
// (cpu/ref_ops.h: rms_norm, rope, attend_cells) on a random int8 cache, at the real layer dimensions of Ornith (hd 256,
// 16 q heads, 2 KV heads, gate, QK norm, RoPE) and Bolt (hd 128, 32 q heads, 2 KV heads, none of them). The decode
// steps run inside the real k_mega with a test-only profile (tests/fattn_rig.h); positions up to 262,144 (the partial
// buffer is sized for that and breaks only there). The int8 cache, f16 tiles and float sums are not the oracle's
// arithmetic: the tolerance is stated in the output. Also: same bits with another grid size and with one token per
// pass instead of several. Needs the GPU, no model.
// Usage: test_gpu_fattn [--bench | --prefill-bench | --prefill-cmp | --stress N]   (--bench: time of the steps against
// the bytes of the cache and the QSA path; --prefill-bench: the prefill kernels in TFLOPS; --prefill-cmp: the prefill
// kernels bit for bit; --stress: the same pass N times, bit comparison and NaN scan, tests/fattn_stress.h)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "cpu/ref_ops.h"
#include "fattn_rig.h"
#include "gpu/prefill_fattn.h"
#include "gpu/prefill_kernels.h"
#include "gpu/probe/probe_kernels.h"

namespace {

using namespace rig;
using tona::gpu::MegaOp;

constexpr float EPS = 1e-6f, BASE = 1e7f;
constexpr int64_t LONG_CTX = 262144;
// Ornith: gate, QK norm, RoPE. Bolt: none of them. Bolt with all three at the small head size.
using gpu::probe::kBolt, gpu::probe::kBoltAll, gpu::probe::kOrnith;
constexpr int FA_SPL = gpu::probe::kFattnSplits;

struct Shape {
    gpu::probe::FattnShape cfg;
    int nh, nkv;
    int os() const { return nh * cfg.hd; }
    int row() const { return 2 * nkv * cfg.hd + nkv * cfg.hd / 16 * 4; }
    int ns() const { return cfg.hd / 32; }
};

std::vector<float> randoms(size_t n, uint32_t seed, float sd, float mean = 0.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> d(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = d(rng);
    return v;
}

// Raw inputs of up to `tokens` tokens: attn_q ([q, gate] per head with gate), k, v and the two norm weights
struct Inputs {
    std::vector<float> qfull, k, v, qw, kw;
    Inputs(const Shape& s, int tokens, uint32_t seed)
        : qfull(randoms((size_t)tokens * s.nh * (s.cfg.gate ? 2 : 1) * s.cfg.hd, seed, 2.0f)),
          k(randoms((size_t)tokens * s.nkv * s.cfg.hd, seed + 1, 2.0f)),
          v(randoms((size_t)tokens * s.nkv * s.cfg.hd, seed + 2, 1.0f)),
          qw(randoms(s.cfg.hd, seed + 3, 0.2f, 1.0f)),
          kw(randoms(s.cfg.hd, seed + 4, 0.2f, 1.0f)) {}
};

struct Outputs {
    std::vector<float> qn, o;
    std::vector<int8_t> q8;
    std::vector<gpu::Q8Meta> meta;
    bool ok = false;
};

// A device buffer holding a copy of a vector
class Dev : public DeviceBuffer {
public:
    template <class T>
    explicit Dev(const std::vector<T>& v) : DeviceBuffer(std::max<size_t>(v.size() * sizeof(T), 16)) {
        CHECK(hipMemcpy(get(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    }
};

template <class T>
std::vector<T> download(const DeviceBuffer& b, size_t n) {
    std::vector<T> v(n);
    CHECK(hipMemcpy(v.data(), b.get(), n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

// Device buffers and the three steps (prepare, split, merge) of a pass for tokens [t0, t0 + nb) of the inputs
struct DecodeProgram {
    Dev qfull, tk, tv, qw, kw;
    DeviceBuffer qn, g, part, o, q8, meta;
    MegaOp prep, split, merge;

    DecodeProgram(const Shape& s, const RandomCache& cache, const Inputs& in, int t0, int nb)
        : qfull(sub(in.qfull, (size_t)s.nh * (s.cfg.gate ? 2 : 1) * s.cfg.hd, t0, nb)),
          tk(sub(in.k, (size_t)s.nkv * s.cfg.hd, t0, nb)),
          tv(sub(in.v, (size_t)s.nkv * s.cfg.hd, t0, nb)),
          qw(in.qw),
          kw(in.kw),
          qn((size_t)nb * s.os() * 4),
          g((size_t)nb * s.os() * 4),
          part((size_t)nb * s.nh * FA_SPL * (s.cfg.hd + 2) * 4),
          o((size_t)nb * s.os() * 4),
          q8((size_t)nb * s.os()),
          meta((size_t)nb * s.os() / 32 * sizeof(gpu::Q8Meta)) {
        prep.kind = gpu::OP_FATTN_PREP, split.kind = gpu::OP_FATTN_SPLIT, merge.kind = gpu::OP_FATTN_MERGE;
        prep.i[0] = split.i[0] = merge.i[0] = s.nh, prep.i[1] = split.i[1] = s.nkv;
        prep.i[6] = split.i[4] = s.row(), merge.i[3] = s.os();
        prep.f[0] = EPS, prep.f[1] = BASE, split.f[0] = 1.0f / std::sqrt((float)s.cfg.hd);
        prep.p[0] = qfull.get(), prep.p[1] = qn.get(), prep.p[2] = g.get(), prep.p[3] = qw.get(), prep.p[4] = tk.get();
        prep.p[5] = kw.get(), prep.p[11] = cache.get(), prep.p[12] = tv.get();
        split.p[0] = qn.get(), split.p[1] = cache.get(), split.p[2] = part.get();
        merge.p[2] = part.get(), merge.p[4] = g.get(), merge.p[5] = o.get(), merge.p[6] = q8.get();
        merge.p[7] = meta.get();
    }

private:
    static std::vector<float> sub(const std::vector<float>& v, size_t per, int t, int n) {
        return std::vector<float>(v.begin() + (size_t)t * per, v.begin() + (size_t)(t + n) * per);
    }
};

gpu::MegaParams pass_params(int64_t pos0, int nb) {
    gpu::MegaParams p{};
    p.pos = pos0, p.nb = nb, p.commit = nb;
    return p;
}

// A pass: tokens [t0, t0 + nb) of the inputs at positions pos0 ..., on `blocks` blocks (0: all)
struct PassSpec {
    int64_t pos0;
    int t0, nb, blocks;
};

Outputs run_pass(Rig& rig, const Shape& s, RandomCache& cache, const Inputs& in, const PassSpec& ps) {
    const int nb = ps.nb;
    const DecodeProgram d(s, cache, in, ps.t0, nb);
    Outputs out;
    out.ok = rig.run({d.prep, d.split, d.merge}, pass_params(ps.pos0, nb), ps.blocks);
    const size_t n = (size_t)nb * s.os();
    out.qn = download<float>(d.qn, n), out.o = download<float>(d.o, n);
    out.q8 = download<int8_t>(d.q8, n), out.meta = download<gpu::Q8Meta>(d.meta, n / 32);
    return out;
}

// The cache on the host, kept in step with the device (the rows a pass writes are fetched after it)
struct HostCache {
    std::vector<uint8_t> bytes;
    const Shape& s;
    HostCache(const Shape& sh, const RandomCache& c) : bytes((size_t)c.rows() * c.row()), s(sh) {
        CHECK(hipMemcpy(bytes.data(), c.get(), bytes.size(), hipMemcpyDeviceToHost));
    }
    void fetch(const RandomCache& c, int64_t pos0, int nb) {
        CHECK(hipMemcpy(&bytes[pos0 * s.row()], c.get() + pos0 * s.row(), (size_t)nb * s.row(), hipMemcpyDeviceToHost));
    }
    // Scale of K (v = false) or V of position pos, head h for dimension i
    float scale(int64_t pos, int h, bool v, int i) const {
        const uint8_t* r = &bytes[pos * s.row()];
        return ((const float*)(r + (int64_t)s.nkv * 2 * s.cfg.hd))[h * 2 * s.ns() + (v ? s.ns() : 0) + i / 32];
    }
    // Dequantized K or V of position pos, head h, dimension i
    float at(int64_t pos, int h, bool v, int i) const {
        const int8_t q = ((const int8_t*)&bytes[pos * s.row()])[(int64_t)h * 2 * s.cfg.hd + (v ? s.cfg.hd : 0) + i];
        return q * scale(pos, h, v, i);
    }
};

// Largest difference in units of the largest magnitude of the reference
double rel_err(const std::vector<float>& got, const std::vector<float>& want) {
    double mx = 1e-12, err = 0;
    for (float w : want) mx = std::max(mx, (double)std::fabs(w));
    for (size_t i = 0; i < want.size(); i++) err = std::max(err, std::fabs((double)got[i] - want[i]));
    return err / mx;
}

// q of token t after the oracle's norm and RoPE, all heads
std::vector<float> ref_q(const Shape& s, const Inputs& in, int t, int64_t pos) {
    const int hd = s.cfg.hd, stride = (s.cfg.gate ? 2 : 1) * hd;
    std::vector<float> q((size_t)s.nh * hd);
    for (int h = 0; h < s.nh; h++) {
        float* x = &q[(size_t)h * hd];
        std::copy_n(&in.qfull[((size_t)t * s.nh + h) * stride], hd, x);
        if (s.cfg.qk_norm) tona::ref::rms_norm(x, hd, EPS, in.qw.data());
        if (s.cfg.rope) tona::ref::rope(x, 64, pos, BASE);
    }
    return q;
}

// The K and V rows the pass wrote for token t: distance of the dequantized int8 from the oracle's norm and RoPE (K) or
// the input (V), in quantization steps. At most 0.5 plus the rounding of the oracle's float math.
double ref_kv_err(const Shape& s, const Inputs& in, const HostCache& hc, int t, int64_t pos) {
    const int hd = s.cfg.hd;
    double worst = 0;
    for (int h = 0; h < s.nkv; h++) {
        const size_t at = ((size_t)t * s.nkv + h) * hd;
        std::vector<float> k(in.k.begin() + at, in.k.begin() + at + hd);
        if (s.cfg.qk_norm) tona::ref::rms_norm(k.data(), hd, EPS, in.kw.data());
        if (s.cfg.rope) tona::ref::rope(k.data(), 64, pos, BASE);
        for (int i = 0; i < hd; i++) {
            worst = std::max(worst, (double)std::fabs(hc.at(pos, h, false, i) - k[i]) / hc.scale(pos, h, false, i));
            worst =
                std::max(worst, (double)std::fabs(hc.at(pos, h, true, i) - in.v[at + i]) / hc.scale(pos, h, true, i));
        }
    }
    return worst;
}

// Attention of every token of the pass in the oracle: K and V of positions 0 .. pos0 + nb - 1 from the cache
std::vector<float> ref_attention(const Shape& s, const Inputs& in, const HostCache& hc, int64_t pos0, int nb) {
    const int hd = s.cfg.hd;
    const int64_t n = pos0 + nb;
    std::vector<float> K((size_t)n * s.nkv * hd), V(K.size());
    for (int64_t p = 0; p < n; p++)
        for (int h = 0; h < s.nkv; h++)
            for (int i = 0; i < hd; i++)
                K[(p * s.nkv + h) * hd + i] = hc.at(p, h, false, i), V[(p * s.nkv + h) * hd + i] = hc.at(p, h, true, i);
    std::vector<float> out;
    std::vector<int64_t> cells;
    for (int b = 0; b < nb; b++) {
        cells.resize((size_t)(pos0 + b + 1));
        for (size_t c = 0; c < cells.size(); c++) cells[c] = (int64_t)c;
        const auto q = ref_q(s, in, b, pos0 + b);
        auto o = tona::ref::attend_cells({s.nh, s.nkv, hd}, q.data(), K, V, cells);
        if (s.cfg.gate)
            for (int h = 0; h < s.nh; h++)
                for (int i = 0; i < hd; i++)
                    o[(size_t)h * hd + i] *= tona::ref::sigmoid(in.qfull[((size_t)b * s.nh + h) * 2 * hd + hd + i]);
        out.insert(out.end(), o.begin(), o.end());
    }
    return out;
}

// Tolerances. o: largest error in units of the largest output; the K, V and Q tiles and P are f16 (2^-11 relative
// each), the cache int8. q: the oracle computes the norm in double and the angle in double; the kernel uses rsqrtf.
constexpr double TOL_O = 5e-3, TOL_Q = 2e-5, TOL_KV = 0.51, TOL_Q8 = 0.51;

// The int8 output of the merge against its float output, in quantization steps of its block of 32
double q8_err(const Outputs& a) {
    double worst = 0;
    for (size_t i = 0; i < a.o.size(); i++) {
        const float d = a.meta[i / 32].d;
        worst = std::max(worst, d > 0 ? std::fabs(a.q8[i] * d - a.o[i]) / d : 0.0);
    }
    return worst;
}

// Same bits on another grid size, and one token per pass instead of nb tokens in one
bool same_bits(Rig& rig, const Shape& s, RandomCache& cache, const Inputs& in, int64_t pos0, const Outputs& a) {
    const int nb = (int)(a.o.size() / s.os());
    const Outputs few = run_pass(rig, s, cache, in, {pos0, 0, nb, 5});
    bool same = few.ok && few.o == a.o && few.q8 == a.q8;
    for (int t = 0; t < nb && nb > 1; t++) {
        const Outputs one = run_pass(rig, s, cache, in, {pos0 + t, t, 1, 0});
        same = same && one.ok && std::equal(one.o.begin(), one.o.end(), a.o.begin() + (size_t)t * s.os());
    }
    return same;
}

bool check_decode(Rig& rig, const Shape& s, RandomCache& cache, HostCache& hc, int64_t pos0, int nb) {
    const Inputs in(s, nb, 7 + (uint32_t)pos0);
    const Outputs a = run_pass(rig, s, cache, in, {pos0, 0, nb, 0});
    hc.fetch(cache, pos0, nb);
    double eq = 0, ekv = 0;
    for (int t = 0; t < nb; t++) {
        const auto q = ref_q(s, in, t, pos0 + t);
        const auto first = a.qn.begin() + (ptrdiff_t)t * s.os();
        eq = std::max(eq, rel_err(std::vector<float>(first, first + s.os()), q));
        ekv = std::max(ekv, ref_kv_err(s, in, hc, t, pos0 + t));
    }
    const double eo = rel_err(a.o, ref_attention(s, in, hc, pos0, nb)), e8 = q8_err(a);
    const bool bits = same_bits(rig, s, cache, in, pos0, a);
    const bool ok = a.ok && eo <= TOL_O && eq <= TOL_Q && ekv <= TOL_KV && e8 <= TOL_Q8 && bits;
    printf(
        "decode  hd %3d nh %2d pos0 %6lld x %d tokens: o %.1e q %.1e kv %.2f steps q8 %.2f steps, same bits %s: %s\n",
        s.cfg.hd, s.nh, (long long)pos0, nb, eo, eq, ekv, e8, bits ? "yes" : "NO", ok ? "ok" : "FAILED");
    return ok;
}

// Prefill kernel on random q (already normed and rotated) against the oracle, for a sample of the tokens
bool check_prefill(const Shape& s, RandomCache& cache, const HostCache& hc, int64_t pos0, int tokens) {
    const int hd = s.cfg.hd;
    const size_t n = (size_t)tokens * s.os();
    const auto q = randoms(n, 31 + (uint32_t)pos0, 2.0f), gate = randoms(n, 32, 2.0f);
    const Dev dq(q), dg(gate);
    const DeviceBuffer o(n * 2);
    const gpu::FattnPrefillArgs a{dq.get<float>(),
                                  cache.get(),
                                  cache.row(),
                                  s.cfg.gate ? dg.get<float>() : nullptr,
                                  o.get<_Float16>(),
                                  pos0,
                                  tokens,
                                  s.nh,
                                  s.nkv,
                                  hd,
                                  1.0f / std::sqrt((float)hd)};
    const bool launched = gpu::launch_fattn_prefill(a, nullptr);
    CHECK(hipDeviceSynchronize());
    const auto got = download<_Float16>(o, n);
    const int64_t total = pos0 + tokens;
    std::vector<float> K((size_t)total * s.nkv * hd), V(K.size());
    for (int64_t p = 0; p < total; p++)
        for (int h = 0; h < s.nkv; h++)
            for (int i = 0; i < hd; i++)
                K[(p * s.nkv + h) * hd + i] = hc.at(p, h, false, i), V[(p * s.nkv + h) * hd + i] = hc.at(p, h, true, i);
    double err = 0;
    for (int t : {0, 1, tokens / 2, tokens - 2, tokens - 1}) {
        std::vector<int64_t> cells((size_t)(pos0 + t + 1));
        for (size_t c = 0; c < cells.size(); c++) cells[c] = (int64_t)c;
        auto want = tona::ref::attend_cells({s.nh, s.nkv, hd}, &q[(size_t)t * s.os()], K, V, cells);
        std::vector<float> have(want.size());
        for (size_t i = 0; i < want.size(); i++) {
            have[i] = (float)got[(size_t)t * s.os() + i];
            if (s.cfg.gate) want[i] *= tona::ref::sigmoid(gate[(size_t)t * s.os() + i]);
        }
        err = std::max(err, rel_err(have, want));
    }
    const bool ok = launched && err <= TOL_O;
    printf("prefill hd %3d nh %2d pos0 %6lld x %d tokens: o %.1e (5 sampled tokens): %s\n", hd, s.nh, (long long)pos0,
           tokens, err, ok ? "ok" : "FAILED");
    return ok;
}

// speed (--bench): the steps against the bytes of the cache and against the QSA path

constexpr double BW_GBS = 315.0;  // measured read bandwidth of the card (docs/backends/reference-platform.md)

// Microseconds per launch of the split and merge steps for one token at position pos (prepare runs once before)
double decode_us(Rig& rig, const Shape& s, RandomCache& cache, int64_t pos) {
    const Inputs in(s, 1, 5);
    const DecodeProgram d(s, cache, in, 0, 1);
    if (!rig.run({d.prep, d.split, d.merge}, pass_params(pos, 1))) return -1;
    return rig.time_us({d.split, d.merge}, pass_params(pos, 1), 20);
}

// The same for the QSA attention of the Coder (attn_wmma and merge, no indexer, dense path at pos < top-k)
double qsa_us(Rig& rig, const Shape& s, RandomCache& cache, int64_t pos) {
    const Dev q(randoms((size_t)s.os(), 3, 2.0f)), gate(randoms((size_t)s.os(), 4, 2.0f));
    const DeviceBuffer part((size_t)s.nh * gpu::ATT_SPL * (s.cfg.hd + 2) * 4), cells((size_t)gpu::MAX_CELLS * 4);
    const DeviceBuffer o((size_t)s.os() * 4), q8((size_t)s.os()), meta((size_t)s.os() / 32 * sizeof(gpu::Q8Meta));
    MegaOp attn, merge;
    attn.kind = gpu::OP_ATTN_WMMA, merge.kind = gpu::OP_ATTN_MERGE;
    attn.i[0] = merge.i[0] = s.nh, attn.i[1] = merge.i[1] = s.nkv, attn.i[4] = s.row(), merge.i[3] = s.os();
    attn.i[5] = merge.i[5] = 4, attn.i[7] = merge.i[7] = 2048, merge.i[6] = 1, attn.f[0] = 1.0f / 16.0f;
    attn.p[0] = q.get(), attn.p[1] = cache.get(), attn.p[2] = part.get(), attn.p[3] = cells.get();
    merge.p[2] = part.get(), merge.p[4] = gate.get(), merge.p[5] = o.get(), merge.p[6] = q8.get(),
    merge.p[7] = meta.get();
    return rig.time_us({attn, merge}, pass_params(pos, 1), 20);
}

void bench_decode(const Shape& s) {
    Rig rig(gpu::probe::fattn_kernels(s.cfg));
    RandomCache cache(LONG_CTX + 64, s.nkv, s.cfg.hd, 99);
    for (int64_t pos : {2047, 32767, 131071, 262143}) {
        const double us = decode_us(rig, s, cache, pos), bytes = (double)(pos + 1) * s.row();
        printf("decode hd %3d nh %2d nkv %d, %6lld positions: %8.1f us, %6.1f GB/s, %.2fx the time at %.0f GB/s\n",
               s.cfg.hd, s.nh, s.nkv, (long long)pos + 1, us, bytes / us / 1e3, us / (bytes / (BW_GBS * 1e3)), BW_GBS);
    }
}

double prefill_us(const Shape& s, RandomCache& cache, int tokens, bool qsa) {
    const size_t n = (size_t)tokens * s.os();
    const Dev q(randoms(n, 5, 2.0f)), gate(randoms(n, 6, 2.0f));
    const DeviceBuffer o(n * 2);
    const float scale = 1.0f / std::sqrt((float)s.cfg.hd);
    const gpu::FattnPrefillArgs a{
        q.get<float>(), cache.get(), cache.row(), gate.get<float>(), o.get<_Float16>(), 0, tokens, s.nh,
        s.nkv,          s.cfg.hd,    scale};
    const gpu::AttnWmmaArgs w{
        q.get<float>(), cache.get(), cache.row(), nullptr, nullptr, gate.get<float>(), o.get<_Float16>(), 0, s.nh,
        s.nkv,          4,           2048,        scale,   false};
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)), CHECK(hipEventCreate(&e1));
    std::vector<float> ms;
    for (int rep = 0; rep < 6; rep++) {
        CHECK(hipEventRecord(e0));
        if (qsa)
            gpu::launch_attn_wmma(w, tokens, nullptr);
        else
            gpu::launch_fattn_prefill(a, nullptr);
        CHECK(hipEventRecord(e1)), CHECK(hipEventSynchronize(e1));
        float t = 0;
        CHECK(hipEventElapsedTime(&t, e0, e1));
        if (rep > 0) ms.push_back(t);
    }
    std::sort(ms.begin(), ms.end());
    return 1000.0 * ms[ms.size() / 2] / tokens;
}

int bench() {
    const Shape coder{kOrnith, 24, 2};  // the Coder's heads on the cache layout both paths read
    {
        Rig qrig(gpu::probe::qsa_attention_kernels()), frig(gpu::probe::fattn_kernels(kOrnith));
        RandomCache cache(2100, coder.nkv, coder.cfg.hd, 98);
        printf("QSA path, dense below top-k, 2048 positions: %.1f us; full attention, same cache: %.1f us\n",
               qsa_us(qrig, coder, cache, 2047), decode_us(frig, coder, cache, 2047));
    }
    bench_decode({kOrnith, 16, 2});
    bench_decode({kBolt, 32, 2});
    const Shape pre{kOrnith, 16, 2};
    RandomCache cache(8300, pre.nkv, pre.cfg.hd, 97);
    for (int tokens : {1024, 2048})
        printf("prefill, %d tokens from position 0: QSA kernel %.2f us, full attention %.2f us per token and layer\n",
               tokens, prefill_us(pre, cache, tokens, true), prefill_us(pre, cache, tokens, false));
    printf("prefill, 8192 tokens from position 0 (full attention only): %.2f us per token and layer\n",
           prefill_us(pre, cache, 8192, false));
    return 0;
}

struct DecodeCase {
    int64_t pos0;
    int nb;
};
struct PrefillCase {
    int64_t pos0;
    int tokens;
};

// All cases of one shape: its own cache of LONG_CTX + 64 positions
bool run_shape(const Shape& s, const std::vector<DecodeCase>& dec, const std::vector<PrefillCase>& pre) {
    Rig rig(gpu::probe::fattn_kernels(s.cfg));
    RandomCache cache(LONG_CTX + 64, s.nkv, s.cfg.hd, 1234);
    HostCache hc(s, cache);
    bool ok = true;
    for (const auto& c : dec) ok = check_decode(rig, s, cache, hc, c.pos0, c.nb) && ok;
    for (const auto& c : pre) ok = check_prefill(s, cache, hc, c.pos0, c.tokens) && ok;
    return ok;
}

}  // namespace

#include "fattn_prefill_cmp.h"
#include "fattn_split_check.h"
#include "fattn_stress.h"

// The options that run one mode instead of the checks: its exit code, or -1 for none
int run_mode(int argc, char** argv) {
    const std::string opt = argc > 1 ? argv[1] : "";
    if (opt == "--bench") return bench();
    if (opt == "--prefill-bench") return prefill_bench();
    if (opt == "--prefill-cmp") return cmp_all() ? 0 : 1;
    if (opt != "--stress") return -1;
    const long n = argc > 2 ? std::strtol(argv[2], nullptr, 10) : 200;
    if (n <= 0) {
        std::fprintf(stderr, "--stress needs a positive repeat count\n");
        return 2;
    }
    return stress_all(static_cast<int>(n)) ? 0 : 1;
}

int main(int argc, char** argv) {
    if (const int rc = run_mode(argc, argv); rc >= 0) return rc;
    bool ok = true;
    ok = run_shape({kOrnith, 16, 2}, {{0, 1}, {1, 2}, {15, 1}, {16, 3}, {100, 4}, {4000, 3}, {32767, 4}, {262140, 3}},
                   {{0, 40}, {100, 70}, {5000, 33}, {262104, 40}}) &&
         ok;
    ok = run_shape({kBolt, 32, 2}, {{0, 1}, {63, 1}, {4097, 4}, {65536, 2}, {262140, 3}},
                   {{0, 37}, {3000, 64}, {262104, 40}}) &&
         ok;
    ok = run_shape({kBoltAll, 16, 2}, {{130, 2}, {9000, 1}}, {{500, 20}}) && ok;
    ok = cmp_all() && ok;
    ok = split_all() && ok;
    printf(ok ? "all ok\n" : "FAILED\n");
    return ok ? 0 : 1;
}
