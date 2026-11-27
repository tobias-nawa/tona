// The plain residual step OP_RES_NORM of the megakernel (gpu/mega_op_resnorm.h, run inside the real k_mega with a
// test-only profile, gpu/probe/ and tests/mega_rig.h) and the prefill kernel (gpu/prefill_res.hip) against float64: R'
// = R + y bit for bit, xn = RMSNorm(R') * w within float rounding (rsqrtf), int8 within half a quantization step. Also:
// without y (embedding), without a norm, in place, and the same bits with another grid size and with one token per pass
// instead of several (the sums are split by PNB, not by the width of the pass). Needs the GPU, no model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/prefill_res.h"
#include "gpu/probe/probe_kernels.h"
#include "mega_rig.h"

namespace {

using namespace rig;
using tona::gpu::MegaOp;

constexpr float EPS = 1e-6f;
int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

std::vector<float> randoms(size_t n, uint32_t seed, float sd, float mean = 0.0f) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(mean, sd);
    std::vector<float> v(n);
    for (float& x : v) x = g(rng);
    return v;
}

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

// What the step gets and writes. in_place: R' goes to R itself.
struct Variant {
    bool y, w, in_place;
};

struct Inputs {
    std::vector<float> R, y, w;
    Inputs(int d, int tokens, uint32_t seed)
        : R(randoms((size_t)d * tokens, seed, 1.5f, 0.3f)),
          y(randoms((size_t)d * tokens, seed + 1, 0.7f)),
          w(randoms(d, seed + 2, 0.2f, 1.0f)) {}
};

struct Result {
    std::vector<float> Rout, xn;
    std::vector<int8_t> q8;
    std::vector<gpu::Q8Meta> meta;
    bool ok = false;
};

// Tokens [t0, t0 + nb) of the inputs through the step on `blocks` blocks (0: all)
Result run_step(Rig& rig, const Inputs& in, int d, const Variant& v, const int (&span)[3]) {
    const int t0 = span[0], nb = span[1], blocks = span[2];
    const size_t n = (size_t)nb * d, off = (size_t)t0 * d;
    const std::vector<float> R(in.R.begin() + off, in.R.begin() + off + n),
        y(in.y.begin() + off, in.y.begin() + off + n);
    const Dev dR(R), dy(y), dw(in.w);
    const DeviceBuffer dRout(n * 4), dxn(n * 4), dq(n), dmeta(n / 32 * sizeof(gpu::Q8Meta));
    // Outputs the step does not write (no w) must compare equal between runs: zero, not whatever the allocator hands
    // out
    CHECK(hipMemset(dRout.get(), 0, n * 4));
    CHECK(hipMemset(dxn.get(), 0, n * 4));
    CHECK(hipMemset(dq.get(), 0, n));
    CHECK(hipMemset(dmeta.get(), 0, n / 32 * sizeof(gpu::Q8Meta)));
    MegaOp op;
    op.kind = gpu::OP_RES_NORM, op.i[0] = d, op.f[0] = EPS;
    op.p[0] = dR.get(), op.p[1] = v.y ? dy.get() : nullptr, op.p[2] = v.w ? dw.get() : nullptr;
    op.p[3] = v.in_place ? dR.get() : dRout.get(), op.p[4] = dxn.get(), op.p[5] = dq.get(), op.p[6] = dmeta.get();
    gpu::MegaParams p{};
    p.nb = nb, p.commit = nb;
    Result r;
    r.ok = rig.run({op}, p, blocks);
    r.Rout = download<float>(v.in_place ? dR : dRout, n), r.xn = download<float>(dxn, n);
    r.q8 = download<int8_t>(dq, n), r.meta = download<gpu::Q8Meta>(dmeta, n / 32);
    return r;
}

// R + y in float as the step adds it; the norm in double
struct Reference {
    std::vector<float> sum, xn;
    Reference(const Inputs& in, int d, int tokens, const Variant& v) : sum((size_t)d * tokens), xn(sum.size()) {
        for (int t = 0; t < tokens; t++) {
            double ss = 0;
            for (int i = 0; i < d; i++) {
                const size_t k = (size_t)t * d + i;
                sum[k] = in.R[k] + (v.y ? in.y[k] : 0.0f);
                ss += (double)sum[k] * sum[k];
            }
            const double scale = 1.0 / std::sqrt(ss / d + EPS);
            for (int i = 0; i < d; i++) xn[(size_t)t * d + i] = (float)(sum[(size_t)t * d + i] * scale * in.w[i]);
        }
    }
};

double worst_rel(const std::vector<float>& got, const std::vector<float>& want) {
    double worst = 0;
    for (size_t i = 0; i < want.size(); i++)
        worst = std::max(worst, std::fabs((double)got[i] - want[i]) / (std::fabs((double)want[i]) + 1e-3));
    return worst;
}

// The int8 output against the float one, in steps of its block of 32
double q8_err(const Result& r) {
    double worst = 0;
    for (size_t i = 0; i < r.xn.size(); i++) {
        const float d = r.meta[i / 32].d;
        worst = std::max(worst, d > 0 ? std::fabs(r.q8[i] * d - r.xn[i]) / d : 0.0);
    }
    return worst;
}

// Same bits on another grid size, and with one token per pass (the first and the last) instead of all nb
bool same_bits(Rig& rig, const Inputs& in, int d, const Variant& v, const Result& a) {
    const int nb = (int)(a.xn.size() / d);
    const Result few = run_step(rig, in, d, v, {0, nb, 5});
    bool same = few.ok && few.Rout == a.Rout && few.xn == a.xn && few.q8 == a.q8;
    for (int t : {0, nb - 1}) {
        const Result one = run_step(rig, in, d, v, {t, 1, 0});
        same = same && one.ok && std::equal(one.xn.begin(), one.xn.end(), a.xn.begin() + (long)t * d) &&
               std::equal(one.Rout.begin(), one.Rout.end(), a.Rout.begin() + (long)t * d);
    }
    return same;
}

void check_decode(Rig& rig, int d, int nb, const Variant& v) {
    const Inputs in(d, nb, 11 + (uint32_t)d + (uint32_t)nb);
    const Reference ref(in, d, nb, v);
    const Result a = run_step(rig, in, d, v, {0, nb, 0});
    double err = 0, e8 = 0;
    const bool bits = a.ok && a.Rout == ref.sum;  // R' = R + y in float, bit for bit (without y: R)
    if (v.w) err = worst_rel(a.xn, ref.xn), e8 = q8_err(a);
    const bool same = same_bits(rig, in, d, v, a);
    const bool ok = bits && err <= 3e-6 && e8 <= 0.51 && same;
    EXPECT(ok);
    std::printf("decode d %4d x %d tokens, y %d w %d in place %d: R' %s, xn %.1e, q8 %.2f steps, same bits %s: %s\n", d,
                nb, v.y, v.w, v.in_place, bits ? "exact" : "DIFFERENT", err, e8, same ? "yes" : "NO",
                ok ? "ok" : "FAILED");
}

void check_prefill(int d, int tokens, const Variant& v) {
    const Inputs in(d, tokens, 77 + (uint32_t)d);
    const Reference ref(in, d, tokens, v);
    const size_t n = (size_t)d * tokens;
    const Dev dR(in.R), dy(in.y), dw(in.w);
    const DeviceBuffer dxn(n * 2);
    gpu::launch_res_norm16({dR.get<float>(), v.y ? dy.get<float>() : nullptr, v.w ? dw.get<float>() : nullptr,
                            dxn.get<_Float16>(), d, EPS},
                           tokens, nullptr);
    CHECK(hipDeviceSynchronize());
    const auto R = download<float>(dR, n);
    const auto xn = download<_Float16>(dxn, n);
    double err = 0;
    for (size_t i = 0; i < n && v.w; i++)
        err = std::max(err, std::fabs((double)xn[i] - ref.xn[i]) / (std::fabs((double)ref.xn[i]) + 1e-2));
    const bool ok = R == (v.y ? ref.sum : in.R) && err <= 1e-3;  // f16: half a unit in the 11th bit
    EXPECT(ok);
    std::printf("prefill d %4d x %d tokens, y %d w %d: R' %s, xn %.1e: %s\n", d, tokens, v.y, v.w,
                R == (v.y ? ref.sum : in.R) ? "exact" : "DIFFERENT", err, ok ? "ok" : "FAILED");
}

}  // namespace

int main() {
    Rig rig(gpu::probe::resnorm_kernels());
    for (int d : {2048, 4096, 2080})
        for (int nb : {1, 3, 4, 8})
            for (const Variant v : {Variant{true, true, false}, Variant{false, true, false},
                                    Variant{true, false, false}, Variant{true, true, true}})
                check_decode(rig, d, nb, v);
    for (int d : {2048, 4096})
        for (const Variant v : {Variant{true, true, false}, Variant{false, true, false}, Variant{true, false, false}})
            check_prefill(d, 37, v);
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
