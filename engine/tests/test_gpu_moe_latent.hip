// The steps of the latent MoE (gpu/mega_op_moe_latent.h, run inside the real k_mega with a test-only profile,
// gpu/probe/ and tests/mega_rig.h) and the prefill kernels (gpu/prefill_moe_latent.hip) against the formulas of the
// oracle (NemotronHOracle::moe): relu(u)^2 bit for bit with the int8 within half a quantization step, the fixed-point
// sum of the weighted expert outputs in the latent space bit for bit against the host's use of the same functions, the
// plain element-wise steps, at the real widths of the Bolt file (latent 1024, expert width 1536 and 2560, shared expert
// 5376, d 4096), one to eight tokens per pass. Needs the GPU, no model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "cpu/detmath.h"
#include "gpu/prefill_moe_latent.h"
#include "gpu/probe/probe_kernels.h"
#include "mega_rig.h"

namespace {

using namespace rig;
using tona::gpu::MegaOp;

constexpr int UMAX = gpu::MEGA_UMAX, LAT = 1024, SFF = 5376, D = 4096;
int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

std::vector<float> randoms(size_t n, uint32_t seed, float sd) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> g(0.0f, sd);
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

// Largest distance of the int8 values from the floats they quantize, in steps of their block of 32
double q8_err(const std::vector<float>& x, const std::vector<int8_t>& q, const std::vector<gpu::Q8Meta>& meta,
              const std::vector<int64_t>& at) {
    double worst = 0;
    for (const int64_t i : at) {
        const float d = meta[i / 32].d;
        worst = std::max(worst, d > 0 ? std::fabs(q[i] * d - x[i]) / d : 0.0);
    }
    return worst;
}

// The places the activation step covers per token: the first n_u * ff values and the shared expert behind UMAX * ff
std::vector<int64_t> act_places(int nb, int n_u, int ff, int64_t stride) {
    std::vector<int64_t> at;
    for (int b = 0; b < nb; b++) {
        for (int64_t i = 0; i < (int64_t)n_u * ff; i++) at.push_back(b * stride + i);
        for (int64_t i = 0; i < SFF; i++) at.push_back(b * stride + (int64_t)UMAX * ff + i);
    }
    return at;
}

void check_relu2(Rig& rig, int nb, int n_u, int ff) {
    const int64_t n = (int64_t)UMAX * ff + SFF, stride = (int64_t)UMAX * 2560 + SFF;  // buffers sized for the widest
    const auto u = randoms((size_t)nb * stride, 5 + nb + n_u + ff, 1.0f);
    const Dev du(u);
    const DeviceBuffer dout(nb * stride * 4), dq(nb * stride), dm(nb * stride / 32 * sizeof(gpu::Q8Meta));
    CHECK(hipMemset(dout.get(), 0, nb * stride * 4));
    CHECK(hipMemset(dq.get(), 0, nb * stride));
    CHECK(hipMemset(dm.get(), 0, nb * stride / 32 * sizeof(gpu::Q8Meta)));
    MegaOp op;
    op.kind = gpu::OP_ACT_RELU2, op.i[0] = (int)n, op.i[1] = ff, op.i[2] = (int)stride;
    op.p[0] = du.get(), op.p[1] = dout.get(), op.p[2] = dq.get(), op.p[3] = dm.get();
    gpu::MegaParams p{};
    p.nb = nb, p.commit = nb, p.n_u = n_u;
    const bool ran = rig.run({op}, p);
    const auto out = download<float>(dout, nb * stride);
    const auto q = download<int8_t>(dq, nb * stride);
    const auto meta = download<gpu::Q8Meta>(dm, nb * stride / 32);
    const auto at = act_places(nb, n_u, ff, stride);
    std::vector<float> want(nb * stride, 0.0f);
    for (const int64_t i : at) want[i] = u[i] > 0.0f ? u[i] * u[i] : 0.0f;
    bool exact = ran;
    for (int64_t i = 0; i < nb * stride; i++) exact = exact && out[i] == want[i];  // also: nothing else was written
    const double e8 = q8_err(want, q, meta, at);
    EXPECT(exact && e8 <= 0.51);
    std::printf("relu2   nb %d n_u %2d ff %d: float %s, q8 %.2f steps: %s\n", nb, n_u, ff,
                exact ? "exact" : "DIFFERENT", e8, exact && e8 <= 0.51 ? "ok" : "FAILED");
}

// Token b chooses `k` of the n_u places (rotating), weights summing to 5 as the router's scale makes them
void make_weights(gpu::MegaParams& p, int nb, int n_u, int k) {
    for (int b = 0; b < nb; b++)
        for (int j = 0; j < k; j++) {
            const int e = (b * 3 + j * 2) % n_u;
            p.ew[b * UMAX + e] = 5.0f * (1.0f + (float)((b + j) % 4)) / (k + 1.5f * k);
            p.umask[e] |= 1u << b;
        }
}

void check_sum_lat(Rig& rig, int nb, int n_u, int k) {
    const size_t n = (size_t)nb * UMAX * LAT;
    const auto ed = randoms(n, 9 + nb + n_u, 0.7f);
    const Dev ded(ed);
    const DeviceBuffer dy((size_t)nb * LAT * 4), dq((size_t)nb * LAT), dm((size_t)nb * LAT / 32 * sizeof(gpu::Q8Meta));
    MegaOp op;
    op.kind = gpu::OP_MOE_SUM_LAT, op.i[0] = UMAX, op.i[1] = LAT;
    op.p[0] = ded.get(), op.p[1] = dy.get(), op.p[2] = dq.get(), op.p[3] = dm.get();
    gpu::MegaParams p{};
    p.nb = nb, p.commit = nb, p.n_u = n_u;
    make_weights(p, nb, n_u, k);
    const bool ran = rig.run({op}, p);
    const auto y = download<float>(dy, (size_t)nb * LAT);
    const auto q = download<int8_t>(dq, (size_t)nb * LAT);
    const auto meta = download<gpu::Q8Meta>(dm, (size_t)nb * LAT / 32);
    std::vector<float> want((size_t)nb * LAT);
    std::vector<int64_t> all((size_t)nb * LAT);
    for (int b = 0; b < nb; b++)
        for (int i = 0; i < LAT; i++) {
            int64_t a = 0;
            for (int e = 0; e < n_u; e++)
                if (p.ew[b * UMAX + e] != 0.0f)
                    a += tona::det::to_fix(p.ew[b * UMAX + e] * ed[((size_t)b * UMAX + e) * LAT + i]);
            want[(size_t)b * LAT + i] = tona::det::from_fix(a);
        }
    std::iota(all.begin(), all.end(), 0);
    const bool exact = ran && y == want;
    const double e8 = q8_err(want, q, meta, all);
    EXPECT(exact && e8 <= 0.51);
    std::printf("sum.lat nb %d n_u %2d k %d: float %s, q8 %.2f steps: %s\n", nb, n_u, k, exact ? "exact" : "DIFFERENT",
                e8, exact && e8 <= 0.51 ? "ok" : "FAILED");
}

void check_quant_and_add(Rig& rig, int nb) {
    const auto x = randoms((size_t)nb * LAT, 21 + nb, 1.3f);
    const Dev dx(x);
    const DeviceBuffer dq((size_t)nb * LAT), dm((size_t)nb * LAT / 32 * sizeof(gpu::Q8Meta));
    MegaOp op;
    op.kind = gpu::OP_QUANT_LAT, op.i[0] = LAT, op.p[0] = dx.get(), op.p[1] = dq.get(), op.p[2] = dm.get();
    gpu::MegaParams p{};
    p.nb = nb, p.commit = nb;
    const bool ran = rig.run({op}, p);
    std::vector<int64_t> all((size_t)nb * LAT);
    std::iota(all.begin(), all.end(), 0);
    const double e8 = q8_err(x, download<int8_t>(dq, x.size()), download<gpu::Q8Meta>(dm, x.size() / 32), all);
    const auto a = randoms((size_t)nb * D, 31 + nb, 1.0f), b = randoms((size_t)nb * D, 32 + nb, 1.0f);
    const Dev da(a), db(b);
    const DeviceBuffer dy((size_t)nb * D * 4);
    MegaOp add;
    add.kind = gpu::OP_MOE_ADD, add.i[0] = D, add.p[0] = da.get(), add.p[1] = db.get(), add.p[2] = dy.get();
    const bool ran2 = rig.run({add}, p);
    const auto y = download<float>(dy, a.size());
    bool exact = ran2;
    for (size_t i = 0; i < a.size(); i++) exact = exact && y[i] == a[i] + b[i];
    EXPECT(ran && e8 <= 0.51 && exact);
    std::printf("quant.lat / add nb %d: q8 %.2f steps, sum %s: %s\n", nb, e8, exact ? "exact" : "DIFFERENT",
                ran && e8 <= 0.51 && exact ? "ok" : "FAILED");
}

// Rows 3 .. rows - 3 of the activation are written, the others stay zero
bool prefill_relu2(int rows, int ff, int n) {
    const auto u = randoms((size_t)rows * ff, 41 + n, 1.0f);
    const Dev du(u);
    const DeviceBuffer dact((size_t)rows * ff * 2);
    CHECK(hipMemset(dact.get(), 0, (size_t)rows * ff * 2));
    gpu::launch_relu2_16({du.get<float>(), dact.get<_Float16>(), 3, rows - 5, ff}, nullptr);
    const auto act = download<_Float16>(dact, (size_t)rows * ff);
    bool ok = true;
    for (size_t i = 0; i < act.size(); i++) {
        const bool inside = i >= (size_t)3 * ff && i < (size_t)(rows - 2) * ff;
        const float v = u[i];
        ok = ok && act[i] == (inside ? (_Float16)(v > 0.0f ? v * v : 0.0f) : (_Float16)0.0f);
    }
    return ok;
}

// The sum in the latent space: f16 output within f16 rounding of the float sum (the device may contract the products
// into fused multiply-adds, the host does not)
bool prefill_sum(int n, int ne) {
    const int rows = n * ne;
    const auto out = randoms((size_t)rows * LAT, 43 + n, 0.7f);
    const auto sw = randoms((size_t)rows, 44 + n, 1.0f);
    std::vector<int32_t> rowmap(rows);
    for (int i = 0; i < rows; i++) rowmap[i] = (i * 7 + 3) % rows;
    const Dev dout(out), dsw(sw), drm(rowmap);
    const DeviceBuffer dy((size_t)n * LAT * 2);
    gpu::launch_moe_sum_latent(
        {dout.get<float>(), drm.get<int32_t>(), dsw.get<float>(), dy.get<_Float16>(), n, LAT, ne}, nullptr);
    const auto y = download<_Float16>(dy, (size_t)n * LAT);
    bool ok = true;
    for (int t = 0; t < n; t++)
        for (int k = 0; k < LAT; k++) {
            float s = 0;
            for (int r = 0; r < ne; r++) s += sw[(size_t)t * ne + r] * out[(size_t)rowmap[t * ne + r] * LAT + k];
            ok = ok && std::fabs((float)y[(size_t)t * LAT + k] - s) <= std::fabs(s) * 1.5e-3f + 1e-6f;
        }
    return ok;
}

bool prefill_convert_and_add(int rows, int n) {
    const auto out = randoms((size_t)rows * LAT, 43 + n, 0.7f), a = randoms((size_t)rows * LAT, 45 + n, 1.0f);
    const Dev dout(out), da(a);
    const DeviceBuffer d16((size_t)rows * LAT * 2);
    gpu::launch_f32_to_f16(dout.get<float>(), d16.get<_Float16>(), (int64_t)rows * LAT, nullptr);
    const auto h = download<_Float16>(d16, (size_t)rows * LAT);
    bool ok = true;
    for (size_t i = 0; i < h.size(); i++) ok = ok && h[i] == (_Float16)out[i];
    gpu::launch_add_into(dout.get<float>(), da.get<float>(), (int64_t)rows * LAT, nullptr);
    const auto s = download<float>(dout, out.size());
    for (size_t i = 0; i < s.size(); i++) ok = ok && s[i] == out[i] + a[i];
    return ok;
}

// The prefill kernels: activation (f16), the sum in the latent space (f16), conversion, add
void check_prefill(int n, int ne, int ff) {
    const bool relu = prefill_relu2(n * ne, ff, n), sum = prefill_sum(n, ne), rest = prefill_convert_and_add(n * ne, n);
    CHECK(hipDeviceSynchronize());
    EXPECT(relu && sum && rest);
    std::printf("prefill kernels n %d ne %d ff %d: relu2 %d, sum %d, convert and add %d: %s\n", n, ne, ff, relu, sum,
                rest, relu && sum && rest ? "ok" : "FAILED");
}

}  // namespace

int main() {
    Rig rig(gpu::probe::latent_moe_kernels());
    for (int nb : {1, 4, 8})
        for (int ff : {1536, 2560}) check_relu2(rig, nb, nb == 1 ? 9 : 31, ff);
    for (int nb : {1, 4, 8}) {
        check_sum_lat(rig, nb, nb == 1 ? 9 : 40, 9);
        check_quant_and_add(rig, nb);
    }
    check_prefill(37, 9, 1536);
    check_prefill(300, 9, 2560);
    std::printf(fails ? "FAILED\n" : "all ok\n");
    return fails ? 1 : 0;
}
