// The sigmoid routing policy on the GPU (backend/route_policy.h): the step of the megakernel (gpu/mega_op_route_sig.h)
// and the prefill kernel (gpu/prefill_route_sig.hip) against the CPU side (backend::ExpertRoute) bit for bit (experts,
// order, weights as float bits: both sides call the same functions), and against the float64 reference of
// tools/gen_route_ref.py (experts exactly, weights within float rounding). Unusable logits, E that is not a multiple of
// 32, up to 512 experts. Needs the GPU, no model.
// Usage: test_gpu_route_sig [route_ref.txt]
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <vector>

#include "backend/expert_route.h"
#include "gpu/mega_ops.h"
#include "gpu/prefill_route_sig.h"
#include "gpu/probe/route_sig_step.h"
#include "tests/route_ref_data.h"
#include "tools/tona_hip_check.h"

using tona::tools::DeviceBuffer;
namespace gpu = tona::gpu;
namespace backend = tona::backend;

namespace {

constexpr int UMAX = gpu::MEGA_UMAX, NB = 3;
constexpr size_t kCntBytes = 4096;  // 1024 int32 counters
int fails = 0;
#define EXPECT(c)                                                                    \
    do {                                                                             \
        if (!(c)) std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), fails++; \
    } while (0)

backend::RoutePolicy policy_of(const routeref::Case& c) {
    return {backend::Gate::Sigmoid, c.scale, c.with_bias ? c.bias.data() : nullptr};
}

// Logits of token t of a case: its own row of the case's stream
std::vector<float> token_logits(const routeref::Case& c, int t) {
    return t == 0 ? c.logits : routeref::make_logits(c.seed + 7 * (uint64_t)t, c.E, c.levels);
}

template <class T>
std::vector<T> download(const DeviceBuffer& b, size_t n) {
    std::vector<T> v(n);
    CHECK(hipMemcpy(v.data(), b.get(), n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

template <class T>
void upload(const DeviceBuffer& b, const std::vector<T>& v) {
    CHECK(hipMemcpy(b.get(), v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
}

size_t sz(int a, int b = 1, int c = 1) { return (size_t)a * (size_t)b * (size_t)c; }

bool same_bits(const float* a, const float* b, size_t n) { return std::memcmp(a, b, n * sizeof(float)) == 0; }

// The union the step left in the parameters against the CPU's, tokens first .. first + n - 1
void compare_union(const gpu::MegaParams& p, const std::vector<int>& uni, const std::vector<uint32_t>& um,
                   const std::vector<float>& ew, int first, int n) {
    EXPECT(p.n_u == (int)uni.size());
    for (size_t u = 0; u < uni.size(); u++) {
        EXPECT(p.slot[u] == uni[u]);
        EXPECT(p.umask[u] == um[u] << first);
    }
    for (int b = 0; b < n; b++) EXPECT(same_bits(&p.ew[sz(first + b, UMAX)], &ew[sz(b, UMAX)], uni.size()));
}

// The megakernel step on tokens 0 .. nb - 1 of a case; all: rows of all tokens (several sequences), else the last only
void check_step(const routeref::Case& c, int nb, bool all) {
    std::vector<float> logits;
    for (int t = 0; t < nb; t++) {
        const auto r = token_logits(c, t);
        logits.insert(logits.end(), r.begin(), r.end());
    }
    const DeviceBuffer dl(logits.size() * 4), dbias(sz(c.E) * 4), dbad(sz(gpu::BMAX) * 4), dp(sizeof(gpu::MegaParams));
    upload(dl, logits);
    if (c.with_bias) upload(dbias, c.bias);
    CHECK(hipMemset(dbad.get(), 0, sz(gpu::BMAX) * 4));
    gpu::MegaParams p{};
    p.nb = nb, p.nseq = all ? 1 : 0;
    CHECK(hipMemcpy(dp.get(), &p, sizeof p, hipMemcpyHostToDevice));
    gpu::MegaOp op;
    op.kind = gpu::OP_ROUTE_SIG, op.i[0] = c.E, op.i[1] = c.K, op.f[0] = c.scale;
    op.p[0] = dl.get(), op.p[1] = dbad.get(), op.p[2] = c.with_bias ? dbias.get() : nullptr;
    gpu::probe::launch_route_sig_step(op, dp.get<gpu::MegaParams>(), nb);
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(&p, dp.get(), sizeof p, hipMemcpyDeviceToHost));
    // The CPU side over the same tokens
    const int first = all ? 0 : nb - 1, n = nb - first;
    backend::ExpertRoute route({c.E, c.K, n, UMAX, policy_of(c)});
    for (int b = 0; b < n; b++) route.rank_token(b, &logits[sz(first + b, c.E)], [](int) { return false; });
    std::vector<float> ew((size_t)n * UMAX, 0.0f);
    std::vector<uint32_t> um(UMAX, 0);
    route.build_union(ew.data(), um.data());
    compare_union(p, route.uni(), um, ew, first, n);
}

// n rows of a case, some with NaN and +inf mixed in
std::vector<float> prefill_logits(const routeref::Case& c, int n) {
    std::vector<float> logits;
    for (int t = 0; t < n; t++) {
        auto r = token_logits(c, t);
        if (t % 11 == 5) r[c.E / 2] = std::nanf("");
        if (t % 13 == 7) r[c.E - 1] = INFINITY;
        logits.insert(logits.end(), r.begin(), r.end());
    }
    return logits;
}

// The prefill kernel on n tokens: rows with NaN and +inf mixed in; the experts, their order and the weights equal the
// CPU's bits, the counts add up
void check_prefill(const routeref::Case& c, int n) {
    const std::vector<float> logits = prefill_logits(c, n);
    constexpr int kNanWord = 700;
    const DeviceBuffer dl(logits.size() * 4), dbias(sz(c.E) * 4), dsel(sz(n, c.K) * 4), dsw(sz(n, c.K) * 4),
        dcnt(kCntBytes);
    upload(dl, logits);
    if (c.with_bias) upload(dbias, c.bias);
    CHECK(hipMemset(dcnt.get(), 0, kCntBytes));
    gpu::launch_route_sig({dl.get<float>(), c.with_bias ? dbias.get<float>() : nullptr, dsel.get<int32_t>(),
                           dsw.get<float>(), dcnt.get<int32_t>(), n, c.E, c.K, kNanWord, c.scale},
                          nullptr);
    CHECK(hipDeviceSynchronize());
    const auto sel = download<int32_t>(dsel, sz(n, c.K)), cnt = download<int32_t>(dcnt, 1024);
    const auto sw = download<float>(dsw, sz(n, c.K));
    std::vector<int> count(c.E, 0);
    int unusable = 0;
    for (int t = 0; t < n; t++) {
        backend::ExpertRoute route({c.E, c.K, 1, UMAX, policy_of(c)});
        unusable += route.rank_token(0, &logits[sz(t, c.E)], [](int) { return false; }) > 0;
        std::vector<float> ew(UMAX, 0.0f);
        std::vector<uint32_t> um(UMAX, 0);
        route.build_union(ew.data(), um.data());
        EXPECT(std::equal(sel.begin() + (long)sz(t, c.K), sel.begin() + (long)sz(t + 1, c.K), route.uni().begin()));
        EXPECT(same_bits(&sw[sz(t, c.K)], ew.data(), c.K));
        for (int k = 0; k < c.K; k++) count[sel[sz(t, c.K) + k]]++;
    }
    EXPECT(cnt[kNanWord] == unusable);
    EXPECT(std::equal(count.begin(), count.end(), cnt.begin()));
}

// Both GPU paths on the first token of a case against the float64 reference
void check_reference(const routeref::Case& c) {
    const DeviceBuffer dl(sz(c.E) * 4), dbias(sz(c.E) * 4), dsel(sz(c.K) * 4), dsw(sz(c.K) * 4), dcnt(kCntBytes);
    upload(dl, c.logits);
    if (c.with_bias) upload(dbias, c.bias);
    CHECK(hipMemset(dcnt.get(), 0, kCntBytes));
    gpu::launch_route_sig({dl.get<float>(), c.with_bias ? dbias.get<float>() : nullptr, dsel.get<int32_t>(),
                           dsw.get<float>(), dcnt.get<int32_t>(), 1, c.E, c.K, -1, c.scale},
                          nullptr);
    CHECK(hipDeviceSynchronize());
    const auto sel = download<int32_t>(dsel, c.K);
    const auto sw = download<float>(dsw, c.K);
    EXPECT(std::equal(sel.begin(), sel.end(), c.experts.begin()));
    for (int k = 0; k < c.K; k++) EXPECT(std::fabs(sw[k] - c.weights[k]) <= 2e-6 * c.weights[k]);
}

}  // namespace

int main(int argc, char** argv) try {
    const auto cases = routeref::load(argc > 1 ? argv[1] : "tests/data/route_ref.txt");
    int n_cases = 0;
    for (const auto& c : cases) {
        if (c.gate != "sigmoid") continue;
        n_cases++;
        check_reference(c);
        check_prefill(c, 77);
        check_step(c, NB, true);
        check_step(c, NB, false);
        check_step(c, 1, false);
    }
    EXPECT(n_cases >= 20);
    std::printf("test_gpu_route_sig: %s (%d sigmoid cases)\n", fails ? "FAILED" : "ok", n_cases);
    return fails ? 1 : 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "test_gpu_route_sig: %s\n", e.what());
    return 2;
}
