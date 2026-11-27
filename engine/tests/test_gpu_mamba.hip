// The Mamba-2 steps of the megakernel (gpu/mega_op_mamba.h, run inside the real k_mega with a test-only profile, see
// gpu/probe/ and tests/mega_rig.h) and the prefill kernels (gpu/prefill_mamba.hip) at the dimensions of Bolt (128 heads
// of 64 rows, state 128, 8 groups, conv 4) against the CPU oracle's state-space step (cpu/ref_mamba2.h), N2 and N4 of
// docs/architecture/model-architectures.md:
//   decode: widths 1, 4, 8 against the oracle (outputs and the committed state);
//   replay: random widths and random accepted counts give the same bits as passes of exactly the accepted tokens;
//   prefill: 128 and 4096 tokens against the oracle, one piece against random pieces and against the decode path in
//   the same bits, and a decode pass continuing a prefilled state in the same bits as the all-decode run.
// Needs the GPU, no model.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "cpu/ref_mamba2.h"
#include "gpu/mega_ops.h"
#include "gpu/prefill_mamba.h"
#include "gpu/probe/probe_kernels.h"
#include "mega_rig.h"

namespace {

using namespace rig;
using tona::gpu::MegaOp;

constexpr int HEADS = 128, P = 64, N = 128, G = 8, K = 4, BMAX = tona::gpu::MEGA_BMAX;
constexpr int INNER = HEADS * P, CONV_DIM = INNER + 2 * G * N, IN_DIM = 2 * INNER + 2 * G * N + HEADS;
constexpr size_t DT_OFF = 2 * INNER + 2 * G * N, GN = (size_t)G * N,
                 GS = INNER / G;  // dt offset in a token, a group's B or C, a norm group
constexpr float EPS = 1e-5f, IN_SCALE = 0.75f, OUT_SCALE = 1.25f;
int fails = 0;
#define EXPECT(c, ...)                                                                                            \
    do {                                                                                                          \
        if (!(c))                                                                                                 \
            std::fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c), std::fprintf(stderr, __VA_ARGS__), fails++; \
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
    explicit Dev(size_t bytes) : DeviceBuffer(bytes) { CHECK(hipMemset(get(), 0, bytes)); }
};

template <class T>
std::vector<T> download(const DeviceBuffer& b, size_t n, size_t first = 0) {
    std::vector<T> v(n);
    CHECK(hipMemcpy(v.data(), b.get<T>() + first, n * sizeof(T), hipMemcpyDeviceToHost));
    return v;
}

// Weights of one layer; A is negative as in the models (a decay rate)
struct Layer {
    std::vector<float> conv_w = randoms((size_t)CONV_DIM * K, 1, 0.5f), conv_b = randoms(CONV_DIM, 2, 0.1f);
    std::vector<float> dt_bias = randoms(HEADS, 3, 0.5f, -1.0f), D = randoms(HEADS, 5, 0.1f, 1.0f);
    std::vector<float> norm = randoms(INNER, 6, 0.1f, 1.0f), A;
    Layer() {
        A = randoms(HEADS, 4, 0.5f, 0.5f);
        for (float& a : A) a = -std::exp(a);
    }
};

// Tokens of a stream: [z | x B C | dt] per token, as the in-projection writes them
std::vector<float> stream(int tokens, uint32_t seed) {
    std::vector<float> s = randoms((size_t)tokens * IN_DIM, seed, 1.0f);
    for (int t = 0; t < tokens; t++)
        for (int h = 0; h < HEADS; h++) s[(size_t)t * IN_DIM + DT_OFF + h] -= 1.0f;
    return s;
}

// The oracle: one token at a time with the primitives of cpu/ref_mamba2.h
class Oracle {
public:
    explicit Oracle(const Layer& L) : L_(L), ssm_((size_t)INNER * N, 0.0f), hist_((size_t)(K - 1) * CONV_DIM, 0.0f) {}
    std::vector<float> step(const float* raw) {
        std::vector<float> zx(IN_DIM);
        for (int i = 0; i < IN_DIM; i++) zx[i] = raw[i] * IN_SCALE;
        std::vector<float> xbc(zx.begin() + INNER, zx.begin() + INNER + CONV_DIM);
        conv(xbc);
        std::vector<float> dt(HEADS), y(INNER);
        for (int h = 0; h < HEADS; h++) dt[h] = zx[2 * INNER + 2 * G * N + h] + L_.dt_bias[h];
        const tona::ref::SsmStepInputs in{
            xbc.data(), xbc.data() + INNER, xbc.data() + INNER + GN, dt.data(), L_.A.data(), L_.D.data(), HEADS, P, N,
            G};
        tona::ref::ssm_step(in, ssm_.data(), y.data());
        for (int i = 0; i < INNER; i++) y[i] *= tona::ref::silu(zx[i]);
        for (int g = 0; g < G; g++) tona::ref::rms_norm(&y[g * GS], (int64_t)GS, EPS, &L_.norm[g * GS]);
        for (float& v : y) v *= OUT_SCALE;
        return y;
    }
    const std::vector<float>& state() const { return ssm_; }

private:
    void conv(std::vector<float>& v) {
        std::vector<float> out(CONV_DIM);
        for (int c = 0; c < CONV_DIM; c++) {
            float acc = 0;
            for (int k = 0; k < K - 1; k++) acc += L_.conv_w[(size_t)c * K + k] * hist_[(size_t)k * CONV_DIM + c];
            acc += L_.conv_w[(size_t)c * K + K - 1] * v[c];
            out[c] = tona::ref::silu(acc + L_.conv_b[c]);
        }
        std::memmove(hist_.data(), hist_.data() + CONV_DIM, (size_t)(K - 2) * CONV_DIM * sizeof(float));
        std::memcpy(hist_.data() + (size_t)(K - 2) * CONV_DIM, v.data(), (size_t)CONV_DIM * sizeof(float));
        v = std::move(out);
    }
    const Layer& L_;
    std::vector<float> ssm_, hist_;
};

// Device side of one layer: weights and the state buffers as Bolt's program allocates them
struct State {
    Dev ssm{(size_t)INNER * N * 4}, ring{(size_t)(K - 1 + BMAX) * CONV_DIM * 4};
    Dev pdec{(size_t)2 * BMAX * HEADS * 4}, pxdt{(size_t)2 * BMAX * INNER * 4}, pb{(size_t)2 * BMAX * G * N * 4};
    void clear() {
        CHECK(hipMemset(ssm.get(), 0, (size_t)INNER * N * 4));
        CHECK(hipMemset(ring.get(), 0, (size_t)(K - 1 + BMAX) * CONV_DIM * 4));
    }
};

struct Weights {
    Dev conv_w, conv_b, dt_bias, A, D, norm;
    explicit Weights(const Layer& L)
        : conv_w(L.conv_w), conv_b(L.conv_b), dt_bias(L.dt_bias), A(L.A), D(L.D), norm(L.norm) {}
};

// The three steps as a program for the rig, over buffers for up to BMAX tokens
class DecodeRun {
public:
    DecodeRun(const Layer& L, State& s)
        : W(L),
          S(s),
          rig(tona::gpu::probe::mamba_kernels()),
          zx((size_t)BMAX * IN_DIM * 4),
          conv((size_t)BMAX * CONV_DIM * 4),
          y((size_t)BMAX * INNER * 4),
          o((size_t)BMAX * INNER * 4),
          q8((size_t)BMAX * INNER),
          meta((size_t)BMAX * INNER / 32 * sizeof(tona::gpu::Q8Meta)) {
        MegaOp c, sc, n;
        c.kind = tona::gpu::OP_MAMBA_CONV;
        c.p[0] = zx.get(), c.p[1] = S.ring.get(), c.p[2] = W.conv_w.get(), c.p[3] = W.conv_b.get(), c.p[4] = conv.get();
        c.i[0] = CONV_DIM, c.i[1] = K, c.i[2] = IN_DIM, c.i[3] = INNER, c.f[0] = IN_SCALE;
        sc.kind = tona::gpu::OP_MAMBA_SCAN;
        sc.p[0] = zx.get(), sc.p[1] = conv.get(), sc.p[2] = W.dt_bias.get(), sc.p[3] = W.A.get(), sc.p[4] = W.D.get();
        sc.p[5] = S.ssm.get(), sc.p[6] = S.pdec.get(), sc.p[7] = S.pxdt.get(), sc.p[8] = S.pb.get(), sc.p[9] = y.get();
        sc.i[0] = HEADS, sc.i[1] = P, sc.i[3] = G, sc.i[4] = IN_DIM, sc.i[5] = 2 * INNER + 2 * G * N,
        sc.i[6] = CONV_DIM;
        sc.i[7] = INNER, sc.f[0] = IN_SCALE;
        n.kind = tona::gpu::OP_MAMBA_NORM;
        n.p[0] = y.get(), n.p[1] = zx.get(), n.p[2] = W.norm.get(), n.p[3] = q8.get(), n.p[4] = meta.get(),
        n.p[5] = o.get();
        n.i[0] = INNER, n.i[1] = G, n.i[2] = IN_DIM, n.f[0] = EPS, n.f[1] = IN_SCALE, n.f[2] = OUT_SCALE;
        prog = {c, sc, n};
    }
    // One pass of nb tokens at position pos; commit tokens of the previous pass are committed first. Returns the
    // normed output of the tokens.
    std::vector<float> pass(const float* tokens, int nb, int64_t pos, int commit) {
        CHECK(hipMemcpy(zx.get(), tokens, (size_t)nb * IN_DIM * 4, hipMemcpyHostToDevice));
        tona::gpu::MegaParams p{};
        p.nb = nb, p.pos = pos, p.commit = commit, p.parity = parity_;
        EXPECT(rig.run(prog, p), "pass aborted\n");
        parity_ ^= 1;
        return download<float>(o, (size_t)nb * INNER);
    }
    void reset() { S.clear(), parity_ = 0; }
    // Microseconds per launch of the three steps for a pass of nb tokens that commits nb tokens (N3)
    // ops: the steps timed (first, count); commit: tokens committed at the start of the scan
    double time_us(const float* tokens, int nb, int commit, int reps, int first = 0, int count = 3) {
        CHECK(hipMemcpy(zx.get(), tokens, (size_t)nb * IN_DIM * 4, hipMemcpyHostToDevice));
        tona::gpu::MegaParams p{};
        p.nb = nb, p.pos = 8, p.commit = commit, p.parity = 1;
        return rig.time_us(std::vector<MegaOp>(prog.begin() + first, prog.begin() + first + count), p, reps);
    }
    Weights W;
    State& S;
    Rig rig;

private:
    Dev zx, conv, y, o, q8, meta;
    std::vector<MegaOp> prog;
    int parity_ = 0;
};

double worst_rel(const float* got, const float* want, size_t n, double floor) {
    double w = 0, top = 0;
    for (size_t i = 0; i < n; i++) top = std::max(top, (double)std::fabs(want[i]));
    for (size_t i = 0; i < n; i++) w = std::max(w, std::fabs((double)got[i] - want[i]) / (top + floor));
    return w;
}

// The state after the committed tokens: a pass of one more token commits `commit`, the stored state is then the state
// after them (the pass itself stores nothing of its own token)
std::vector<float> committed_state(DecodeRun& r, const std::vector<float>& s, int tokens_done, int commit) {
    r.pass(s.data() + (size_t)tokens_done * IN_DIM, 1, tokens_done, commit);
    return download<float>(r.S.ssm, (size_t)INNER * N);
}

// N2 at one width: passes of nb tokens, each committed in full, against the oracle
void check_decode(const Layer& L, int nb) {
    const int T = 24;
    const std::vector<float> s = stream(T + 1, 100 + nb);
    State st;
    DecodeRun r(L, st);
    Oracle ref(L);
    double worst = 0;
    int commit = 0;
    for (int t = 0; t < T; t += nb) {
        const int w = std::min(nb, T - t);
        const std::vector<float> out = r.pass(s.data() + (size_t)t * IN_DIM, w, t, commit);
        for (int b = 0; b < w; b++) {
            const std::vector<float> want = ref.step(s.data() + (size_t)(t + b) * IN_DIM);
            worst = std::max(worst, worst_rel(out.data() + (size_t)b * INNER, want.data(), INNER, 1e-3));
        }
        commit = w;
    }
    const std::vector<float> got = committed_state(r, s, T, commit);
    const double sd = worst_rel(got.data(), ref.state().data(), got.size(), 1e-3);
    std::printf("decode width %d: output worst %.2e, state worst %.2e (relative to the largest value)\n", nb, worst,
                sd);
    EXPECT(worst < 1e-5 && sd < 1e-5, "decode width %d\n", nb);
}

// N4: random widths and random accepted counts (the rest is re-fed) against passes of exactly the accepted tokens:
// every accepted token's output and the final state must have the same bits
void check_replay(const Layer& L, uint32_t seed) {
    const int T = 48;
    const std::vector<float> s = stream(T + 1, 200 + seed);
    std::mt19937 rng(seed);
    State sa, sb;
    DecodeRun a(L, sa), b(L, sb);
    std::vector<std::vector<float>> out_a(T), out_b(T);
    int pos = 0, commit = 0, commit_b = 0, passes = 0;
    while (pos < T) {
        const int nb = std::min<int>(1 + (int)(rng() % 8), T - pos), m = 1 + (int)(rng() % nb);
        const std::vector<float> oa = a.pass(s.data() + (size_t)pos * IN_DIM, nb, pos, commit);
        const std::vector<float> ob = b.pass(s.data() + (size_t)pos * IN_DIM, m, pos, commit_b);
        for (int i = 0; i < m; i++)
            out_a[pos + i].assign(oa.begin() + (size_t)i * INNER, oa.begin() + (size_t)(i + 1) * INNER),
                out_b[pos + i].assign(ob.begin() + (size_t)i * INNER, ob.begin() + (size_t)(i + 1) * INNER);
        pos += m, commit = commit_b = m, passes++;
    }
    const bool same_out = out_a == out_b;
    const bool same_state = committed_state(a, s, T, commit) == committed_state(b, s, T, commit_b);
    EXPECT(same_out && same_state, "replay differs (seed %u)\n", seed);
    std::printf("replay seed %u: %d passes with rollbacks, outputs bit-equal: %s, state bit-equal: %s\n", seed, passes,
                same_out ? "yes" : "NO", same_state ? "yes" : "NO");
}

}  // namespace

#include "test_gpu_mamba_prefill.h"

namespace {

// N3: time of the decode steps per layer (the state is 4 MiB: read, and written at a commit) and of the prefill
// kernels per token; printed, no pass or fail
void bench(const Layer& L) {
    const std::vector<float> s = stream(4096, 9);
    State st;
    DecodeRun dec(L, st);
    for (int nb : {1, 4, 8}) {
        std::printf(
            "decode width %d, commit %d: %.1f us per layer (conv %.1f, scan %.1f, norm %.1f; scan without commit "
            "%.1f)\n",
            nb, nb, dec.time_us(s.data(), nb, nb, 200), dec.time_us(s.data(), nb, nb, 200, 0, 1),
            dec.time_us(s.data(), nb, nb, 200, 1, 1), dec.time_us(s.data(), nb, nb, 200, 2, 1),
            dec.time_us(s.data(), nb, 0, 200, 1, 1));
    }
    PrefillRun pf(L, 4096);
    pf.run(s.data(), 4096, 0);
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0)), CHECK(hipEventCreate(&e1));
    const char* names[] = {"all", "conv", "ring", "scan", "norm"};
    for (unsigned mask : {15u, 1u, 2u, 4u, 8u}) {
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < 3; i++) pf.run(s.data(), 4096, 0, {0, 0, false, mask});
        CHECK(hipEventRecord(e1)), CHECK(hipEventSynchronize(e1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("prefill 4096 tokens, %s: %.2f us per token and layer\n",
                    names[mask == 15 ? 0 : 1 + __builtin_ctz(mask)], 1000.0 * ms / 3 / 4096);
    }
    // the same with the chunked scan (dt, the scores and the scan: mask 4; all: 15)
    for (unsigned mask : {15u, 4u}) {
        CHECK(hipEventRecord(e0));
        for (int i = 0; i < 3; i++) pf.run(s.data(), 4096, 0, {0, 0, false, mask, true});
        CHECK(hipEventRecord(e1)), CHECK(hipEventSynchronize(e1));
        float ms = 0;
        CHECK(hipEventElapsedTime(&ms, e0, e1));
        std::printf("prefill 4096 tokens, chunked scan, %s: %.2f us per token and layer\n",
                    mask == 15 ? "all" : "dt and scan", 1000.0 * ms / 3 / 4096);
    }
}

}  // namespace

int main() {
    const Layer L;
    for (int nb : {1, 4, 8}) check_decode(L, nb);
    for (uint32_t seed : {1u, 2u, 3u}) check_replay(L, seed);
    check_prefill(L, 128);
    check_prefill(L, 4096);
    check_prefill(L, 128, true);
    check_prefill(L, 4096, true);
    check_prefill(L, 4000, true);
    check_prefill_cuts(L);
    check_ssd_cuts(L);
    bench(L);
    std::printf(fails ? "FAILED: %d\n" : "OK\n", fails);
    return fails ? 1 : 0;
}
