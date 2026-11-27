// The prefill half of tests/test_gpu_mamba.hip: the kernels of gpu/prefill_mamba.hip against the oracle, cut invariance
// and the same bits as the decode path.
#pragma once

namespace {

// Open: the m tokens still open from a decode pass (half `half` of the pending buffers). Timing runs skip the upload
// and select kernels by mask bits (1 conv, 2 ring, 4 dt and scan, 8 norm).
struct Open {
    int m = 0, half = 0;
    bool upload = true;
    unsigned mask = 15;
    bool ssd = false;  // the chunked scan on the matrix units instead of the sequential one
};

// Device side of a prefill of up to `cap` tokens
class PrefillRun {
public:
    PrefillRun(const Layer& L, int cap)
        : W(L),
          zx((size_t)cap * IN_DIM * 4),
          conv((size_t)cap * CONV_DIM * 4),
          y((size_t)cap * INNER * 4),
          o((size_t)cap * INNER * 2),
          decay((size_t)cap * HEADS * 4),
          dt((size_t)cap * HEADS * 4),
          da((size_t)cap * HEADS * 4),
          cb((size_t)(cap / 64 + 2) * G * 64 * 64 * 4 * 3) {}
    void run(const float* tokens, int n, int64_t pos0, const Open& open = {}) {
        const auto [m, half, upload, mask, ssd] = open;
        namespace g = tona::gpu;
        if (upload) CHECK(hipMemcpy(zx.get(), tokens, (size_t)n * IN_DIM * 4, hipMemcpyHostToDevice));
        const int dt_off = 2 * INNER + 2 * G * N;
        if (mask & 1)
            g::launch_mamba_conv({zx.get<float>(), S.ring.get<float>(), W.conv_w.get<float>(), W.conv_b.get<float>(),
                                  conv.get<float>(), n, pos0, CONV_DIM, K, IN_DIM, INNER, IN_SCALE},
                                 nullptr);
        if (mask & 2)
            g::launch_mamba_ring({zx.get<float>(), S.ring.get<float>(), n, pos0, CONV_DIM, K, IN_DIM, INNER, IN_SCALE},
                                 nullptr);
        if (mask & 4) {
            g::launch_mamba_dt({zx.get<float>(), W.dt_bias.get<float>(), W.A.get<float>(), decay.get<float>(),
                                dt.get<float>(), da.get<float>(), n, HEADS, IN_DIM, dt_off, IN_SCALE},
                               nullptr);
            if (!ssd || m > 0)  // the sequential scan also commits the open decode tokens
                g::launch_mamba_scan({conv.get<float>(), decay.get<float>(), dt.get<float>(), W.D.get<float>(),
                                      S.ssm.get<float>(), y.get<float>(), S.pdec.get<float>(), S.pxdt.get<float>(),
                                      S.pb.get<float>(), ssd ? 0 : n, m, half, HEADS, P, G, CONV_DIM, INNER},
                                     nullptr);
            if (ssd)
                g::launch_mamba_ssd(
                    {conv.get<float>(), da.get<float>(), dt.get<float>(), W.D.get<float>(), S.ssm.get<float>(),
                     y.get<float>(), cb.get<float>(), pos0, n, 0, HEADS, P, G, CONV_DIM, INNER},
                    nullptr);
        }
        if (mask & 8)
            g::launch_mamba_norm({y.get<float>(), zx.get<float>(), W.norm.get<float>(), o.get<_Float16>(), n, INNER, G,
                                  IN_DIM, EPS, IN_SCALE, OUT_SCALE},
                                 nullptr);
        CHECK(hipDeviceSynchronize());
    }
    std::vector<_Float16> out(int n) const { return download<_Float16>(o, (size_t)n * INNER); }
    Weights W;
    State S;

private:
    Dev zx, conv, y, o, decay, dt, da, cb;
};

// N2 for the prefill: T tokens in one call against the oracle (normed output as f16, and the state). The sequential
// scan computes the oracle's steps in float (bounds 2e-3 and 1e-5); the chunked one reorders the sums on f16 operands
// (bound 3e-3 each, stated before the first run: f16 operands carry 5e-4 relative each, a product of two 1e-3 per term,
// and the sums average that down; the bound is about twice what that gives)
void check_prefill(const Layer& L, int T, bool ssd = false) {
    const std::vector<float> s = stream(T, 300 + T);
    PrefillRun pf(L, T);
    Open open;
    open.ssd = ssd;
    pf.run(s.data(), T, 0, open);
    const std::vector<_Float16> got = pf.out(T);
    Oracle ref(L);
    double worst = 0;
    std::vector<float> g(INNER);
    for (int t = 0; t < T; t++) {
        const std::vector<float> want = ref.step(s.data() + (size_t)t * IN_DIM);
        for (int i = 0; i < INNER; i++) g[i] = (float)got[(size_t)t * INNER + i];
        worst = std::max(worst, worst_rel(g.data(), want.data(), INNER, 1e-3));
    }
    const std::vector<float> st = download<float>(pf.S.ssm, (size_t)INNER * N);
    const double sd = worst_rel(st.data(), ref.state().data(), st.size(), 1e-3);
    std::printf(
        "prefill %d tokens, %s scan: output worst %.2e (f16), state worst %.2e (relative to the largest value)\n", T,
        ssd ? "chunked" : "sequential", worst, sd);
    EXPECT(worst < (ssd ? 3e-3 : 2e-3) && sd < (ssd ? 3e-3 : 1e-5), "prefill %d, ssd %d\n", T, (int)ssd);
}

// N4 for the prefill: one piece, random pieces and the decode path end in the same bits; a decode pass after a
// prefilled state gives the same bits as after the same tokens run by decode
void check_prefill_cuts(const Layer& L) {
    const int T = 130, X = 4;
    const std::vector<float> s = stream(T + X, 400);
    PrefillRun whole(L, T);
    whole.run(s.data(), T, 0);
    PrefillRun cut(L, T);
    std::mt19937 rng(7);
    int pos = 0;
    while (pos < T) {
        const int n = std::min<int>(1 + (int)(rng() % 40), T - pos);
        cut.run(s.data() + (size_t)pos * IN_DIM, n, pos);
        std::vector<_Float16> part = cut.out(n);
        EXPECT(std::equal(part.begin(), part.end(), whole.out(T).begin() + (size_t)pos * INNER), "piece at %d\n", pos);
        pos += n;
    }
    const auto ssm = [](PrefillRun& r) { return download<float>(r.S.ssm, (size_t)INNER * N); };
    EXPECT(ssm(whole) == ssm(cut), "state of random pieces differs from one piece\n");
    State sd;
    DecodeRun dec(L, sd);
    int commit = 0;
    for (int t = 0; t < T; t++) dec.pass(s.data() + (size_t)t * IN_DIM, 1, t, commit), commit = 1;
    const std::vector<float> after = dec.pass(s.data() + (size_t)T * IN_DIM, X, T, commit);
    EXPECT(download<float>(sd.ssm, (size_t)INNER * N) == ssm(whole), "decode state differs from the prefilled one\n");
    DecodeRun cont(L, whole.S);
    EXPECT(cont.pass(s.data() + (size_t)T * IN_DIM, X, T, 0) == after, "decode after prefill differs from decode\n");
    std::printf("prefill cuts: pieces, decode path and continuation compared, T = %d\n", T);
}

// The chunked scan: a prompt cut at multiples of 64 gives the same bits as the whole (the chunk grid is the absolute
// position), a prompt cut anywhere stays within the bound of the whole, and a decode pass after a chunked prefill stays
// within the bound of the one after a sequential prefill (the states differ by the rounding of the f16 operands)
void check_ssd_cuts(const Layer& L) {
    const int T = 320, X = 4;
    const std::vector<float> s = stream(T + X, 500);
    Open open;
    open.ssd = true;
    PrefillRun whole(L, T);
    whole.run(s.data(), T, 0, open);
    const auto ssm = [](PrefillRun& r) { return download<float>(r.S.ssm, (size_t)INNER * N); };
    const std::vector<_Float16> all = whole.out(T);
    PrefillRun grid(L, T);
    int pos = 0;
    for (const int n : {64, 128, 64, 64}) {
        grid.run(s.data() + (size_t)pos * IN_DIM, n, pos, open);
        const std::vector<_Float16> part = grid.out(n);
        EXPECT(std::equal(part.begin(), part.end(), all.begin() + (size_t)pos * INNER), "grid piece at %d\n", pos);
        pos += n;
    }
    EXPECT(ssm(grid) == ssm(whole), "state of pieces cut on the grid differs from the whole\n");
    PrefillRun odd(L, T);
    std::mt19937 rng(11);
    for (pos = 0; pos < T;) {
        const int n = std::min<int>(1 + (int)(rng() % 90), T - pos);
        odd.run(s.data() + (size_t)pos * IN_DIM, n, pos, open);
        pos += n;
    }
    const std::vector<float> a = ssm(odd), b = ssm(whole);
    const double sd = worst_rel(a.data(), b.data(), a.size(), 1e-3);
    EXPECT(sd < 3e-3, "state of pieces cut anywhere differs by %.2e\n", sd);
    PrefillRun seq(L, T);
    seq.run(s.data(), T, 0);
    DecodeRun dc(L, whole.S), ds(L, seq.S);
    const std::vector<float> got = dc.pass(s.data() + (size_t)T * IN_DIM, X, T, 0);
    const std::vector<float> want = ds.pass(s.data() + (size_t)T * IN_DIM, X, T, 0);
    const double dd = worst_rel(got.data(), want.data(), got.size(), 1e-3);
    EXPECT(dd < 3e-3, "decode after the chunked prefill differs by %.2e\n", dd);
    std::printf(
        "chunked prefill: grid cuts bit-equal, random cuts: state within %.2e, decode after it within %.2e of the "
        "decode after the sequential prefill, T = %d\n",
        sd, dd, T);
}

}  // namespace
