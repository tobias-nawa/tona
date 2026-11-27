// f16 GEMM with WMMA (gfx12): two kernel generations and the host entry points gemm_f16 / gemm_f16_mix.
// Y[T x N] = X[T x K] * W[N x K]^T. Measurement: research/measurements/2026-10-23-gemm-wmma.md.
#include <cstdio>
#include <cstdlib>

#include "backend/env.h"
#include "gpu/gemm_dev.h"

namespace tona::gpu {

namespace {

struct GemmArgs {
    GemmSeg s[GEMM_MAX_SEGS];
    int32_t tile0[GEMM_MAX_SEGS + 1];  // first tile of each segment
    int32_t nseg;
};

// First generation: K step 32, double-buffered LDS

__global__ __launch_bounds__(256) void k_gemm(GemmArgs a) {
    __shared__ __attribute__((aligned(16))) _Float16 sa[2][BT * LDK], sb[2][BN * LDK];
    const int b = blockIdx.x;
    int si = 0;  // segment of block b: the last one whose first tile is <= b
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const GemmSeg& S = a.s[si];
    const int T = S.t, N = S.n, K = S.k;
    const int nt = (T + BT - 1) / BT, nn = (N + BN - 1) / BN, loc = b - a.tile0[si];
    const TileOrigin o = tile_origin(loc, nt, nn, BT, BN);
    const int t0 = o.t0, n0 = o.n0;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = (wave >> 2) * WT, wn = (wave & 3) * WN;
    // Loading: per thread 2 x 16 bytes of X and of W per K step; rows past the edge read the last valid one
    const int lr = tid >> 2, lc = (tid & 3) * 8;
    const _Float16* xa0 = S.x + (int64_t)min(t0 + lr, T - 1) * S.ldx + lc;
    const _Float16* xa1 = S.x + (int64_t)min(t0 + lr + 64, T - 1) * S.ldx + lc;
    const _Float16* wb0 = S.w + (int64_t)min(n0 + lr, N - 1) * S.ldw + lc;
    const _Float16* wb1 = S.w + (int64_t)min(n0 + lr + 64, N - 1) * S.ldw + lc;
    int4v ra0, ra1, rb0, rb1;
    auto gload = [&](int k) {
        ra0 = *reinterpret_cast<const int4v*>(xa0 + k);
        ra1 = *reinterpret_cast<const int4v*>(xa1 + k);
        rb0 = *reinterpret_cast<const int4v*>(wb0 + k);
        rb1 = *reinterpret_cast<const int4v*>(wb1 + k);
    };
    auto sstore = [&](int buf) {
        *reinterpret_cast<int4v*>(&sa[buf][lr * LDK + lc]) = ra0;
        *reinterpret_cast<int4v*>(&sa[buf][(lr + 64) * LDK + lc]) = ra1;
        *reinterpret_cast<int4v*>(&sb[buf][lr * LDK + lc]) = rb0;
        *reinterpret_cast<int4v*>(&sb[buf][(lr + 64) * LDK + lc]) = rb1;
    };
    float8 acc[4][2] = {};
    gload(0);
    sstore(0);
    lds_sync();
    const WaveFrag f{wt, wn, lane & 15, (lane >> 4) * 8};
    for (int k = 0, buf = 0; k < K; k += BK, buf ^= 1) {
        if (k + BK < K) gload(k + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) mma_kstep<4, 2, LDK>(sa[buf], sb[buf], f, kk, acc);
        if (k + BK < K) sstore(buf ^ 1);
        lds_sync();
    }
    store_tile<4, 2>(&S, o, f, acc);
}

// Second generation
// Measured under sustained load at the power limit (182 W, bench/micro/gemm2.hip): K step 64 with a single LDS
// buffer (the load into registers overlaps the computation), tile 128 x BN2, 8 waves 2 x 4 of 64 x BN2 / 4.
// 2560 -> 10240: 53.9 -> 67.1 TFLOPS (BN2 = 256); 6144 -> 2560: 37.7 -> 48.5 TFLOPS (BN2 = 128).

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }

template <int BN2>
struct Shape2 {
    static constexpr int WN2 = BN2 / 4, FJ = WN2 / 16, CPR = BK2 / 8, LA = BT * CPR / 256, LB = BN2 * CPR / 256;
};

// What one block needs everywhere: its segment, tile, wave position and the two LDS tiles
template <int BN2>
struct Gemm2Ctx {
    const GemmSeg* s;  // a pointer: a copy would load every field of the segment up front
    GemmMix mx;
    TileOrigin o;
    _Float16 *sa, *sb;
    int tid, T, N, K;
};

template <int BN2>
__device__ __forceinline__ WaveFrag g2_frag(int tid) {
    const int lane = tid & 31, wave = tid >> 5;
    return {(wave >> 2) * WT, (wave & 3) * Shape2<BN2>::WN2, lane & 15, (lane >> 4) * 8};
}

// MIX: column n' = 4k + g reads weight row g * d + k; the four streams of k sit next to each other in a tile
template <int BN2, bool MIX>
__device__ __forceinline__ void g2_load(const Gemm2Ctx<BN2> c, int k, int4v (&ra)[Shape2<BN2>::LA],
                                        int4v (&rb)[Shape2<BN2>::LB]) {
    using Sh = Shape2<BN2>;
#pragma unroll
    for (int q = 0; q < Sh::LA; q++) {
        const int e = c.tid + q * 256, row = e / Sh::CPR, col = (e % Sh::CPR) * 8;
        ra[q] = *reinterpret_cast<const int4v*>(c.s->x + (int64_t)min(c.o.t0 + row, c.T - 1) * c.s->ldx + k + col);
    }
#pragma unroll
    for (int q = 0; q < Sh::LB; q++) {
        const int e = c.tid + q * 256, row = e / Sh::CPR, col = (e % Sh::CPR) * 8;
        const int nr = min(c.o.n0 + row, c.N - 1);
        const int64_t wr = MIX ? (int64_t)(nr & 3) * c.mx.d + (nr >> 2) : nr;
        rb[q] = *reinterpret_cast<const int4v*>(c.s->w + wr * c.s->ldw + k + col);
    }
}

template <int BN2>
__device__ __forceinline__ void g2_store(const Gemm2Ctx<BN2> c, const int4v (&ra)[Shape2<BN2>::LA],
                                         const int4v (&rb)[Shape2<BN2>::LB]) {
    using Sh = Shape2<BN2>;
#pragma unroll
    for (int q = 0; q < Sh::LA; q++) {
        const int e = c.tid + q * 256;
        *reinterpret_cast<int4v*>(&c.sa[(e / Sh::CPR) * LDK2 + (e % Sh::CPR) * 8]) = ra[q];
    }
#pragma unroll
    for (int q = 0; q < Sh::LB; q++) {
        const int e = c.tid + q * 256;
        *reinterpret_cast<int4v*>(&c.sb[(e / Sh::CPR) * LDK2 + (e % Sh::CPR) * 8]) = rb[q];
    }
}

// Accumulators of a wave: 4 x FJ WMMA tiles
template <int BN2>
struct Acc2 {
    float8 v[4][Shape2<BN2>::FJ];
};

template <int BN2, bool MIX>
__device__ __forceinline__ Acc2<BN2> g2_mainloop(const Gemm2Ctx<BN2> c) {
    int4v ra[Shape2<BN2>::LA], rb[Shape2<BN2>::LB];
    Acc2<BN2> acc{};
    g2_load<BN2, MIX>(c, 0, ra, rb);
    g2_store<BN2>(c, ra, rb);
    lds_sync();
    const WaveFrag f = g2_frag<BN2>(c.tid);
    for (int k = 0; k < c.K; k += BK2) {
        if (k + BK2 < c.K) g2_load<BN2, MIX>(c, k + BK2, ra, rb);
#pragma unroll
        for (int kk = 0; kk < BK2; kk += 16) mma_kstep<4, Shape2<BN2>::FJ, LDK2>(c.sa, c.sb, f, kk, acc.v);
        if (k + BK2 < c.K) {
            lds_sync();
            g2_store<BN2>(c, ra, rb);
        }
        lds_sync();
    }
    return acc;
}

// mixed[t][k] = mean over the streams g of R * sc * w * sigmoid(gate), as k_mix; gv: the four gates of (t, k)
__device__ __forceinline__ void g2_mix_one(const GemmMix mx, const float* gv, int t, int kk) {
    constexpr int hc = 4;
    const int64_t nd = (int64_t)hc * mx.d;
    float s = 0;
    for (int g = 0; g < hc; g++) {
        const int64_t jj = (int64_t)t * nd + g * mx.d + kk;
        s += mx.R[jj] * mx.sc[t * 4 + g] * mx.w[g * mx.d + kk] * sigm(gv[g]);
    }
    const float m = s / hc;
    mx.mixed[(int64_t)t * mx.d + kk] = m;
    mx.mixed16[(int64_t)t * mx.d + kk] = (_Float16)m;
}

// Through the LDS in four passes of 32 tokens (16 per wave half): store the tile first, then mix each (token, k)
template <int BN2>
__device__ __forceinline__ void g2_mix_epilogue(const Gemm2Ctx<BN2> c, const float8 (&acc)[4][Shape2<BN2>::FJ]) {
    static_assert(BN2 == 256 && 32 * (BN2 + 4) * 4 <= (BT + BN2) * LDK2 * 2);
    constexpr int LS = BN2 + 4;
    float* const ls = reinterpret_cast<float*>(c.sa);
    const int k0 = c.o.n0 >> 2, kl = c.tid & 63;
    const WaveFrag f = g2_frag<BN2>(c.tid);
    lds_sync();
#pragma unroll
    for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < Shape2<BN2>::FJ; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) ls[((c.tid >> 7) * 16 + f.kh + e) * LS + f.wn + j * 16 + f.r] = acc[i][j][e];
        lds_sync();
        for (int tl = c.tid >> 6; tl < 32; tl += 4) {
            const int t = c.o.t0 + (tl >> 4) * WT + i * 16 + (tl & 15), kk = k0 + kl;
            if (t < c.T && kk < c.mx.d) g2_mix_one(c.mx, ls + tl * LS + kl * 4, t, kk);
        }
        lds_sync();
    }
}

template <int BN2, bool MIX>
__global__ __launch_bounds__(256) void k_gemm2(GemmArgs a, GemmMix mx) {
    using Sh = Shape2<BN2>;
    __shared__ __attribute__((aligned(16))) _Float16 smem[(BT + BN2) * LDK2];
    const int b = blockIdx.x;
    int si = 0;  // segment of block b: the last one whose first tile is <= b
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const GemmSeg& S = a.s[si];
    const int T = S.t, N = S.n, K = S.k;
    const int nt = (T + BT - 1) / BT, nn = (N + BN2 - 1) / BN2, loc = b - a.tile0[si];
    const int tid = threadIdx.x;
    const Gemm2Ctx<BN2> c{&S, mx, tile_origin(loc, nt, nn, BT, BN2), smem, smem + BT * LDK2, tid, T, N, K};
    const Acc2<BN2> acc = g2_mainloop<BN2, MIX>(c);
    if constexpr (MIX)
        g2_mix_epilogue<BN2>(c, acc.v);
    else
        store_tile<4, Sh::FJ>(c.s, c.o, g2_frag<BN2>(tid), acc.v);
}

// Host

void check_f16_seg(const GemmSeg& s) {
    if (s.k % BK || s.ldx % 8 || s.ldw % 8 ||
        (reinterpret_cast<uintptr_t>(s.x) | reinterpret_cast<uintptr_t>(s.w)) % 16)
        throw std::runtime_error(std::format("gemm_f16: K = {} or alignment does not fit", s.k));
}

// One launch worth of segments: up to GEMM_MAX_SEGS non-empty ones starting at segs[first]
struct F16Launch {
    GemmArgs a{};
    bool k64 = true, wide = true;
    int tiles = 0, next = 0;
};

F16Launch collect_f16(const GemmSeg* segs, int nseg, int first, bool allow_k64) {
    F16Launch l;
    l.k64 = allow_k64;
    int i = first;
    for (; i < nseg && l.a.nseg < GEMM_MAX_SEGS; i++) {
        const GemmSeg& s = segs[i];
        if (s.t <= 0 || s.n <= 0) continue;
        check_f16_seg(s);
        l.k64 = l.k64 && s.k % BK2 == 0;
        l.wide = l.wide && s.n >= 4096;
        l.a.s[l.a.nseg++] = s;
    }
    l.next = i;
    // Kernel per launch: K step 64 if all segments allow it, with BN 256 for wide outputs (all N >= 4096)
    const int bn = !l.k64 ? BN : l.wide ? 256 : 128;
    for (int k = 0; k < l.a.nseg; k++) {
        l.a.tile0[k] = l.tiles;
        l.tiles += ((l.a.s[k].t + BT - 1) / BT) * ((l.a.s[k].n + bn - 1) / bn);
    }
    l.a.tile0[l.a.nseg] = l.tiles;
    return l;
}

}  // namespace

void gemm_f16(const GemmSeg* segs, int nseg, hipStream_t stream) {
    static const bool v1 = tona::env::flag("TONA_GEMM_V1");  // first generation only
    static const bool glog = tona::env::text("TONA_GEMM_LOG") != nullptr;
    for (int i0 = 0; i0 < nseg;) {
        const F16Launch l = collect_f16(segs, nseg, i0, !v1);
        i0 = l.next;
        if (l.tiles == 0) continue;
        if (glog)
            for (int i = 0; i < l.a.nseg; i++)
                std::fprintf(stderr, "GEMMLOG %d %d %d %d\n", l.a.s[i].t, l.a.s[i].n, l.a.s[i].k, i);
        if (!l.k64)
            k_gemm<<<l.tiles, 256, 0, stream>>>(l.a);
        else if (l.wide)
            k_gemm2<256, false><<<l.tiles, 256, 0, stream>>>(l.a, GemmMix{});
        else
            k_gemm2<128, false><<<l.tiles, 256, 0, stream>>>(l.a, GemmMix{});
        HIPCHECK(hipGetLastError());
    }
}

void gemm_f16_mix(const GemmSeg& s, const GemmMix& m, hipStream_t stream) {
    if (s.n != 4 * m.d || m.d % 64 || s.k % BK2 || s.ldx % 8 || s.ldw % 8 ||
        (reinterpret_cast<uintptr_t>(s.x) | reinterpret_cast<uintptr_t>(s.w)) % 16)
        throw std::runtime_error(std::format("gemm_f16_mix: N = {}, d = {}, K = {} do not fit", s.n, m.d, s.k));
    if (s.t <= 0) return;
    GemmArgs a{};
    a.s[0] = s;
    a.nseg = 1;
    a.tile0[0] = 0;
    a.tile0[1] = ((s.t + BT - 1) / BT) * ((s.n + 255) / 256);
    k_gemm2<256, true><<<a.tile0[1], 256, 0, stream>>>(a, m);
    HIPCHECK(hipGetLastError());
}

}  // namespace tona::gpu
