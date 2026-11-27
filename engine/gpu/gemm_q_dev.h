// GEMM with the weights in GGUF format: each block unpacks the 128 rows of its output tile into the LDS per K step,
// without an f16 copy in memory. For experts with few tokens per chunk. The kernels and the launch planning are
// here; gemm_q.hip and gemm_q_bolt.hip each name the formats they instantiate (a format of one architecture does
// not change the machine code of the units of the others).
#pragma once

#include <cstdlib>

#include "backend/env.h"
#include "gpu/gemm_deq_dev.h"
#include "gpu/gemm_dev.h"

namespace tona::gpu {

namespace {

constexpr int QT = 64;  // tokens per tile

struct GemmQArgs {
    GemmQSeg s[GEMM_MAX_SEGS];
    int32_t tile0[GEMM_MAX_SEGS + 1];
    int32_t nseg;
};

// Sub-block to f16 in the LDS. Formats with an int8 unpacker (matvec_dev.h: tables from the LDS, IQ4 values through
// v_perm from registers) go through decode_q8, w = s * q + b per 16-value half; the float formats through deq_sub.
// With deq_sub alone (tables read from memory per value) IQ4_NL took three times as long as the grid formats.
template <uint32_t TY>
__device__ __forceinline__ void deq_lds(const uint8_t* row, int in, int sub, const Tabs& tb, _Float16* dst) {
    if constexpr (TY == 0 || TY == 1 || TY == 30) {
        float o[32];
        deq_sub<TY>(row, sub, o);
#pragma unroll
        for (int q = 0; q < 4; q++) {
            half8 h;
#pragma unroll
            for (int j = 0; j < 8; j++) h[j] = (_Float16)o[8 * q + j];
            *reinterpret_cast<half8*>(dst + 8 * q) = h;
        }
    } else {
        Dec o;
        decode_row<TY>(row, in, sub, tb, o);
#pragma unroll
        for (int q = 0; q < 4; q++) {  // 8 values = 2 words
            const float sc = q < 2 ? o.s_lo : o.s_hi, bo = q < 2 ? o.b_lo : o.b_hi;
            half8 h;
#pragma unroll
            for (int j = 0; j < 8; j++) {
                const uint32_t wd = (uint32_t)o.w[2 * q + j / 4], by = (wd >> (8 * (j % 4))) & 0xff;
                const float v = q8_unsigned<TY>() ? (float)by : (float)(int8_t)by;
                h[j] = (_Float16)(sc * v + bo);
            }
            *reinterpret_cast<half8*>(dst + 8 * q) = h;
        }
    }
}

// Per K step (32 values) threads 0..127 unpack one row of the output tile, 128..255 load X (64 x 32).
// 8 waves as 2 (T) x 4 (N), 32 x 32 per wave = 2 x 2 WMMA tiles.
template <uint32_t TY>
__global__ __launch_bounds__(256) void k_gemm_q(GemmQArgs a) {
    if constexpr (TY == 16) load_iq2xxs_lds();  // before the barrier in TABS_DECL
    TABS_DECL;
    __shared__ __attribute__((aligned(16))) _Float16 sa[QT * LDK], sb[BN * LDK];
    const int b = blockIdx.x;
    int si = 0;
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const GemmQSeg& S = a.s[si];
    const int T = S.t, N = S.n, K = S.k;
    const int nt = (T + QT - 1) / QT, nn = (N + BN - 1) / BN, loc = b - a.tile0[si];
    const int g = loc / (GT * nn), gt = min(GT, nt - g * GT), in = loc % (GT * nn);
    const int t0 = (g * GT + in % gt) * QT, n0 = (in / gt) * BN;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = (wave >> 2) * 32, wn = (wave & 3) * 32;
    const uint8_t* wrow = row_at<TY>(S.w, min(n0 + tid, N - 1), S.row_bytes);  // for tid < 128
    const int xi = tid - 128, xr = xi >> 1, xc = (xi & 1) * 16;
    const _Float16* xrow = S.x + (int64_t)min(t0 + xr, T - 1) * S.ldx + xc;  // for tid >= 128
    float8 acc[2][2] = {};
    const WaveFrag f{wt, wn, lane & 15, (lane >> 4) * 8};
    for (int k = 0; k < K; k += BK) {
        if (tid < 128) {
            deq_lds<TY>(wrow, K, k / 32, tabs, &sb[tid * LDK]);
        } else {
            *reinterpret_cast<int4v*>(&sa[xr * LDK + xc]) = *reinterpret_cast<const int4v*>(xrow + k);
            *reinterpret_cast<int4v*>(&sa[xr * LDK + xc + 8]) = *reinterpret_cast<const int4v*>(xrow + k + 8);
        }
        lds_sync();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) mma_kstep<2, 2, LDK>(sa, sb, f, kk, acc);
        lds_sync();
    }
    store_tile_q<2, 2>(&S, {t0, n0}, f, acc);
}

// Second generation: K step 64, all 256 threads unpack one sub-block each (row tid / 2, sub-block tid % 2) and load
// TT / 32 x 16 bytes of X each (TT x 64). In the first one only 128 threads unpacked while the others loaded 4 KB and
// waited. TT tokens per tile: 64, or 128 for experts with many tokens (every tile unpacks the 128 weight rows of its
// output columns again, so twice the tokens per tile halve the unpacking per token). The K sum of every output element
// runs in the same order of WMMA steps whatever TT is, so the results are bit-identical.
template <uint32_t TY, int TT>
__global__ __launch_bounds__(256) void k_gemm_q2(GemmQArgs a) {
    constexpr int NI = TT / 32, XR = TT / 32;
    if constexpr (TY == 16) load_iq2xxs_lds();  // before the barrier in TABS_DECL
    TABS_DECL;
    __shared__ __attribute__((aligned(16))) _Float16 sa[TT * LDK2], sb[BN * LDK2];
    const int b = blockIdx.x;
    int si = 0;
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const GemmQSeg& S = a.s[si];
    const int T = S.t, N = S.n, K = S.k;
    const int nt = (T + TT - 1) / TT, nn = (N + BN - 1) / BN, loc = b - a.tile0[si];
    const int g = loc / (GT * nn), gt = min(GT, nt - g * GT), in = loc % (GT * nn);
    const int t0 = (g * GT + in % gt) * TT, n0 = (in / gt) * BN;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    // A partial last token tile of up to TT / 2 tokens (the experts see few tokens per pass: 42 % of the tile rows were
    // empty at 13K): the four waves of the upper token half have nothing to compute and skip the WMMAs. The order of
    // the K sum of every output element is unchanged, so the result is bit-identical.
    const bool half = T - t0 <= TT / 2;
    const WaveFrag f{(wave >> 2) * (TT / 2), (wave & 3) * 32, lane & 15, (lane >> 4) * 8};
    const int dr = tid >> 1, dh = tid & 1;
    const uint8_t* wrow = row_at<TY>(S.w, min(n0 + dr, N - 1), S.row_bytes);
    const _Float16* xr[XR];
#pragma unroll
    for (int q = 0; q < XR; q++) xr[q] = S.x + (int64_t)min(t0 + 32 * q + (tid >> 3), T - 1) * S.ldx + (tid & 7) * 8;
    float8 acc[NI][2] = {};
    for (int k = 0; k < K; k += BK2) {
        int4v xv[XR];
#pragma unroll
        for (int q = 0; q < XR; q++) xv[q] = *reinterpret_cast<const int4v*>(xr[q] + k);
        deq_lds<TY>(wrow, K, k / 32 + dh, tabs, &sb[dr * LDK2 + dh * 32]);
#pragma unroll
        for (int q = 0; q < XR; q++)
            *reinterpret_cast<int4v*>(&sa[(32 * q + (tid >> 3)) * LDK2 + (tid & 7) * 8]) = xv[q];
        lds_sync();
        if (!half || wave < 4) {
#pragma unroll
            for (int kk = 0; kk < BK2; kk += 16) mma_kstep<NI, 2, LDK2>(sa, sb, f, kk, acc);
        }
        lds_sync();
    }
    if (!half || wave < 4) store_tile_q<NI, 2>(&S, {t0, n0}, f, acc);
}

using QLauncher = void (*)(const GemmQArgs&, int, bool, bool, hipStream_t);

template <uint32_t TY>
void launch_gemm_q(const GemmQArgs& a, int tiles, bool k64, bool wide, hipStream_t stream) {
    if (k64 && wide)
        k_gemm_q2<TY, 128><<<tiles, 256, 0, stream>>>(a);
    else if (k64)
        k_gemm_q2<TY, 64><<<tiles, 256, 0, stream>>>(a);
    else
        k_gemm_q<TY><<<tiles, 256, 0, stream>>>(a);
}

struct QFormat {
    uint32_t type;
    QLauncher launch;
};

// One launch worth of segments: up to GEMM_MAX_SEGS non-empty ones starting at segs[first]
struct QLaunch {
    GemmQArgs a{};
    int tiles = 0, next = 0;
    bool k64 = true, wide = false;
};

// Tokens per tile 128 from this mean of tokens per segment on (TONA_GEMMQ_T128_MIN, 0 turns the wide tile off):
// below it the second half of most tiles is empty and only the unpacking is spent twice as wide.
// 96: between T = 64 (wide tile 1 to 25 % slower) and T = 100 (8 to 13 % faster) of tona-gemm-q-bench
// (research/measurements/2026-11-16-prefill-lever7.md); T = 160 is slower again (5 to 13 %), the mean is a weak
// predictor, the end-to-end gain (+0.45 to +0.62 %) is in the same file
int wide_min() {
    static const int v = (int)tona::env::integer("TONA_GEMMQ_T128_MIN", 96);
    return v;
}

QLaunch collect_q(const GemmQSeg* segs, int nseg, int first, bool allow_k64) {
    QLaunch l;
    l.k64 = allow_k64;
    int i = first;  // empty segments do not count: continue at i
    int64_t tokens = 0;
    for (; i < nseg && l.a.nseg < GEMM_MAX_SEGS; i++) {
        const GemmQSeg& s = segs[i];
        if (s.t <= 0 || s.n <= 0) continue;
        if (s.k % BK || s.ldx % 8 || reinterpret_cast<uintptr_t>(s.x) % 16)
            throw std::runtime_error(std::format("gemm_q: K = {} or alignment does not fit", s.k));
        l.k64 = l.k64 && s.k % BK2 == 0;
        l.a.s[l.a.nseg++] = s;
        tokens += s.t;
    }
    l.wide = l.k64 && wide_min() > 0 && l.a.nseg > 0 && tokens >= (int64_t)wide_min() * l.a.nseg;
    const int tt = l.wide ? 2 * QT : QT;
    for (int k = 0; k < l.a.nseg; k++) {
        l.a.tile0[k] = l.tiles;
        l.tiles += ((l.a.s[k].t + tt - 1) / tt) * ((l.a.s[k].n + BN - 1) / BN);
    }
    l.a.tile0[l.a.nseg] = l.tiles;
    l.next = i;
    return l;
}

// All launches of one gemm_q call with the table of one unit; false if the table has no kernel for the type
inline bool run_gemm_q(const QFormat* table, size_t n, uint32_t type, const GemmQSeg* segs, int nseg,
                       hipStream_t stream) {
    static const bool v1 = tona::env::flag("TONA_GEMMQ_V1");  // first generation only
    const QFormat* fmt = nullptr;
    for (size_t i = 0; i < n; i++)
        if (table[i].type == type) fmt = &table[i];
    if (!fmt) {  // an unsupported type is an error only if something would run
        for (int i = 0; i < nseg; i++)
            if (segs[i].t > 0 && segs[i].n > 0) return false;
        return true;
    }
    for (int i0 = 0; i0 < nseg;) {
        const QLaunch l = collect_q(segs, nseg, i0, !v1);
        i0 = l.next;
        if (l.tiles == 0) continue;
        fmt->launch(l.a, l.tiles, l.k64, l.wide, stream);
        HIPCHECK(hipGetLastError());
    }
    return true;
}

}  // namespace
}  // namespace tona::gpu
