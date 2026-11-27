// FP8 GEMM (E4M3, WMMA fp8, gfx12) and the f16 -> E4M3 conversion of its operands.
#include "backend/env.h"
#include "gpu/gemm_dev.h"

namespace tona::gpu {

namespace {

using int2v = int __attribute__((ext_vector_type(2)));
constexpr int BK8 = 64, LDK8 = BK8 + 16;  // bytes per K step; LDS row stride in bytes

struct Gemm8Args {
    Gemm8Seg s[GEMM_MAX_SEGS];
    int32_t tile0[GEMM_MAX_SEGS + 1];
    int32_t nseg;
};

// acc[i][j]: token rows wt + i*16 + kh + e, output column wn + j*16 + r; scaled by the row factors
__device__ __forceinline__ void store_tile8(const Gemm8Seg* S, const TileOrigin o, const WaveFrag f,
                                            const float8 (&acc)[4][2]) {
#pragma unroll
    for (int j = 0; j < 2; j++) {
        const int n = o.n0 + f.wn + j * 16 + f.r;
        if (n >= S->n) continue;
        const float swn = S->sw[n];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = o.t0 + f.wt + i * 16 + f.kh + e;
                if (t < S->t) S->y[(int64_t)t * S->ldy + n] = acc[i][j][e] * S->sx[t] * swn;
            }
    }
}

// Like k_gemm with 8-bit values: tile 128 x 128, K step 64, 8 waves 2 x 4 of 64 x 32. The fp8 WMMA layout is the
// f16 one: lane l holds row or column l % 16 and the K values (l / 16) * 8 ... + 7 (8 bytes).
__global__ __launch_bounds__(256) void k_gemm8(Gemm8Args a) {
    __shared__ __attribute__((aligned(16))) uint8_t sa[2][BT * LDK8], sb[2][BN * LDK8];
    const int b = blockIdx.x;
    int si = 0;  // segment of block b: the last one whose first tile is <= b
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const Gemm8Seg& S = a.s[si];
    const int T = S.t, N = S.n, K = S.k;
    const int nt = (T + BT - 1) / BT, nn = (N + BN - 1) / BN, loc = b - a.tile0[si];
    const TileOrigin o = tile_origin(loc, nt, nn, BT, BN);
    const int t0 = o.t0, n0 = o.n0;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int wt = (wave >> 2) * WT, wn = (wave & 3) * WN;
    const int lr = tid >> 2, lc = (tid & 3) * 16;
    const uint8_t* xa0 = S.x + (int64_t)min(t0 + lr, T - 1) * S.ldx + lc;
    const uint8_t* xa1 = S.x + (int64_t)min(t0 + lr + 64, T - 1) * S.ldx + lc;
    const uint8_t* wb0 = S.w + (int64_t)min(n0 + lr, N - 1) * S.ldw + lc;
    const uint8_t* wb1 = S.w + (int64_t)min(n0 + lr + 64, N - 1) * S.ldw + lc;
    int4v ra0, ra1, rb0, rb1;
    auto gload = [&](int k) {
        ra0 = *reinterpret_cast<const int4v*>(xa0 + k);
        ra1 = *reinterpret_cast<const int4v*>(xa1 + k);
        rb0 = *reinterpret_cast<const int4v*>(wb0 + k);
        rb1 = *reinterpret_cast<const int4v*>(wb1 + k);
    };
    auto sstore = [&](int buf) {
        *reinterpret_cast<int4v*>(&sa[buf][lr * LDK8 + lc]) = ra0;
        *reinterpret_cast<int4v*>(&sa[buf][(lr + 64) * LDK8 + lc]) = ra1;
        *reinterpret_cast<int4v*>(&sb[buf][lr * LDK8 + lc]) = rb0;
        *reinterpret_cast<int4v*>(&sb[buf][(lr + 64) * LDK8 + lc]) = rb1;
    };
    float8 acc[4][2] = {};
    gload(0);
    sstore(0);
    lds_sync();
    const int r = lane & 15, kh = (lane >> 4) * 8;
    for (int k = 0, buf = 0; k < K; k += BK8, buf ^= 1) {
        if (k + BK8 < K) gload(k + BK8);
#pragma unroll
        for (int kk = 0; kk < BK8; kk += 16) {
            int2v av[4], bv[2];
#pragma unroll
            for (int i = 0; i < 4; i++)
                av[i] = *reinterpret_cast<const int2v*>(&sa[buf][(wt + i * 16 + r) * LDK8 + kk + kh]);
#pragma unroll
            for (int j = 0; j < 2; j++)
                bv[j] = *reinterpret_cast<const int2v*>(&sb[buf][(wn + j * 16 + r) * LDK8 + kk + kh]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(av[i], bv[j], acc[i][j]);
        }
        if (k + BK8 < K) sstore(buf ^ 1);
        lds_sync();
    }
    store_tile8(&S, o, {wt, wn, r, kh}, acc);
}

// Second generation, like k_gemm2 of the f16 GEMM: tile 128 x BN2, K step 128 bytes in one LDS buffer (the loads of the
// next step go through registers while the WMMAs run), 8 waves 2 x 4 of 64 x BN2 / 4. A lane reads 16 bytes of a row
// per 32 values and feeds them to two WMMAs (128-bit LDS loads, GPUOpen WMMA guide part 2): the first WMMA gets
// bytes 0-7 of the 16, the second 8-15, for the operand A and B alike, so the sum over K is the same set of products.
constexpr int BK8B = 128, LDK8B = BK8B + 16;

template <int BN2>
__device__ __forceinline__ void store_tile8b(const Gemm8Seg* S, const TileOrigin o, const WaveFrag f,
                                             const float8 (&acc)[4][BN2 / 64]) {
#pragma unroll
    for (int j = 0; j < BN2 / 64; j++) {
        const int n = o.n0 + f.wn + j * 16 + f.r;
        if (n >= S->n) continue;
        const float swn = S->sw[n];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int t = o.t0 + f.wt + i * 16 + f.kh + e;
                if (t < S->t) S->y[(int64_t)t * S->ldy + n] = acc[i][j][e] * S->sx[t] * swn;
            }
    }
}

// The block's segment, tile and LDS tiles for the loads
template <int BN2>
struct G8Ctx {
    const Gemm8Seg* s;
    TileOrigin o;
    uint8_t *sa, *sb;
    int tid;
};
constexpr int CPR8 = BK8B / 16;  // 16-byte chunks per row and K step

template <int BN2>
__device__ __forceinline__ void g8_load(const G8Ctx<BN2> c, int k, int4v (&ra)[BT * CPR8 / 256],
                                        int4v (&rb)[BN2 * CPR8 / 256]) {
#pragma unroll
    for (int q = 0; q < BT * CPR8 / 256; q++) {
        const int e = c.tid + q * 256, row = e / CPR8, col = (e % CPR8) * 16;
        ra[q] = *reinterpret_cast<const int4v*>(c.s->x + (int64_t)min(c.o.t0 + row, c.s->t - 1) * c.s->ldx + k + col);
    }
#pragma unroll
    for (int q = 0; q < BN2 * CPR8 / 256; q++) {
        const int e = c.tid + q * 256, row = e / CPR8, col = (e % CPR8) * 16;
        rb[q] = *reinterpret_cast<const int4v*>(c.s->w + (int64_t)min(c.o.n0 + row, c.s->n - 1) * c.s->ldw + k + col);
    }
}

template <int BN2>
__device__ __forceinline__ void g8_store(const G8Ctx<BN2> c, const int4v (&ra)[BT * CPR8 / 256],
                                         const int4v (&rb)[BN2 * CPR8 / 256]) {
#pragma unroll
    for (int q = 0; q < BT * CPR8 / 256; q++) {
        const int e = c.tid + q * 256;
        *reinterpret_cast<int4v*>(&c.sa[(e / CPR8) * LDK8B + (e % CPR8) * 16]) = ra[q];
    }
#pragma unroll
    for (int q = 0; q < BN2 * CPR8 / 256; q++) {
        const int e = c.tid + q * 256;
        *reinterpret_cast<int4v*>(&c.sb[(e / CPR8) * LDK8B + (e % CPR8) * 16]) = rb[q];
    }
}

// One K step of 128 bytes: per 32 bytes every lane loads 16 bytes of each fragment row and feeds two WMMAs
template <int BN2>
__device__ __forceinline__ void g8_mma(const uint8_t* sa, const uint8_t* sb, const WaveFrag f, int lk,
                                       float8 (&acc)[4][BN2 / 64]) {
    constexpr int FJ = BN2 / 64;
#pragma unroll
    for (int kk = 0; kk < BK8B; kk += 32) {
        int4v av[4], bv[FJ];
#pragma unroll
        for (int i = 0; i < 4; i++)
            av[i] = *reinterpret_cast<const int4v*>(&sa[(f.wt + i * 16 + f.r) * LDK8B + kk + lk]);
#pragma unroll
        for (int j = 0; j < FJ; j++)
            bv[j] = *reinterpret_cast<const int4v*>(&sb[(f.wn + j * 16 + f.r) * LDK8B + kk + lk]);
#pragma unroll
        for (int h = 0; h < 2; h++)
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < FJ; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(
                        h ? int2v{av[i][2], av[i][3]} : int2v{av[i][0], av[i][1]},
                        h ? int2v{bv[j][2], bv[j][3]} : int2v{bv[j][0], bv[j][1]}, acc[i][j]);
    }
}

template <int BN2>
__global__ __launch_bounds__(256) void k_gemm8b(Gemm8Args a) {
    __shared__ __attribute__((aligned(16))) uint8_t sa[BT * LDK8B], sb[BN2 * LDK8B];
    const int b = blockIdx.x;
    int si = 0;
    while (si + 1 < a.nseg && b >= a.tile0[si + 1]) si++;
    const Gemm8Seg& S = a.s[si];
    const int nt = (S.t + BT - 1) / BT, nn = (S.n + BN2 - 1) / BN2, loc = b - a.tile0[si];
    const TileOrigin o = tile_origin(loc, nt, nn, BT, BN2);
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const WaveFrag f{(wave >> 2) * WT, (wave & 3) * (BN2 / 4), lane & 15, (lane >> 4) * 8};
    const G8Ctx<BN2> c{&S, o, sa, sb, tid};
    int4v ra[BT * CPR8 / 256], rb[BN2 * CPR8 / 256];
    float8 acc[4][BN2 / 64] = {};
    g8_load<BN2>(c, 0, ra, rb);
    g8_store<BN2>(c, ra, rb);
    lds_sync();
    for (int k = 0; k < S.k; k += BK8B) {
        const bool more = k + BK8B < S.k;
        if (more) g8_load<BN2>(c, k + BK8B, ra, rb);
        g8_mma<BN2>(sa, sb, f, (lane >> 4) * 16, acc);
        if (more) {
            lds_sync();
            g8_store<BN2>(c, ra, rb);
        }
        lds_sync();
    }
    store_tile8b<BN2>(&S, o, f, acc);
}

// Block per row: absolute maximum, then a packed pair of E4M3 per 2 values (saturating)
__global__ __launch_bounds__(256) void k_to_fp8(const _Float16* in, int64_t ld_in, int64_t k, uint8_t* out,
                                                int64_t ld_out, float* scale) {
    __shared__ float red[8];
    const _Float16* x = in + (int64_t)blockIdx.x * ld_in;
    float mx = 0;
    for (int64_t i = threadIdx.x; i < k; i += 256) mx = fmaxf(mx, fabsf((float)x[i]));
    for (int o = 16; o > 0; o >>= 1) mx = fmaxf(mx, __shfl_xor(mx, o, 32));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = mx;
    lds_sync();
    mx = 0;
    for (int i = 0; i < 8; i++) mx = fmaxf(mx, red[i]);
    const float sc = mx > 0 ? mx / 448.0f : 1.0f, inv = 1.0f / sc;
    if (threadIdx.x == 0) scale[blockIdx.x] = sc;
    uint32_t* o = reinterpret_cast<uint32_t*>(out + (int64_t)blockIdx.x * ld_out);
    for (int64_t i = threadIdx.x; i < k / 4; i += 256) {
        int p = __builtin_amdgcn_cvt_pk_fp8_f32((float)x[4 * i] * inv, (float)x[4 * i + 1] * inv, 0, false);
        p = __builtin_amdgcn_cvt_pk_fp8_f32((float)x[4 * i + 2] * inv, (float)x[4 * i + 3] * inv, p, true);
        o[i] = (uint32_t)p;
    }
}

// Up to GEMM_MAX_SEGS non-empty segments starting at segs[first]; whether all allow the K128 kernel and BN 256
struct Fp8Launch {
    Gemm8Args a{};
    bool k128 = true, wide = true;
    int tiles = 0, next = 0;
};

Fp8Launch collect_fp8(const Gemm8Seg* segs, int nseg, int first, bool allow_k128) {
    Fp8Launch l;
    l.k128 = allow_k128;
    int i = first;  // empty segments do not count: continue at i
    for (; i < nseg && l.a.nseg < GEMM_MAX_SEGS; i++) {
        const Gemm8Seg& s = segs[i];
        if (s.t <= 0 || s.n <= 0) continue;
        if (s.k % BK8 || s.ldx % 16 || s.ldw % 16 ||
            (reinterpret_cast<uintptr_t>(s.x) | reinterpret_cast<uintptr_t>(s.w)) % 16)
            throw std::runtime_error(std::format("gemm_fp8: K = {} or alignment does not fit", s.k));
        l.k128 = l.k128 && s.k % BK8B == 0;
        l.wide = l.wide && s.n >= 4096;
        l.a.s[l.a.nseg++] = s;
    }
    l.next = i;
    // Kernel per launch: K step 128 with BN 256 for wide outputs (all N >= 4096) if all segments allow it
    const int bn = !l.k128 ? BN : l.wide ? 256 : BN;
    for (int k = 0; k < l.a.nseg; k++) {
        l.a.tile0[k] = l.tiles;
        l.tiles += ((l.a.s[k].t + BT - 1) / BT) * ((l.a.s[k].n + bn - 1) / bn);
    }
    l.a.tile0[l.a.nseg] = l.tiles;
    return l;
}

}  // namespace

void gemm_fp8(const Gemm8Seg* segs, int nseg, hipStream_t stream) {
    static const bool v1 = tona::env::flag("TONA_FP8_V1");  // first generation only
    for (int i0 = 0; i0 < nseg;) {
        const Fp8Launch l = collect_fp8(segs, nseg, i0, !v1);
        i0 = l.next;
        if (l.tiles == 0) continue;
        if (!l.k128)
            k_gemm8<<<l.tiles, 256, 0, stream>>>(l.a);
        else if (l.wide)
            k_gemm8b<256><<<l.tiles, 256, 0, stream>>>(l.a);
        else
            k_gemm8b<128><<<l.tiles, 256, 0, stream>>>(l.a);
        HIPCHECK(hipGetLastError());
    }
}

void to_fp8_rows(const Fp8Rows& r, hipStream_t stream) {
    if (r.k % 4 || r.ld_out % 4) throw std::runtime_error("to_fp8_rows: k and ld_out must be multiples of 4");
    if (r.rows <= 0) return;
    k_to_fp8<<<(unsigned)r.rows, 256, 0, stream>>>(r.in, r.ld_in, r.k, r.out, r.ld_out, r.scale);
    HIPCHECK(hipGetLastError());
}

}  // namespace tona::gpu
