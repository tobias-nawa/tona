// Variants of the f16 GEMM (WMMA, gfx12) for prefill, compared: Y[T x N] = X[T x K] * W[N x K]^T.
// Shapes as in prefill (T = 2,048). Checks each variant against the first one (the same summation order in K per
// tile is not guaranteed, so the tolerance is relative).
// Build: $TONA_HIPCXX $TONA_HIPFLAGS -O3 bench/micro/gemm2.hip -o work/build/gemm2 $TONA_HIPLDFLAGS
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                                      \
    do {                                                                              \
        hipError_t e_ = (x);                                                          \
        if (e_ != hipSuccess) {                                                       \
            printf("%s:%d: %s: %s\n", __FILE__, __LINE__, #x, hipGetErrorString(e_)); \
            exit(2);                                                                  \
        }                                                                             \
    } while (0)

typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef float float8 __attribute__((ext_vector_type(8)));
typedef int int4v __attribute__((ext_vector_type(4)));

// Cooperative global -> LDS tile loader in chunks of 8 halves: BT*BK/8 resp. BN*BK/8 chunks spread over NT threads.
// The chunks go through registers (ra, rb) so the next global load overlaps the math of the current tile.
template <int BT, int BN, int BK, int NT>
struct TileLoader {
    static constexpr int CPR = BK / 8, LA = BT * CPR / NT, LB = BN * CPR / NT, LDK = BK + 8;
    const _Float16 *x, *w;
    int t0, n0, K, tid;  // first rows of this block in X and W
    int4v ra[LA], rb[LB];

    __device__ void gload(int k) {
#pragma unroll
        for (int q = 0; q < LA; q++) {
            const int c = tid + q * NT, row = c / CPR, col = (c % CPR) * 8;
            ra[q] = *reinterpret_cast<const int4v*>(x + (size_t)(t0 + row) * K + k + col);
        }
#pragma unroll
        for (int q = 0; q < LB; q++) {
            const int c = tid + q * NT, row = c / CPR, col = (c % CPR) * 8;
            rb[q] = *reinterpret_cast<const int4v*>(w + (size_t)(n0 + row) * K + k + col);
        }
    }
    __device__ void sstore(_Float16* sa, _Float16* sb) {
#pragma unroll
        for (int q = 0; q < LA; q++) {
            const int c = tid + q * NT, row = c / CPR, col = (c % CPR) * 8;
            *reinterpret_cast<int4v*>(&sa[row * LDK + col]) = ra[q];
        }
#pragma unroll
        for (int q = 0; q < LB; q++) {
            const int c = tid + q * NT, row = c / CPR, col = (c % CPR) * 8;
            *reinterpret_cast<int4v*>(&sb[row * LDK + col]) = rb[q];
        }
    }
};

// Where a lane sits in the block tile: wave offset (wt, wn) in rows/columns, lane row r, K half kh.
struct WaveCoord {
    int wt, wn, r, kh;
};

// One 16-wide K slice: FI x FJ WMMA tiles per wave.
// MODE 2 skips the LDS reads and uses fixed fragments, to measure the math alone.
template <int FI, int FJ, int LDK, int MODE>
__device__ inline void mma_k16(float8 (&acc)[FI][FJ], const _Float16* sa, const _Float16* sb, WaveCoord c, int kk,
                               int k) {
    half8 av[FI], bv[FJ];
    if constexpr (MODE == 2) {
#pragma unroll
        for (int i = 0; i < FI; i++) av[i] = (half8)(_Float16)(kk + i);
#pragma unroll
        for (int j = 0; j < FJ; j++) bv[j] = (half8)(_Float16)(k + j);
    } else {
#pragma unroll
        for (int i = 0; i < FI; i++)
            av[i] = *reinterpret_cast<const half8*>(&sa[(c.wt + i * 16 + c.r) * LDK + kk + c.kh]);
#pragma unroll
        for (int j = 0; j < FJ; j++)
            bv[j] = *reinterpret_cast<const half8*>(&sb[(c.wn + j * 16 + c.r) * LDK + kk + c.kh]);
    }
#pragma unroll
    for (int i = 0; i < FI; i++)
#pragma unroll
        for (int j = 0; j < FJ; j++)
            acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av[i], bv[j], acc[i][j]);
}

// Writes the accumulators; yb points at this lane's first output element.
template <int FI, int FJ>
__device__ inline void store_acc(float* yb, const float8 (&acc)[FI][FJ], int N) {
#pragma unroll
    for (int i = 0; i < FI; i++)
#pragma unroll
        for (int j = 0; j < FJ; j++)
#pragma unroll
            for (int e = 0; e < 8; e++) yb[(size_t)(i * 16 + e) * N + j * 16] = acc[i][j][e];
}

// BT x BN tile, BK per step, WR x WC waves, per wave (BT/WR) x (BN/WC); DB: double LDS buffer.
// MODE 0: normal; 1: no loads from memory in the loop; 2: no LDS reads in the loop (fixed fragments).
template <int BT, int BN, int BK, int WR, int WC, bool DB, int MODE = 0>
__global__ __launch_bounds__(WR * WC * 32) void k_gemm(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                                       float* __restrict__ Y, int T, int N, int K) {
    constexpr int NT = WR * WC * 32, LDK = BK + 8, WT = BT / WR, WN = BN / WC, FI = WT / 16, FJ = WN / 16;
    constexpr int NBUF = DB ? 2 : 1;
    __shared__ __attribute__((aligned(16))) _Float16 sa[NBUF][BT * LDK], sb[NBUF][BN * LDK];
    constexpr int GT = 8;  // tile rows per group: consecutive blocks share W columns and X rows in L2
    const int nt = T / BT, nn = N / BN, b = blockIdx.x;
    const int g = b / (GT * nn), gt = min(GT, nt - g * GT), in = b % (GT * nn);
    const int t0 = (g * GT + in % gt) * BT, n0 = (in / gt) * BN;
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const WaveCoord c{(wave / WC) * WT, (wave % WC) * WN, lane & 15, (lane >> 4) * 8};
    TileLoader<BT, BN, BK, NT> ld{X, W, t0, n0, K, tid, {}, {}};
    float8 acc[FI][FJ] = {};
    ld.gload(0);
    ld.sstore(sa[0], sb[0]);
    __syncthreads();
    for (int k = 0, buf = 0; k < K; k += BK) {
        if (MODE != 1 && k + BK < K) ld.gload(k + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) mma_k16<FI, FJ, LDK, MODE>(acc, sa[buf], sb[buf], c, kk, k);
        if (k + BK < K) {
            if constexpr (DB) {
                ld.sstore(sa[buf ^ 1], sb[buf ^ 1]);
                buf ^= 1;
            } else {
                __syncthreads();
                ld.sstore(sa[0], sb[0]);
            }
        }
        __syncthreads();
    }
    store_acc<FI, FJ>(Y + (size_t)(t0 + c.wt + c.kh) * N + n0 + c.wn + c.r, acc, N);
}

struct Var {
    const char* name;
    void (*launch)(const _Float16*, const _Float16*, float*, int, int, int);
};
template <int BT, int BN, int BK, int WR, int WC, bool DB, int MODE = 0>
void launch(const _Float16* x, const _Float16* w, float* y, int T, int N, int K) {
    k_gemm<BT, BN, BK, WR, WC, DB, MODE><<<(T / BT) * (N / BN), WR * WC * 32>>>(x, w, y, T, N, K);
}

static int REPS = 10;
int main(int argc, char** argv) {
    if (argc > 1) REPS = atoi(argv[1]);
    const int T = 2048;
    struct Shape {
        const char* name;
        int N, K;
    };
    const Shape shapes[] = {{"qkv 2560->10240", 10240, 2560}, {"ssm_out 6144->2560", 2560, 6144}};
    const Var vars[] = {
        {"current 128x128 K32 W2x4 (64x32) DB", launch<128, 128, 32, 2, 4, true>},
        {"128x128 K64 W2x4 single", launch<128, 128, 64, 2, 4, false>},
        {"128x256 K32 W2x4 (64x64) DB", launch<128, 256, 32, 2, 4, true>},
        {"256x128 K32 W4x2 (64x64) DB", launch<256, 128, 32, 4, 2, true>},
        {"128x256 K64 W2x4 (64x64) single", launch<128, 256, 64, 2, 4, false>},
        {"128x128 K32 W2x2 (64x64) DB", launch<128, 128, 32, 2, 2, true>},
        {"256x256 K32 W4x4 (64x64) single", launch<256, 256, 32, 4, 4, false>},
        {"128x256 K64 no memory loads", launch<128, 256, 64, 2, 4, false, 1>},
        {"128x256 K64 no LDS reads", launch<128, 256, 64, 2, 4, false, 2>},
        {"current, no memory loads", launch<128, 128, 32, 2, 4, true, 1>},
        {"current, no LDS reads", launch<128, 128, 32, 2, 4, true, 2>},
    };
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    for (const Shape& s : shapes) {
        const int N = s.N, K = s.K;
        std::vector<_Float16> hx((size_t)T * K), hw((size_t)N * K);
        srand(1);
        for (auto& v : hx) v = (_Float16)((rand() % 2001 - 1000) / 1000.0f);
        for (auto& v : hw) v = (_Float16)((rand() % 2001 - 1000) / 1000.0f);
        _Float16 *dx, *dw;
        float *dy, *dref;
        CHECK(hipMalloc(&dx, hx.size() * 2));
        CHECK(hipMalloc(&dw, hw.size() * 2));
        CHECK(hipMalloc(&dy, (size_t)T * N * 4));
        CHECK(hipMalloc(&dref, (size_t)T * N * 4));
        CHECK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        vars[0].launch(dx, dw, dref, T, N, K);
        std::vector<float> ref((size_t)T * N), got(ref.size());
        CHECK(hipMemcpy(ref.data(), dref, ref.size() * 4, hipMemcpyDeviceToHost));
        printf("%s, T = %d\n", s.name, T);
        for (const Var& v : vars) {
            CHECK(hipMemset(dy, 0, (size_t)T * N * 4));
            v.launch(dx, dw, dy, T, N, K);
            CHECK(hipGetLastError());
            CHECK(hipMemcpy(got.data(), dy, got.size() * 4, hipMemcpyDeviceToHost));
            double err = 0;
            for (size_t i = 0; i < got.size(); i++) err = fmax(err, fabs(got[i] - ref[i]) / (fabs(ref[i]) + 1));
            float best = 1e9;
            for (int run = 0; run < 3; run++) {
                CHECK(hipEventRecord(e0));
                for (int i = 0; i < REPS; i++) v.launch(dx, dw, dy, T, N, K);
                CHECK(hipEventRecord(e1));
                CHECK(hipEventSynchronize(e1));
                float ms;
                CHECK(hipEventElapsedTime(&ms, e0, e1));
                best = fminf(best, ms / REPS);
            }
            printf("  %-36s %7.3f ms  %5.1f TFLOPS  dev. %.1e\n", v.name, best, 2.0 * T * N * K / (best * 1e-3) / 1e12,
                   err);
        }
        CHECK(hipFree(dx));
        CHECK(hipFree(dw));
        CHECK(hipFree(dy));
        CHECK(hipFree(dref));
    }
}
