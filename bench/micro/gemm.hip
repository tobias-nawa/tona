// f16 GEMM with WMMA (gfx12, wave32) for prefill: Y[T x N] = X[T x K] * W[N x K]^T, X and W in f16, Y in f32.
// W is row-major as in the GGUF (one output per row, K contiguous), X row-major per token.
// Measures TFLOPS for the shapes of the model and checks samples against a CPU reference.
// Build: $TONA_HIPCXX $TONA_HIPFLAGS -O3 bench/micro/gemm.hip -o work/build/gemm $TONA_HIPLDFLAGS
// Usage: work/build/gemm [T, default 4096]
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

// Tile per block: BT tokens x BN outputs, K in steps of BK. 8 waves as 2 (T) x 4 (N),
// per wave 64 x 32 = 4 x 2 WMMA tiles of 16 x 16.
constexpr int BT = 128, BN = 128, BK = 32, WT = 64, WN = 32;
constexpr int LDK = BK + 8;  // row pitch in LDS in halves (80 bytes), against bank conflicts

// Layout gfx12: lane l holds row (A) or column (B) l%16, K values (l/16)*8 .. +7.
// Result: lane l holds column l%16, rows (l/16)*8 .. +7.
__global__ __launch_bounds__(256) void k_gemm(const _Float16* __restrict__ X, const _Float16* __restrict__ W,
                                              float* __restrict__ Y, int T, int N, int K) {
    __shared__ __attribute__((aligned(16))) _Float16 sa[2][BT * LDK], sb[2][BN * LDK];
    const int tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    // Block order: per group of GT token tiles all output tiles, token tile fastest.
    // Blocks running at the same time then share ~1024 tokens of X and few rows of W (L2/Infinity Cache).
    constexpr int GT = 8;
    const int nt = T / BT, nn = N / BN, b = blockIdx.x;
    const int g = b / (GT * nn), gt = min(GT, nt - g * GT), in = b % (GT * nn);
    const int t0 = (g * GT + in % gt) * BT, n0 = (in / gt) * BN;
    const int wt = (wave >> 2) * WT, wn = (wave & 3) * WN;
    // Load: per thread 2 x 16 bytes from X and from W per K step (128 rows x 64 bytes)
    const int lr = tid >> 2, lc = (tid & 3) * 8;  // row 0..63, column 0/8/16/24
    const _Float16* xa = X + (size_t)(t0 + lr) * K + lc;
    const _Float16* wb = W + (size_t)(n0 + lr) * K + lc;
    const size_t x64 = (size_t)64 * K;
    int4v ra0, ra1, rb0, rb1;
    auto gload = [&](int k) {
        ra0 = *reinterpret_cast<const int4v*>(xa + k);
        ra1 = *reinterpret_cast<const int4v*>(xa + x64 + k);
        rb0 = *reinterpret_cast<const int4v*>(wb + k);
        rb1 = *reinterpret_cast<const int4v*>(wb + x64 + k);
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
    __syncthreads();
    const int r = lane & 15, kh = (lane >> 4) * 8;
    for (int k = 0, buf = 0; k < K; k += BK, buf ^= 1) {
        if (k + BK < K) gload(k + BK);
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            half8 a[4], b[2];
#pragma unroll
            for (int i = 0; i < 4; i++)
                a[i] = *reinterpret_cast<const half8*>(&sa[buf][(wt + i * 16 + r) * LDK + kk + kh]);
#pragma unroll
            for (int j = 0; j < 2; j++)
                b[j] = *reinterpret_cast<const half8*>(&sb[buf][(wn + j * 16 + r) * LDK + kk + kh]);
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 2; j++)
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a[i], b[j], acc[i][j]);
        }
        if (k + BK < K) sstore(buf ^ 1);
        __syncthreads();
    }
    // acc[i][j]: token rows wt + i*16 + kh + e, output column wn + j*16 + r
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
#pragma unroll
            for (int e = 0; e < 8; e++)
                Y[(size_t)(t0 + wt + i * 16 + kh + e) * N + n0 + wn + j * 16 + r] = acc[i][j][e];
}

int main(int argc, char** argv) {
    const int T = argc > 1 ? atoi(argv[1]) : 4096;
    struct Shape {
        const char* name;
        int N, K;
    };
    const Shape shapes[] = {{"attn_qkv 2560->10240", 10240, 2560}, {"attn_q 2560->12288", 12288, 2560},
                            {"attn_gate 2560->6144", 6144, 2560},  {"ssm_out 6144->2560", 2560, 6144},
                            {"expert up 2560->640", 640, 2560},    {"expert down 640->2560", 2560, 640}};
    hipEvent_t e0, e1;
    CHECK(hipEventCreate(&e0));
    CHECK(hipEventCreate(&e1));
    srand(1);
    for (const Shape& s : shapes) {
        const int N = s.N, K = s.K;
        std::vector<_Float16> hx((size_t)T * K), hw((size_t)N * K);
        for (auto& v : hx) v = (_Float16)((rand() % 2001 - 1000) / 1000.0f);
        for (auto& v : hw) v = (_Float16)((rand() % 2001 - 1000) / 1000.0f);
        _Float16 *dx, *dw;
        float* dy;
        CHECK(hipMalloc(&dx, hx.size() * 2));
        CHECK(hipMalloc(&dw, hw.size() * 2));
        CHECK(hipMalloc(&dy, (size_t)T * N * 4));
        CHECK(hipMemcpy(dx, hx.data(), hx.size() * 2, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dw, hw.data(), hw.size() * 2, hipMemcpyHostToDevice));
        const dim3 grid((N / BN) * (T / BT));
        k_gemm<<<grid, 256>>>(dx, dw, dy, T, N, K);
        CHECK(hipDeviceSynchronize());
        // samples against the CPU (sum in double)
        std::vector<float> hy((size_t)T * N);
        CHECK(hipMemcpy(hy.data(), dy, hy.size() * 4, hipMemcpyDeviceToHost));
        double maxerr = 0;
        for (int q = 0; q < 512; q++) {
            const int t = rand() % T, n = rand() % N;
            double ref = 0;
            for (int k = 0; k < K; k++) ref += (double)(float)hx[(size_t)t * K + k] * (float)hw[(size_t)n * K + k];
            maxerr = fmax(maxerr, fabs(ref - hy[(size_t)t * N + n]) / (fabs(ref) + 1));
        }
        const int reps = 20;
        std::vector<float> ms;
        for (int run = 0; run < 3; run++) {
            CHECK(hipEventRecord(e0));
            for (int i = 0; i < reps; i++) k_gemm<<<grid, 256>>>(dx, dw, dy, T, N, K);
            CHECK(hipEventRecord(e1));
            CHECK(hipEventSynchronize(e1));
            float t_ms;
            CHECK(hipEventElapsedTime(&t_ms, e0, e1));
            ms.push_back(t_ms / reps);
        }
        auto tf = [&](float m) { return 2.0 * T * N * K / (m * 1e-3) / 1e12; };
        float lo = ms[0], hi = ms[0];
        for (float m : ms) lo = fminf(lo, m), hi = fmaxf(hi, m);
        printf("%-24s T=%d: %6.3f ms, %5.1f TFLOPS (range %.1f-%.1f, n=3), error %.1e\n", s.name, T, lo, tf(lo), tf(hi),
               tf(lo), maxerr);
        CHECK(hipFree(dx));
        CHECK(hipFree(dw));
        CHECK(hipFree(dy));
    }
}
