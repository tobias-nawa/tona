// Toolchain smoke test: one WMMA instruction (gfx12, wave32) against a CPU reference.
// Build and run: scripts/setup.sh (after `source toolchain/env.sh`: $TONA_HIPCXX $TONA_HIPFLAGS ...)
#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>

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

// Layout gfx12: lane l holds row (A) or column (B) l%16, K half l/16.
// Result: lane l holds column l%16, rows (l/16)*8 .. +7.
__global__ void wmma16(const _Float16* A, const _Float16* B, float* D) {
    int l = threadIdx.x, r = l % 16, k0 = (l / 16) * 8;
    half8 a, b;
    for (int i = 0; i < 8; i++) {
        a[i] = A[r * 16 + k0 + i];    // A row-major, MxK
        b[i] = B[(k0 + i) * 16 + r];  // B row-major, KxN
    }
    float8 c = {};
    c = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
    for (int i = 0; i < 8; i++) D[(k0 + i) * 16 + r] = c[i];
}

int main() {
    _Float16 A[256], B[256];
    float D[256], ref[256];
    srand(1);
    for (int i = 0; i < 256; i++) {
        A[i] = (_Float16)(rand() % 7 - 3);
        B[i] = (_Float16)(rand() % 7 - 3);
    }
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 16; n++) {
            float s = 0;
            for (int k = 0; k < 16; k++) s += (float)A[m * 16 + k] * (float)B[k * 16 + n];
            ref[m * 16 + n] = s;
        }
    _Float16 *dA, *dB;
    float* dD;
    CHECK(hipMalloc(&dA, sizeof A));
    CHECK(hipMalloc(&dB, sizeof B));
    CHECK(hipMalloc(&dD, sizeof D));
    CHECK(hipMemcpy(dA, A, sizeof A, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(dB, B, sizeof B, hipMemcpyHostToDevice));
    wmma16<<<1, 32>>>(dA, dB, dD);
    CHECK(hipGetLastError());
    CHECK(hipMemcpy(D, dD, sizeof D, hipMemcpyDeviceToHost));
    // Headers and runtime must come from the same ROCm (Fedora installs its own headers in /usr/include/hip)
    int rt = 0;
    CHECK(hipRuntimeGetVersion(&rt));
    if (rt / 100000 != HIP_VERSION / 100000) {
        printf("headers HIP %d.%d, runtime HIP %d.%d: wrong headers on the include path\n", HIP_VERSION_MAJOR,
               HIP_VERSION_MINOR, rt / 10000000, rt / 100000 % 100);
        return 3;
    }
    int bad = 0;
    for (int i = 0; i < 256; i++) bad += D[i] != ref[i];
    hipDeviceProp_t p;
    CHECK(hipGetDeviceProperties(&p, 0));
    printf("%s (%s), HIP %d.%d: WMMA 16x16x16 f16->f32 %s, %d/256 wrong\n", p.name, p.gcnArchName, HIP_VERSION_MAJOR,
           HIP_VERSION_MINOR, bad ? "WRONG" : "correct", bad);
    return bad ? 1 : 0;
}
