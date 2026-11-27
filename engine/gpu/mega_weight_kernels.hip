// Kernels that prepare weights on the GPU, with their launch wrappers (gpu/mega_kernels.h). They are no part of the
// megakernel: a launch needs the kernel in the same translation unit (no -fgpu-rdc), so they live in a unit of their
// own, apart from the architectures' kernels.

#include <cmath>
#include <cstring>

#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_kernels.h"
#include "gpu/mega_util.h"

namespace tona::gpu {

// Q8_0 -> Q2_0 on the GPU, one thread per block of 64 (two Q8_0 blocks), the same method as quantize_q2_0 in float:
// d from 2 * 12 candidates (both signs), each with a least-squares refit. On the CPU this took ~7 min for the 512
// experts. Arguments see Q8ToQ2.

// Level 0 ... 3 of x for the scale dd
__device__ __forceinline__ int q2_level(float x, float dd) { return min(max((int)roundf(x / dd) + 1, 0), 3); }

// Dequantizes the two Q8_0 blocks at s into v and returns the largest magnitude
__device__ __forceinline__ float q2_load(const uint8_t* s, float (&v)[64]) {
    float amax = 0;
    for (int h = 0; h < 2; h++) {
        _Float16 dh;
        memcpy(&dh, s + h * 34, 2);
        const float dd = (float)dh;
        for (int k = 0; k < 32; k++) {
            v[h * 32 + k] = dd * (float)(int8_t)s[h * 34 + 2 + k];
            amax = fmaxf(amax, fabsf(v[h * 32 + k]));
        }
    }
    return amax;
}

// Scale and squared error of the candidate dd after one least-squares refit
struct Q2Fit {
    float d, err;
};
__device__ __forceinline__ Q2Fit q2_fit(const float (&v)[64], float dd) {
    for (int it = 0; it < 2; it++) {
        float xl = 0, ll = 0;
#pragma nounroll
        for (int k = 0; k < 64; k++) {
            const int l = q2_level(v[k], dd) - 1;
            xl += v[k] * l, ll += (float)(l * l);
        }
        if (it == 0 && ll > 0) dd = xl / ll;
    }
    const float dh = (float)(_Float16)dd;
    float e = 0;
    for (int k = 0; k < 64; k++) {
        const int l = q2_level(v[k], dd) - 1;
        const float rr = v[k] - l * dh;
        e += rr * rr;
    }
    return {dd, e};
}

// The candidate scale with the smallest error: both signs, 12 magnitudes each
__device__ __forceinline__ float q2_search(const float (&v)[64], float amax) {
    float best_d = 0, best_e = INFINITY;
    for (int sg = -1; sg <= 1 && amax > 0; sg += 2)
        for (int c = 1; c <= 12; c++) {
            const Q2Fit f = q2_fit(v, sg * amax * c / 18.0f);
            if (f.err < best_e) best_e = f.err, best_d = f.d;
        }
    return best_d;
}

// The sizes of Q8ToQ2; the pointers stay kernel parameters of their own (inside a struct they would be flat)
struct Q2Dims {
    int64_t nblk;
    int in, rows, e0;
    int64_t ebytes, rb;
};

__global__ void k_q8_to_q2(const uint8_t* src, uint8_t* dst, const Q2Dims a) {
    const int64_t b = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= a.nblk) return;
    const int per = a.in / 64;
    const int64_t r = b / per;
    const int j = (int)(b % per);
    float v[64];
    const float amax = q2_load(src + r * (a.in / 32 * 34) + (int64_t)j * 68, v);
    const float best_d = q2_search(v, amax);
    const int64_t e = a.e0 + r / a.rows, i = r % a.rows;
    uint8_t* o = dst + e * a.ebytes + i * a.rb + (int64_t)j * 18;
    const _Float16 h = (_Float16)best_d;
    memcpy(o, &h, 2);
    for (int q = 0; q < 16; q++) {
        uint8_t byte = 0;
        for (int k = 0; k < 4; k++) {
            const int l = best_d != 0 ? q2_level(v[4 * q + k], best_d) : 1;
            byte |= (uint8_t)(l << (2 * k));
        }
        o[2 + q] = byte;
    }
}

// Rows r0 ... of the truncated MTP head from the full LM head in VRAM (token per row from map), one block per row
__global__ void k_mtp_rows(const uint8_t* head, uint8_t* dst, const int32_t* map, int64_t r0, int row_bytes) {
    const int64_t r = r0 + blockIdx.x;
    copy_row(dst + r * row_bytes, head + (int64_t)map[r] * row_bytes, row_bytes, threadIdx.x, blockDim.x);
}

void launch_q8_to_q2(const Q8ToQ2& a) {
    hipLaunchKernelGGL(k_q8_to_q2, dim3((unsigned)((a.nblk + 255) / 256)), dim3(256), 0, 0, a.src, a.dst,
                       Q2Dims{a.nblk, a.in, a.rows, a.e0, a.ebytes, a.rb});
    HIPCHECK(hipGetLastError());
}

void launch_mtp_rows(const uint8_t* head, uint8_t* dst, const int32_t* map, int64_t r0, int row_bytes, unsigned count) {
    hipLaunchKernelGGL(k_mtp_rows, dim3(count), dim3(256), 0, 0, head, dst, map, r0, row_bytes);
}

}  // namespace tona::gpu
