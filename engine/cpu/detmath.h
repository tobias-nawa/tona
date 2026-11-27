// Math that CPU and GPU must execute bit-identically (an expert is computed by one side or the other depending on
// the cache state). Only basic IEEE operations in a fixed order, fma explicit; nothing may be contracted (GPU: the
// pragma below, CPU: -ffp-contract=off for the including file). The vector version in cpu/experts.cpp performs the
// same steps.
#pragma once

#include <cmath>
#include <cstdint>

#if defined(__HIPCC__)
#define TONA_HD __host__ __device__
#else
#define TONA_HD
#endif

namespace tona::det {

// e^x: x = k ln 2 + r, |r| <= ln 2 / 2, polynomial after Cephes expf (e^r = 1 + r + r^2 p(r), ~1 ulp), result
// times 2^k through the exponent bits
inline constexpr float kL2e = 1.44269504f, kLn2Hi = 0.693359375f, kLn2Lo = -2.12194440e-4f;
inline constexpr float kP[6] = {1.9875691500e-4f, 1.3981999507e-3f, 8.3334519073e-3f,
                                4.1665795894e-2f, 1.6666665459e-1f, 5.0000001201e-1f};

TONA_HD inline float exp(float x) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    if (!(x < 88.0f)) return x != x ? x : INFINITY;
    if (x < -87.0f) return 0.0f;
    const float k = rintf(x * kL2e);
    float r = fmaf(-k, kLn2Hi, x);
    r = fmaf(-k, kLn2Lo, r);
    float p = kP[0];
    for (int i = 1; i < 6; i++) p = fmaf(p, r, kP[i]);
    const float y = fmaf(p, r * r, r) + 1.0f;
    return y * __builtin_bit_cast(float, ((int32_t)k + 127) << 23);
}

TONA_HD inline float silu(float x) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    return x / (1.0f + exp(-x));
}

// Sum over the experts in fixed point (2^-36): addition is then associative, so the sum does not depend on which side
// computes which expert or in which order. Range +-2^26 per summand (outputs are far below).
inline constexpr float kFix = 0x1p36f;
TONA_HD inline int64_t to_fix(float v) {
#if defined(__clang__)
#pragma clang fp contract(off)
#endif
    const float s = fminf(fmaxf(v * kFix, -0x1p62f), 0x1p62f);
    return (int64_t)rintf(s);
}
TONA_HD inline float from_fix(int64_t a) { return (float)((double)a * 0x1p-36); }

}  // namespace tona::det
