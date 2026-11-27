// int8 path: unpacking of NVFP4 (type 40) into a Dec. A block of 64 weights is 36 bytes: four UE4M3 scale bytes, one
// per sub-block of 16, then 32 bytes of 4-bit E2M1 codes. In a sub-block of 16, byte j holds weight j in its low and
// weight j + 8 in its high nibble. The codes are the E2M1 values doubled (0, 1, 2, 3, 4, 6, 8, 12 and their negatives,
// as kvalues_mxfp4), small integers, so the dot product with int8 x is exact and the scale (halved, because the values
// are doubled) is applied per 16 weights, as the two halves of Dec. gfx1200 has no FP4 instruction. The per-tensor F32
// ".scale" of the file is not part of the row: the caller applies it after the matvec. Part of matvec_dev.h; do not
// include on its own.
#pragma once

#include "gpu/matvec_q8_core.h"

namespace tona::gpu {
namespace {

// kvalues_mxfp4 as the four words of lut16 (entry i in byte i)
constexpr int kFp4Lut0 = 0x03020100, kFp4Lut1 = 0x0C080604, kFp4Lut2 = (int)0xFDFEFF00u, kFp4Lut3 = (int)0xF4F8FAFCu;

// UE4M3 byte (unsigned, 4 exponent bits with bias 7, 3 mantissa bits) halved; 0 and 0x7F (NaN) give 0. Both branches
// are exact in float, so the value equals ggml_ue4m3_to_fp32 and the CPU dequantization bit for bit.
__device__ __forceinline__ float ue4m3_half(uint32_t x) {
    const int e = (x >> 3) & 0xF, m = x & 7;
    const float sub = (float)m * 0x1p-10f;                                               // e = 0: m / 8 * 2^-6 / 2
    const float nrm = (float)(8 + m) * __uint_as_float((uint32_t)(e - 11 + 127) << 23);  // (1 + m / 8) * 2^(e-7) / 2
    return x == 0x7F ? 0.0f : (e == 0 ? sub : nrm);
}

// One sub-block of 16 weights from its 8 code bytes (lo: bytes 0..3, hi: bytes 4..7) into 4 words
__device__ __forceinline__ void fp4_words(int lo, int hi, int* w) {
    w[0] = lut16(lo & 0x0F0F0F0F, kFp4Lut0, kFp4Lut1, kFp4Lut2, kFp4Lut3);
    w[1] = lut16(hi & 0x0F0F0F0F, kFp4Lut0, kFp4Lut1, kFp4Lut2, kFp4Lut3);
    w[2] = lut16((lo >> 4) & 0x0F0F0F0F, kFp4Lut0, kFp4Lut1, kFp4Lut2, kFp4Lut3);
    w[3] = lut16((hi >> 4) & 0x0F0F0F0F, kFp4Lut0, kFp4Lut1, kFp4Lut2, kFp4Lut3);
}

template <>
__device__ __forceinline__ void decode_q8<40>(const uint8_t* row, int sub, const Tabs&, Dec& o) {
    const uint8_t* b = row + (sub / 2) * 36;
    const int h = sub & 1;  // the sub-block of 32 is the sub-blocks of 16 numbered 2h and 2h + 1 of the block
    const uint8_t* q = b + 4 + 16 * h;
    fp4_words(ld4(q), ld4(q + 4), o.w);
    fp4_words(ld4(q + 8), ld4(q + 12), o.w + 4);
    o.s_lo = ue4m3_half(b[2 * h]);
    o.s_hi = ue4m3_half(b[2 * h + 1]);
}

}  // namespace
}  // namespace tona::gpu
