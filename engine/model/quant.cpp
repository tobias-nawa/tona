// Ported from ggml/src/ggml-quants.c (llama.cpp b11822, MIT). Expressions keep the order used there; compiled with
// -ffp-contract=off, otherwise the compiler fuses a*b-c into an FMA and the last bit differs.
#include "model/quant.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <stdexcept>

#include "model/quant_tables.h"

namespace tona {

using namespace tables;

namespace {

constexpr int QK_K = 256;

struct BlockQ8_0 {
    uint16_t d;
    int8_t qs[32];
};
struct BlockQ2_0 {
    uint16_t d;
    uint8_t qs[16];
};
struct BlockQ4K {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qs[QK_K / 2];
};
struct BlockQ5K {
    uint16_t d, dmin;
    uint8_t scales[12];
    uint8_t qh[QK_K / 8];
    uint8_t qs[QK_K / 2];
};
struct BlockQ6K {
    uint8_t ql[QK_K / 2];
    uint8_t qh[QK_K / 4];
    int8_t scales[QK_K / 16];
    uint16_t d;
};
struct BlockIQ2XXS {
    uint16_t d;
    uint16_t qs[QK_K / 8];
};
struct BlockIQ2S {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t scales[QK_K / 32];
};
struct BlockIQ3XXS {
    uint16_t d;
    uint8_t qs[3 * QK_K / 8];
};
struct BlockIQ3S {
    uint16_t d;
    uint8_t qs[QK_K / 4];
    uint8_t qh[QK_K / 32];
    uint8_t signs[QK_K / 8];
    uint8_t scales[QK_K / 64];
};
struct BlockIQ4NL {
    uint16_t d;
    uint8_t qs[16];
};
// NVFP4: 64 weights as 4 sub-blocks of 16, each with a UE4M3 scale byte and 4-bit E2M1 codes (the tensor's own F32
// ".scale" is applied by the caller, as in llama.cpp)
struct BlockNVFP4 {
    uint8_t d[4];
    uint8_t qs[32];
};
struct BlockIQ4XS {
    uint16_t d;
    uint16_t scales_h;
    uint8_t scales_l[QK_K / 64];
    uint8_t qs[QK_K / 2];
};

static_assert(sizeof(BlockQ8_0) == 34 && sizeof(BlockQ2_0) == 18 && sizeof(BlockQ4K) == 144);
static_assert(sizeof(BlockQ5K) == 176 && sizeof(BlockQ6K) == 210 && sizeof(BlockIQ2S) == 82);
static_assert(sizeof(BlockIQ3XXS) == 98 && sizeof(BlockIQ3S) == 110 && sizeof(BlockIQ4NL) == 18);
static_assert(sizeof(BlockIQ4XS) == 136 && sizeof(BlockIQ2XXS) == 66 && sizeof(BlockNVFP4) == 36);

void get_scale_min_k4(int j, const uint8_t* q, uint8_t* d, uint8_t* m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void deq_q8_0(const BlockQ8_0* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / 32; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 32; ++j) y[i * 32 + j] = x[i].qs[j] * d;
    }
}

void deq_q2_0(const BlockQ2_0* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / 64; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 64; ++j) {
            const uint8_t q = (x[i].qs[j / 4] >> ((j % 4) * 2)) & 0x03;
            y[i * 64 + j] = ((int)q - 1) * d;  // 00=-1, 01=0, 10=+1, 11=+2
        }
    }
}

void deq_q4_k(const BlockQ4K* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / QK_K; i++) {
        const uint8_t* q = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l] >> 4) - m2;
            q += 32;
            is += 2;
        }
    }
}

void deq_q5_k(const BlockQ5K* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / QK_K; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const float d = fp16_to_fp32(x[i].d);
        const float min = fp16_to_fp32(x[i].dmin);
        int is = 0;
        uint8_t sc, m;
        uint8_t u1 = 1, u2 = 2;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc, m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc, m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l] >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

void deq_q6_k(const BlockQ6K* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / QK_K; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t* sc = x[i].scales;
        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                const int8_t q1 = (int8_t)((ql[l + 0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

// E2M1 values doubled (kvalues_fp4 in ggml-common.h): the code is sign (bit 3), exponent (bits 1-2), mantissa (bit 0)
constexpr int8_t kFp4Doubled[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// UE4M3 scale byte as ggml_ue4m3_to_fp32: unsigned, 4 exponent bits (bias 7), 3 mantissa bits; 0 and 0x7F (NaN) give 0.
// Halved, because the code values above are doubled.
float ue4m3_half(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exp = (x >> 3) & 0xF, man = x & 0x7;
    const float raw = exp == 0 ? std::ldexp((float)man, -9) : std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    return raw * 0.5f;
}

// Within a sub-block of 16, byte j holds weight j in its low and weight j + 8 in its high nibble
void deq_nvfp4(const BlockNVFP4* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / 64; i++) {
        for (int64_t s = 0; s < 4; s++) {
            const float d = ue4m3_half(x[i].d[s]);
            float* yb = y + i * 64 + s * 16;
            for (int j = 0; j < 8; j++) {
                const uint8_t q = x[i].qs[s * 8 + j];
                yb[j] = kFp4Doubled[q & 0x0F] * d;
                yb[j + 8] = kFp4Doubled[q >> 4] * d;
            }
        }
    }
}

void deq_iq2_xxs(const BlockIQ2XXS* x, float* y, int64_t k) {
    uint32_t aux32[2];
    const uint8_t* aux8 = (const uint8_t*)aux32;
    for (int64_t i = 0; i < k / QK_K; i++) {
        const float d = fp16_to_fp32(x[i].d);
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(aux32, x[i].qs + 4 * (ptrdiff_t)ib32, 2 * sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t* grid = (const uint8_t*)(iq2xxs_grid + aux8[l]);
                const uint8_t signs = ksigns_iq2xs[(aux32[1] >> 7 * l) & 127];
                for (int j = 0; j < 8; ++j) y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
        }
    }
}

void deq_iq2_s(const BlockIQ2S* x, float* y, int64_t k) {
    float db[2];
    for (int64_t i = 0; i < k / QK_K; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = qs + QK_K / 8;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            db[0] = d * (0.5f + (x[i].scales[ib32] & 0xf)) * 0.25f;
            db[1] = d * (0.5f + (x[i].scales[ib32] >> 4)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const float dl = db[l / 2];
                const uint8_t* grid = (const uint8_t*)(iq2s_grid + (qs[l] | (qh[ib32] << (8 - 2 * l) & 0x300)));
                for (int j = 0; j < 8; ++j) y[j] = dl * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f);
                y += 8;
            }
            qs += 4;
            signs += 4;
        }
    }
}

void deq_iq3_xxs(const BlockIQ3XXS* x, float* y, int64_t k);

// TONA_IQ3_XXS_R (docs/use/tona-types.md): a 196-byte super-block holds two IQ3_XXS blocks as index bytes (0..127),
// sign and scale words (128..191) and block scales (192..195); rebuilt into the blocks, so the weights are IQ3_XXS's
// bits
void deq_iq3_xxs_r(const uint8_t* x, float* y, int64_t k) {
    BlockIQ3XXS b[2];
    for (int64_t s = 0; s < k / 512; s++, x += 196, y += 512) {
        for (ptrdiff_t i = 0; i < 2; i++) {
            std::memcpy(&b[i].d, x + 192 + 2 * i, 2);
            std::memcpy(b[i].qs, x + 64 * i, 64);
            std::memcpy(b[i].qs + 64, x + 128 + 32 * i, 32);
        }
        deq_iq3_xxs(b, y, 512);
    }
}

void deq_iq3_xxs(const BlockIQ3XXS* x, float* y, int64_t k) {
    uint32_t aux32;
    for (int64_t i = 0; i < k / QK_K; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* scales_and_signs = qs + QK_K / 4;
        for (int ib32 = 0; ib32 < QK_K / 32; ++ib32) {
            std::memcpy(&aux32, scales_and_signs + 4 * (int64_t)ib32, sizeof(uint32_t));
            const float db = d * (0.5f + (aux32 >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * l) & 127];
                const uint8_t* grid1 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 0]);
                const uint8_t* grid2 = (const uint8_t*)(iq3xxs_grid + qs[2 * l + 1]);
                for (int j = 0; j < 4; ++j) {
                    y[j + 0] = db * grid1[j] * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f);
                    y[j + 4] = db * grid2[j] * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
        }
    }
}

// 32 values: 4 groups of 8 with scale db
void deq_iq3_s_half(float db, const uint8_t* qs, uint8_t qh, const uint8_t* signs, float* y) {
    for (int l = 0; l < 4; ++l) {
        const uint8_t* grid1 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 0] | ((qh << (8 - 2 * l)) & 256)));
        const uint8_t* grid2 = (const uint8_t*)(iq3s_grid + (qs[2 * l + 1] | ((qh << (7 - 2 * l)) & 256)));
        for (int j = 0; j < 4; ++j) {
            y[j + 0] = db * grid1[j] * (signs[l] & kmask_iq2xs[j + 0] ? -1.f : 1.f);
            y[j + 4] = db * grid2[j] * (signs[l] & kmask_iq2xs[j + 4] ? -1.f : 1.f);
        }
        y += 8;
    }
}

void deq_iq3_s(const BlockIQ3S* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / QK_K; i++) {
        const float d = fp16_to_fp32(x[i].d);
        const uint8_t* qs = x[i].qs;
        const uint8_t* qh = x[i].qh;
        const uint8_t* signs = x[i].signs;
        for (int ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
            const float db1 = d * (1 + 2 * (x[i].scales[ib32 / 2] & 0xf));
            const float db2 = d * (1 + 2 * (x[i].scales[ib32 / 2] >> 4));
            for (int half = 0; half < 2; half++) {
                deq_iq3_s_half(half ? db2 : db1, qs, qh[half], signs, y);
                y += 32;
                qs += 8;
                signs += 4;
            }
            qh += 2;
        }
    }
}

void deq_iq4_nl(const BlockIQ4NL* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / 32; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        for (int j = 0; j < 16; ++j) {
            y[j + 0] = d * kvalues_iq4nl[qs[j] & 0xf];
            y[j + 16] = d * kvalues_iq4nl[qs[j] >> 4];
        }
        y += 32;
    }
}

void deq_iq4_xs(const BlockIQ4XS* x, float* y, int64_t k) {
    for (int64_t i = 0; i < k / QK_K; i++) {
        const uint8_t* qs = x[i].qs;
        const float d = fp16_to_fp32(x[i].d);
        for (int ib = 0; ib < QK_K / 32; ++ib) {
            const int ls = ((x[i].scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x[i].scales_h >> 2 * ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j + 0] = dl * kvalues_iq4nl[qs[j] & 0xf];
                y[j + 16] = dl * kvalues_iq4nl[qs[j] >> 4];
            }
            y += 32;
            qs += 16;
        }
    }
}

}  // namespace

float fp16_to_fp32(uint16_t h) {
    _Float16 f;
    std::memcpy(&f, &h, 2);
    return (float)f;
}

float bf16_to_fp32(uint16_t h) {
    uint32_t u = (uint32_t)h << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}

bool dequant_supported(uint32_t type) {
    static constexpr uint32_t kTypes[] = {0, 1, 8, 12, 13, 14, 16, 18, 20, 21, 22, 23, 30, 40, 42, 1018, 1019};
    return std::find(std::begin(kTypes), std::end(kTypes), type) != std::end(kTypes);
}

namespace {

void deq_f16(const uint16_t* s, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = fp16_to_fp32(s[i]);
}

void deq_bf16(const uint16_t* s, float* dst, int64_t n) {
    for (int64_t i = 0; i < n; i++) dst[i] = bf16_to_fp32(s[i]);
}

// The float types and the 32/64-weight blocks; false for every other type
bool deq_plain(uint32_t type, const void* src, float* dst, int64_t n) {
    switch (type) {
        case 0:
            std::memcpy(dst, src, n * 4);
            return true;  // F32
        case 1:
            return deq_f16(static_cast<const uint16_t*>(src), dst, n), true;
        case 30:
            return deq_bf16(static_cast<const uint16_t*>(src), dst, n), true;
        case 8:
            return deq_q8_0(static_cast<const BlockQ8_0*>(src), dst, n), true;
        case 40:
            return deq_nvfp4(static_cast<const BlockNVFP4*>(src), dst, n), true;
        case 42:
            return deq_q2_0(static_cast<const BlockQ2_0*>(src), dst, n), true;
        default:
            return false;
    }
}

}  // namespace

void dequantize_row(uint32_t type, const void* src, float* dst, int64_t n) {
    if (deq_plain(type, src, dst, n)) return;
    switch (type) {
        case 12:
            return deq_q4_k(static_cast<const BlockQ4K*>(src), dst, n);
        case 13:
            return deq_q5_k(static_cast<const BlockQ5K*>(src), dst, n);
        case 14:
            return deq_q6_k(static_cast<const BlockQ6K*>(src), dst, n);
        case 16:
            return deq_iq2_xxs(static_cast<const BlockIQ2XXS*>(src), dst, n);
        case 18:
            return deq_iq3_xxs(static_cast<const BlockIQ3XXS*>(src), dst, n);
        case 1018:
            return deq_iq3_xxs_r(static_cast<const uint8_t*>(src), dst, n);
        case 20:
            return deq_iq4_nl(static_cast<const BlockIQ4NL*>(src), dst, n);
        case 21:
            return deq_iq3_s(static_cast<const BlockIQ3S*>(src), dst, n);
        case 22:
            return deq_iq2_s(static_cast<const BlockIQ2S*>(src), dst, n);
        case 23:
            return deq_iq4_xs(static_cast<const BlockIQ4XS*>(src), dst, n);
        case 1019:
            throw std::runtime_error("dequantize_row: TONA_Q6_K_T rows are not contiguous; use dequantize_rows");
        default:
            break;
    }
    throw std::runtime_error(std::format("dequantize_row: ggml type {} is not supported", type));
}

}  // namespace tona
