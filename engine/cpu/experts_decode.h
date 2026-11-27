// Unpacking of the expert weight formats for the CPU kernels in experts.cpp: one decode<T> per format (signs and
// magnitudes of 64 values in a zmm) and scales8<T> (the scales of 8 sub-blocks); the K-quants Q4_K and Q6_K, which
// store unsigned values with an offset, take scales_bias8<T> (scales and offsets). Internal to cpu/.
#pragma once

#include <immintrin.h>

#include <cstdint>
#include <cstring>

#include "model/quant_tables.h"

namespace tona::cpu::detail {

using namespace tona::tables;

inline float fp16(const uint8_t* p) {
    _Float16 h;
    std::memcpy(&h, p, 2);
    return (float)h;
}
inline uint32_t ld32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, 4);
    return v;
}
inline uint64_t ld64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    return v;
}

// Two sub-blocks sub, sub + 1 (64 values) unpacked into one zmm: magnitudes u (u8), signs neg (bit j: value j
// negative), scale per 16 values (a quarter of the register); Q2_0 stores q = value + 1, so the result gets -d *
// sum(x). 512 bits instead of two 256-bit halves and signs from bits instead of tables: IQ3_XXS 1 thread 122 -> 76 us
// (1 token), 214 -> 105 us (4 tokens). Gather is slower than single loads on Zen 5 (105 us).
// The scales come as a vector per 8 sub-blocks from scales8 (like decode_q8 on the GPU)
struct Dec {
    __m512i u;
    __mmask64 neg;
};

// IQ3_XXS: 4 x 7 sign bits, the 8th of each group is the parity (an even number of negative values)
[[gnu::always_inline]] inline uint32_t even_signs(uint32_t aux) {
    const uint32_t x = _pdep_u32(aux, 0x7f7f7f7f);
    uint32_t p = x ^ (x >> 4);
    p ^= p >> 2;
    p ^= p >> 1;
    return x | ((p & 0x01010101u) << 7);
}

template <uint32_t T>
[[gnu::always_inline]] inline void decode(const uint8_t* row, int64_t sub, Dec& o);

template <>
[[gnu::always_inline]] inline void decode<22>(const uint8_t* row, int64_t sub, Dec& o) {  // IQ2_S, 82 bytes per 256
    const uint8_t* b = row + (sub / 8) * 82;
    const int64_t ib = sub % 8;
    // Index l of the two sub-blocks: byte l of qs, bits 2l, 2l+1 of qh as bits 8, 9, built as a vector and loaded one
    // by one through the stack: 84 us per matrix, gather 91, scalar index per value 195
    const uint16_t qh2 = (uint16_t)(b[66 + ib] | b[67 + ib] << 8);
    const __m256i hi = _mm256_slli_epi32(
        _mm256_and_si256(_mm256_srlv_epi32(_mm256_set1_epi32(qh2), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
                         _mm256_set1_epi32(3)),
        8);
    const __m256i idx =
        _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 2 + 4 * ib))), hi);
    alignas(32) uint32_t id[8];
    _mm256_store_si256(reinterpret_cast<__m256i*>(id), idx);
    o.u = _mm512_set_epi64((long long)iq2s_grid[id[7]], (long long)iq2s_grid[id[6]], (long long)iq2s_grid[id[5]],
                           (long long)iq2s_grid[id[4]], (long long)iq2s_grid[id[3]], (long long)iq2s_grid[id[2]],
                           (long long)iq2s_grid[id[1]], (long long)iq2s_grid[id[0]]);
    o.neg = ld64(b + 34 + 4 * ib);
}

// IQ2_XXS, 66 bytes per 256: sub-block ib is 8 bytes at 2 + 8 ib (four grid indices, then four 7-bit sign indices and
// the scale in bits 28..31). The pair (sub, sub + 1) is 16 contiguous bytes; sub is even.
template <>
[[gnu::always_inline]] inline void decode<16>(const uint8_t* row, int64_t sub, Dec& o) {
    const uint8_t* q = row + (sub / 8) * 66 + 2 + 8 * (sub % 8);
    const uint8_t* r = q + 8;
    o.u = _mm512_set_epi64((long long)iq2xxs_grid[r[3]], (long long)iq2xxs_grid[r[2]], (long long)iq2xxs_grid[r[1]],
                           (long long)iq2xxs_grid[r[0]], (long long)iq2xxs_grid[q[3]], (long long)iq2xxs_grid[q[2]],
                           (long long)iq2xxs_grid[q[1]], (long long)iq2xxs_grid[q[0]]);
    o.neg = (uint64_t)even_signs(ld32(q + 4)) | (uint64_t)even_signs(ld32(r + 4)) << 32;
}

template <>
[[gnu::always_inline]] inline void decode<18>(const uint8_t* row, int64_t sub, Dec& o) {  // IQ3_XXS, 98 bytes per 256
    const uint8_t* b = row + (sub / 8) * 98;
    const int64_t ib = sub % 8;
    const uint8_t* q = b + 2 + 8 * ib;
    const uint32_t a0 = ld32(b + 66 + 4 * ib), a1 = ld32(b + 70 + 4 * ib);
    const __m256i lo = _mm256_set_epi32((int)iq3xxs_grid[q[7]], (int)iq3xxs_grid[q[6]], (int)iq3xxs_grid[q[5]],
                                        (int)iq3xxs_grid[q[4]], (int)iq3xxs_grid[q[3]], (int)iq3xxs_grid[q[2]],
                                        (int)iq3xxs_grid[q[1]], (int)iq3xxs_grid[q[0]]);
    const __m256i hi = _mm256_set_epi32((int)iq3xxs_grid[q[15]], (int)iq3xxs_grid[q[14]], (int)iq3xxs_grid[q[13]],
                                        (int)iq3xxs_grid[q[12]], (int)iq3xxs_grid[q[11]], (int)iq3xxs_grid[q[10]],
                                        (int)iq3xxs_grid[q[9]], (int)iq3xxs_grid[q[8]]);
    o.u = _mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1);
    o.neg = (uint64_t)even_signs(a0) | (uint64_t)even_signs(a1) << 32;
}

template <>
[[gnu::always_inline]] inline void decode<21>(const uint8_t* row, int64_t sub, Dec& o) {  // IQ3_S, 110 bytes per 256
    const uint8_t* b = row + (sub / 8) * 110;
    const int64_t ib = sub % 8;
    const uint8_t* q = b + 2 + 8 * ib;
    // Index k: byte k of q, bit k of qh as bit 8; single loads 115 us, gather 122, indices through the stack 126
    auto g = [&](int k) { return (int)iq3s_grid[q[k] | (((b[66 + ib + k / 8] >> (k % 8)) & 1) << 8)]; };
    const __m256i lo = _mm256_set_epi32(g(7), g(6), g(5), g(4), g(3), g(2), g(1), g(0));
    const __m256i hi = _mm256_set_epi32(g(15), g(14), g(13), g(12), g(11), g(10), g(9), g(8));
    o.u = _mm512_inserti32x8(_mm512_castsi256_si512(lo), hi, 1);
    o.neg = ld64(b + 74 + 4 * ib);
}

// 16 bytes with two 4-bit indices each (low = value k, high = value k + 16), looked up in kvalues_iq4nl
[[gnu::always_inline]] inline __m256i iq4_values(const uint8_t* qs) {
    const __m128i q = _mm_loadu_si128(reinterpret_cast<const __m128i*>(qs));
    const __m128i m = _mm_set1_epi8(0x0f);
    const __m256i idx = _mm256_set_m128i(_mm_and_si128(_mm_srli_epi16(q, 4), m), _mm_and_si128(q, m));
    const __m256i tab = _mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i*>(kvalues_iq4nl)));
    return _mm256_shuffle_epi8(tab, idx);
}

[[gnu::always_inline]] inline void iq4_pair(const uint8_t* q0, const uint8_t* q1, Dec& o) {
    const __m512i w = _mm512_inserti32x8(_mm512_castsi256_si512(iq4_values(q0)), iq4_values(q1), 1);
    o.u = _mm512_abs_epi8(w);
    o.neg = _mm512_movepi8_mask(w);
}

template <>
[[gnu::always_inline]] inline void decode<20>(const uint8_t* row, int64_t sub, Dec& o) {  // IQ4_NL, 18 bytes per 32
    const uint8_t* b = row + sub * 18;
    iq4_pair(b + 2, b + 20, o);
}

template <>
[[gnu::always_inline]] inline void decode<23>(const uint8_t* row, int64_t sub, Dec& o) {  // IQ4_XS, 136 bytes per 256
    const uint8_t* b = row + (sub / 8) * 136;
    const int64_t ib = sub % 8;
    iq4_pair(b + 8 + 16 * ib, b + 24 + 16 * ib, o);
}

template <>
[[gnu::always_inline]] inline void decode<42>(const uint8_t* row, int64_t sub,
                                              Dec& o) {  // Q2_0, 18 bytes per 64, value = q - 1
    const uint8_t* b = row + (sub / 2) * 18;
    // Byte j of 64-bit lane i takes the bits from 16 (i mod 4) + 2 j: value 8 i + j; lanes 0-3 from the first, 4-7
    // from the second 64-bit word
    const __m512i ctl =
        _mm512_set_epi64(0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL,
                         0x3e3c3a3836343230LL, 0x2e2c2a2826242220LL, 0x1e1c1a1816141210LL, 0x0e0c0a0806040200LL);
    const long long q0 = (long long)ld64(b + 2), q1 = (long long)ld64(b + 10);
    const __m512i src = _mm512_set_epi64(q1, q1, q1, q1, q0, q0, q0, q0);
    o.u = _mm512_and_si512(_mm512_multishift_epi64_epi8(ctl, src), _mm512_set1_epi8(3));
    o.neg = 0;
}

// Q4_K, 144 bytes per 256: 4 groups of 64 values, each 32 bytes of nibbles (the first 32 values are the low nibbles,
// the next 32 the high ones), so the two sub-blocks of a pair are the two nibble planes of the same 32 bytes. Unsigned
// values 0..15; the offset is in the scales (scales_bias8)
template <>
[[gnu::always_inline]] inline void decode<12>(const uint8_t* row, int64_t sub, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 144;
    const __m256i q = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 16 + 32 * ((sub % 8) / 2)));
    const __m256i m = _mm256_set1_epi8(0x0f);
    o.u = _mm512_inserti64x4(_mm512_castsi256_si512(_mm256_and_si256(q, m)),
                             _mm256_and_si256(_mm256_srli_epi16(q, 4), m), 1);
    o.neg = 0;
}

// Q6_K, 210 bytes per 256: two halves of 128 values. In a half, 64 bytes of low nibbles (values 0..31 and 32..63 in the
// low nibbles of bytes 0..31 and 32..63, values 64..95 and 96..127 in the high nibbles) and 32 bytes of 2-bit high
// parts (value k of the sub-block part p in bits 2p, 2p + 1 of byte k). Unsigned values 0..63, offset -32 in the
// scales. The sub-block pair (p, p + 1) with p even shares the 64 bytes of low parts and the 32 bytes of high parts.
template <>
[[gnu::always_inline]] inline void decode<14>(const uint8_t* row, int64_t sub, Dec& o) {
    const uint8_t* b = row + (sub / 8) * 210;
    const int64_t ib = sub % 8;
    const int p = (int)(ib % 4);  // even: the pair is (p, p + 1)
    const __m512i ql = _mm512_loadu_si512(b + 64 * (ib / 4));
    const __m512i lo =
        _mm512_and_si512(_mm512_srl_epi16(ql, _mm_cvtsi32_si128(p >= 2 ? 4 : 0)), _mm512_set1_epi8(0x0f));
    const __m256i qh = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 128 + 32 * (ib / 4)));
    const __m512i sh = _mm512_inserti64x4(_mm512_set1_epi16((short)(2 * p)),
                                          _mm512_castsi512_si256(_mm512_set1_epi16((short)(2 * p + 2))), 1);
    const __m512i hi = _mm512_srlv_epi16(_mm512_inserti64x4(_mm512_castsi256_si512(qh), qh, 1), sh);
    o.u = _mm512_or_si512(lo, _mm512_slli_epi16(_mm512_and_si512(hi, _mm512_set1_epi8(0x03)), 4));
    o.neg = 0;
}

// Scales of the sub-blocks s0 ... s0 + 7 (nsub of them valid, the rest 0) per 16-value half, the same steps as
// decode_q8 on the GPU (d * (float)k with integer k)
template <uint32_t T>
[[gnu::always_inline]] inline void scales8(const uint8_t* row, int64_t s0, int nsub, __m256& SL, __m256& SH);

[[gnu::always_inline]] inline __m256i nibbles8(const uint8_t* p) {  // nibbles of p[0..3], low one first
    const __m256i v = _mm256_cvtepu8_epi32(_mm_shuffle_epi8(
        _mm_cvtsi32_si128((int)ld32(p)), _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0)));
    return _mm256_and_si256(_mm256_srlv_epi32(v, _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4)), _mm256_set1_epi32(0xf));
}
[[gnu::always_inline]] inline __m256 odd_scale(__m256i k, float d) {  // d · (float)(1 + 2k)
    return _mm256_mul_ps(_mm256_set1_ps(d),
                         _mm256_cvtepi32_ps(_mm256_add_epi32(_mm256_slli_epi32(k, 1), _mm256_set1_epi32(1))));
}

template <>
[[gnu::always_inline]] inline void scales8<22>(const uint8_t* row, int64_t s0, int, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 8) * 82;
    const __m256i v = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(b + 74)));
    const float d = fp16(b) * 0.125f;
    SL = odd_scale(_mm256_and_si256(v, _mm256_set1_epi32(0xf)), d);
    SH = odd_scale(_mm256_srli_epi32(v, 4), d);
}
template <>
[[gnu::always_inline]] inline void scales8<16>(const uint8_t* row, int64_t s0, int, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 8) * 66;
    // The second dword of each 8-byte group holds the scale; the 64 bytes are the block's own qs
    const __m512i v = _mm512_srli_epi64(_mm512_loadu_si512(b + 2), 32);
    SL = SH = odd_scale(_mm256_srli_epi32(_mm512_cvtepi64_epi32(v), 28), fp16(b) * 0.125f);
}
template <>
[[gnu::always_inline]] inline void scales8<18>(const uint8_t* row, int64_t s0, int, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 8) * 98;
    const __m256i a = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(b + 66));
    SL = SH = odd_scale(_mm256_srli_epi32(a, 28), fp16(b) * 0.25f);
}
template <>
[[gnu::always_inline]] inline void scales8<21>(const uint8_t* row, int64_t s0, int, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 8) * 110;
    SL = SH = odd_scale(nibbles8(b + 106), fp16(b));
}
template <>
[[gnu::always_inline]] inline void scales8<23>(const uint8_t* row, int64_t s0, int, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 8) * 136;
    const __m256i hi = _mm256_and_si256(
        _mm256_srlv_epi32(_mm256_set1_epi32(b[2] | (b[3] << 8)), _mm256_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14)),
        _mm256_set1_epi32(3));
    const __m256i ls = _mm256_or_si256(nibbles8(b + 4), _mm256_slli_epi32(hi, 4));
    SL = SH = _mm256_mul_ps(_mm256_set1_ps(fp16(b)), _mm256_cvtepi32_ps(_mm256_sub_epi32(ls, _mm256_set1_epi32(32))));
}
template <>
[[gnu::always_inline]] inline void scales8<20>(const uint8_t* row, int64_t s0, int nsub, __m256& SL, __m256& SH) {
    __m128i h = _mm_setzero_si128();
    const uint8_t* b = row + s0 * 18;
    auto u16 = [&](int64_t k) { return k < nsub ? (b[18 * k] | (b[18 * k + 1] << 8)) : 0; };
    h = _mm_insert_epi16(h, u16(0), 0), h = _mm_insert_epi16(h, u16(1), 1), h = _mm_insert_epi16(h, u16(2), 2);
    h = _mm_insert_epi16(h, u16(3), 3), h = _mm_insert_epi16(h, u16(4), 4), h = _mm_insert_epi16(h, u16(5), 5);
    h = _mm_insert_epi16(h, u16(6), 6), h = _mm_insert_epi16(h, u16(7), 7);
    SL = SH = _mm256_cvtph_ps(h);
}
template <>
[[gnu::always_inline]] inline void scales8<42>(const uint8_t* row, int64_t s0, int nsub, __m256& SL, __m256& SH) {
    const uint8_t* b = row + (s0 / 2) * 18;
    auto u16 = [&](int64_t k) { return 2 * k < nsub ? (b[18 * k] | (b[18 * k + 1] << 8)) : 0; };
    __m128i h = _mm_cvtsi32_si128(u16(0) | u16(1) << 16);
    h = _mm_insert_epi32(h, u16(2) | u16(3) << 16, 1);
    h = _mm_unpacklo_epi16(h, h);  // f0 f0 f1 f1 f2 f2 f3 f3
    SL = SH = _mm256_cvtph_ps(h);
}

// Formats with an offset per 16-value half: sum w*x = d_x * (s * dot(w, x) + b * sum(x)) with the sum of the x values
// of the half (Q8Meta::s0, s1), as apply_q8 on the GPU. Scales and offsets of the sub-blocks s0 ... s0 + 7 (s0 a
// multiple of 8, the block of 256 values), the same float steps as decode_q8 on the GPU.
template <uint32_t T>
[[gnu::always_inline]] inline void scales_bias8(const uint8_t* row, int64_t s0, __m256& SL, __m256& SH, __m256& BL,
                                                __m256& BH);

// Q4_K: s = d * sc, b = -dmin * mn per sub-block; sc and mn are 6-bit values packed in 12 bytes (get_scale_min_k4),
// unpacked as in ggml's AVX2 kernel: bytes 0..7 of u are sc, bytes 8..15 are mn
template <>
[[gnu::always_inline]] inline void scales_bias8<12>(const uint8_t* row, int64_t s0, __m256& SL, __m256& SH, __m256& BL,
                                                    __m256& BH) {
    const uint8_t* b = row + (s0 / 8) * 144;
    uint32_t u[4] = {};
    std::memcpy(u, b + 4, 12);
    constexpr uint32_t k1 = 0x3f3f3f3f, k2 = 0x0f0f0f0f, k3 = 0x03030303;
    u[3] = ((u[2] >> 4) & k2) | (((u[1] >> 6) & k3) << 4);
    const uint32_t aux = u[1] & k1;
    u[1] = (u[2] & k2) | (((u[0] >> 6) & k3) << 4);
    u[2] = aux;
    u[0] &= k1;
    const __m256i sc = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(u)));
    const __m256i mn = _mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(u + 2)));
    SL = SH = _mm256_mul_ps(_mm256_set1_ps(fp16(b)), _mm256_cvtepi32_ps(sc));
    BL = BH = _mm256_mul_ps(_mm256_set1_ps(-fp16(b + 2)), _mm256_cvtepi32_ps(mn));
}

// Q6_K: s = d * sc (int8 per 16 values: sub-block k has sc[2k] and sc[2k + 1]), b = -32 * s
template <>
[[gnu::always_inline]] inline void scales_bias8<14>(const uint8_t* row, int64_t s0, __m256& SL, __m256& SH, __m256& BL,
                                                    __m256& BH) {
    const uint8_t* b = row + (s0 / 8) * 210;
    const __m128i sc = _mm_shuffle_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i*>(b + 192)),
                                        _mm_setr_epi8(0, 2, 4, 6, 8, 10, 12, 14, 1, 3, 5, 7, 9, 11, 13, 15));
    const __m256 d = _mm256_set1_ps(fp16(b + 208));
    SL = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(sc)));
    SH = _mm256_mul_ps(d, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(sc, 8))));
    BL = _mm256_mul_ps(_mm256_set1_ps(-32.0f), SL);
    BH = _mm256_mul_ps(_mm256_set1_ps(-32.0f), SH);
}

// Formats that go through scales_bias8 (the others use scales8)
template <uint32_t T>
inline constexpr bool kOffsetFormat = T == 12 || T == 14;

}  // namespace tona::cpu::detail
