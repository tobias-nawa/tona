// The activations between the up and the down matvec of the CPU experts, quantized to int8 per 32 values the way
// wave_q8 does on the GPU (bit-identical): silu(gate) * up for the gated experts of the Coder and Ornith, relu(up)^2
// for the ungated experts of Bolt. Internal to cpu/.
#pragma once

#include <immintrin.h>

#include <cstdint>

#include "cpu/detmath.h"
#include "cpu/experts.h"

namespace tona::cpu::detail {

// det::silu for 16 values, the same steps as the scalar version (bit-identical)
[[gnu::always_inline]] inline __m512 silu16(__m512 x) {
    const __m512 sign = _mm512_set1_ps(-0.0f);
    const __m512 nx = _mm512_xor_ps(x, sign);  // e^(-x); clamped only so the exponent bits stay valid
    const __m512 c = _mm512_min_ps(_mm512_max_ps(nx, _mm512_set1_ps(-87.0f)), _mm512_set1_ps(88.0f));
    const __m512 k = _mm512_roundscale_ps(_mm512_mul_ps(c, _mm512_set1_ps(det::kL2e)), _MM_FROUND_TO_NEAREST_INT);
    const __m512 nk = _mm512_xor_ps(k, sign);
    __m512 r = _mm512_fmadd_ps(nk, _mm512_set1_ps(det::kLn2Hi), c);
    r = _mm512_fmadd_ps(nk, _mm512_set1_ps(det::kLn2Lo), r);
    __m512 p = _mm512_set1_ps(det::kP[0]);
    for (int i = 1; i < 6; i++) p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(det::kP[i]));
    const __m512 y = _mm512_add_ps(_mm512_fmadd_ps(p, _mm512_mul_ps(r, r), r), _mm512_set1_ps(1.0f));
    const __m512i bits = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(k), _mm512_set1_epi32(127)), 23);
    __m512 e = _mm512_mul_ps(y, _mm512_castsi512_ps(bits));
    // Edges as det::exp: >= 88 -> inf (NaN stays NaN), < -87 -> 0
    e = _mm512_mask_blend_ps(
        _mm512_cmp_ps_mask(nx, _mm512_set1_ps(88.0f), _CMP_NLT_UQ), e,
        _mm512_mask_blend_ps(_mm512_cmp_ps_mask(nx, nx, _CMP_UNORD_Q), _mm512_set1_ps(INFINITY), nx));
    e = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(nx, _mm512_set1_ps(-87.0f), _CMP_LT_OQ), e, _mm512_setzero_ps());
    return _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1.0f), e));
}

// Two vectors of 16 activations of one 32-value block to int8 with metadata, like quantize_q8 (wave_q8 on the GPU)
[[gnu::always_inline]] inline void quant_block(__m512 a0, __m512 a1, int8_t* q, Q8Meta* m) {
    const float amax = _mm512_reduce_max_ps(_mm512_max_ps(_mm512_abs_ps(a0), _mm512_abs_ps(a1)));
    const float d = amax / 127.0f;
    __m512i q0 = _mm512_setzero_si512(), q1 = q0;
    if (d > 0) {
        const __m512 dv = _mm512_set1_ps(d);
        q0 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_div_ps(a0, dv), _MM_FROUND_TO_NEAREST_INT));
        q1 = _mm512_cvtps_epi32(_mm512_roundscale_ps(_mm512_div_ps(a1, dv), _MM_FROUND_TO_NEAREST_INT));
    }
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q), _mm512_cvtsepi32_epi8(q0));
    _mm_storeu_si128(reinterpret_cast<__m128i*>(q + 16), _mm512_cvtsepi32_epi8(q1));
    *m = {d, (int16_t)_mm512_reduce_add_epi32(q0), (int16_t)_mm512_reduce_add_epi32(q1)};
}

// a = silu(g) * u (in g), then int8 per 32 like quantize_q8 (wave_q8 on the GPU)
inline void act_quant(float* g, const float* u, int64_t n, int8_t* q, Q8Meta* m) {
    for (int64_t b = 0; b < n / 32; b++) {
        float* gb = g + b * 32;
        const __m512 a0 = _mm512_mul_ps(silu16(_mm512_loadu_ps(gb)), _mm512_loadu_ps(u + b * 32));
        const __m512 a1 = _mm512_mul_ps(silu16(_mm512_loadu_ps(gb + 16)), _mm512_loadu_ps(u + b * 32 + 16));
        _mm512_storeu_ps(gb, a0), _mm512_storeu_ps(gb + 16, a1);
        quant_block(a0, a1, q + b * 32, m + b);
    }
}

// a = relu(u)^2 (the ungated experts of Bolt, op_act_relu2 on the GPU: v > 0 ? v * v : 0, NaN gives 0), then int8
inline void relu2_quant(const float* u, int64_t n, int8_t* q, Q8Meta* m) {
    const __m512 zero = _mm512_setzero_ps();
    for (int64_t b = 0; b < n / 32; b++) {
        __m512 a[2];
        for (int64_t h = 0; h < 2; h++) {
            const __m512 v = _mm512_loadu_ps(u + b * 32 + h * 16);
            a[h] = _mm512_maskz_mul_ps(_mm512_cmp_ps_mask(v, zero, _CMP_GT_OQ), v, v);
        }
        quant_block(a[0], a[1], q + b * 32, m + b);
    }
}

}  // namespace tona::cpu::detail
