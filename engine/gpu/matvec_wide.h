// Wide passes: a wave computes 16 rows at once with f16 WMMA 16x16x16. B = weights, unpacked straight into registers
// (decode_q8; 4 bytes at a time by v_perm into the mantissa of 1024, packed -1024 and fma with scale/offset in f16),
// A = x of the tokens as f16 (row = token l % 16, the last one repeated from nb on), D: lane l holds row l % 16 for
// the tokens 8 * (l / 16) + e. Lane group g unpacks sub-block sub + g, the missing halves come by
// permlanex16 (lane l ^ 16). Every token runs the same steps no matter how many tokens the pass has, so the result
// is bit-identical across widths. Measurement: research/measurements/2026-10-25-width-8.md (k_wide), at 8 tokens
// Q6_K reads 191 instead of 125 GB/s.
// Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_q8_core.h"
#include "gpu/matvec_q8_iq.h"
#include "gpu/matvec_q8_kquants.h"
#include "gpu/matvec_q8_nvfp4.h"
#include "gpu/matvec_q8_tile.h"
#include "gpu/matvec_task.h"

namespace tona::gpu {
namespace {

using wh8 = _Float16 __attribute__((ext_vector_type(8)));
using wh2 = _Float16 __attribute__((ext_vector_type(2)));
using wf8 = float __attribute__((ext_vector_type(8)));

template <bool US>
__device__ __forceinline__ void wide_cvt4(uint32_t wd, wh2 sc, wh2 bo, _Float16* o) {
    if (!US) wd ^= 0x80808080u;
    const uint32_t p0 = __builtin_amdgcn_perm(0x64646464u, wd, 0x04010400u);
    const uint32_t p1 = __builtin_amdgcn_perm(0x64646464u, wd, 0x04030402u);
    const wh2 off = {(_Float16)(US ? 1024.0f : 1152.0f), (_Float16)(US ? 1024.0f : 1152.0f)};
    wh2 h0 = __builtin_bit_cast(wh2, p0) - off, h1 = __builtin_bit_cast(wh2, p1) - off;
    h0 = h0 * sc + bo, h1 = h1 * sc + bo;
    o[0] = h0[0], o[1] = h0[1], o[2] = h1[0], o[3] = h1[1];
}

__device__ __forceinline__ wh8 wide_x8(const float* p) {
    const float4 a = reinterpret_cast<const float4*>(p)[0], b = reinterpret_cast<const float4*>(p)[1];
    return wh8{(_Float16)a.x, (_Float16)a.y, (_Float16)a.z, (_Float16)a.w,
               (_Float16)b.x, (_Float16)b.y, (_Float16)b.z, (_Float16)b.w};
}

__device__ __forceinline__ uint32_t pick(bool first, uint32_t a, uint32_t b) { return first ? a : b; }

// D of the WMMA: lane group g holds the tokens 8 * g + e of this lane's row
__device__ __forceinline__ void wide_store(const DevTask& t, const wf8& acc, int64_t row, int g, int nb) {
    if (row < t.rows)
#pragma unroll
        for (int e = 0; e < 8; e++)
            if (8 * g + e < nb) t.y[(8 * g + e) * t.yb + row] = epilogue(t, acc[e]);
}

template <uint32_t T>
__device__ __forceinline__ void wide_loop(const DevTask& t, int64_t u0, int64_t W, int lane, const Tabs& tb,
                                          const Batch& bt) {
    constexpr bool US = q8_unsigned<T>();
    const int r = lane & 15, g = lane >> 4, nb = bt.nb;
    const float* xr = t.x + (int64_t)min(r, nb - 1) * t.xb + 8 * g;
    const int subs = t.in / 32;
    for (; u0 < t.units; u0 += W) {
        const int64_t row = u0 * 16 + r;
        const uint8_t* wr = row_at<T>(t.w, min<int64_t>(row, t.rows - 1), t.row_bytes);
        wf8 acc = {};
#pragma unroll 2
        for (int sub = 0; sub < subs; sub += 2) {
            Dec o;
            decode_row<T>(wr, t.in, sub + g, tb, o);
            const uint32_t own[4] = {pick(g != 0, o.w[2], o.w[0]), pick(g != 0, o.w[3], o.w[1]),
                                     pick(g != 0, o.w[6], o.w[4]), pick(g != 0, o.w[7], o.w[5])};
            const uint32_t snd[4] = {pick(g != 0, o.w[0], o.w[2]), pick(g != 0, o.w[1], o.w[3]),
                                     pick(g != 0, o.w[4], o.w[6]), pick(g != 0, o.w[5], o.w[7])};
            uint32_t got[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                got[k] = (uint32_t)__builtin_amdgcn_permlanex16((int)snd[k], (int)snd[k], 0x76543210, 0xfedcba98, false,
                                                                false);
            const float sc4[4] = {o.s_lo, o.s_hi, o.b_lo, o.b_hi};
            float gs[4];
#pragma unroll
            for (int k = 0; k < 4; k++)
                gs[k] = __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, sc4[k]),
                                                                               __builtin_bit_cast(int, sc4[k]),
                                                                               0x76543210, 0xfedcba98, false, false));
#pragma unroll
            for (int q = 0; q < 2; q++) {  // sub-block sub (g = 0 own words, g = 1 received), then sub + 1
                const bool mine = (q == 0) == (g == 0);
                const uint32_t* wv = mine ? own : got;
                const float* sv = mine ? sc4 : gs;
                const wh2 sl = {(_Float16)sv[0], (_Float16)sv[0]}, bl = {(_Float16)sv[2], (_Float16)sv[2]};
                const wh2 sh = {(_Float16)sv[1], (_Float16)sv[1]}, bh = {(_Float16)sv[3], (_Float16)sv[3]};
                _Float16 v[16];
                wide_cvt4<US>(wv[0], sl, bl, v);
                wide_cvt4<US>(wv[1], sl, bl, v + 4);
                wide_cvt4<US>(wv[2], sh, bh, v + 8);
                wide_cvt4<US>(wv[3], sh, bh, v + 12);
                wh8 b0, b1;
#pragma unroll
                for (int j = 0; j < 8; j++) b0[j] = v[j], b1[j] = v[8 + j];
                const wh8 a0 = wide_x8(xr + (sub + q) * 32), a1 = wide_x8(xr + (sub + q) * 32 + 16);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, acc);
                acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, acc);
            }
        }
        wide_store(t, acc, row, g, nb);
    }
}

}  // namespace
}  // namespace tona::gpu
