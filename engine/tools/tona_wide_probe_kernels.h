// Kernels of tona-wide-probe: the megakernel's int8 matvec phase (k_probe) and the wide-pass candidate that unpacks
// into f16 and uses WMMA (k_wide). Included once into tona_wide_probe.hip.
#pragma once

#include "gpu/matvec_dev.h"

namespace {

constexpr int PW = 16, PT = PW * tona::gpu::WAVE;  // waves per block as in k_mega

using half8 = _Float16 __attribute__((ext_vector_type(8)));
using half2v = _Float16 __attribute__((ext_vector_type(2)));
using float8 = float __attribute__((ext_vector_type(8)));

// x of all nb tokens into LDS (as OP_MATVEC with i[0] > 0), then the units as in run_phase
template <int NB>
__global__ void __launch_bounds__(PT)
    k_probe(const tona::gpu::DevTask* t, const int8_t* xq, const tona::gpu::Q8Meta* xm, int nb, int64_t xs) {
    using namespace tona::gpu;
    TABS_DECL;
    __shared__ __attribute__((aligned(16))) int8_t lx[8 * 2560 + 8 * 80 * sizeof(Q8Meta)];
    Q8Meta* lm = reinterpret_cast<Q8Meta*>(lx + nb * xs);
    for (int64_t i = threadIdx.x; i < nb * xs / 16; i += PT)
        reinterpret_cast<uint4*>(lx)[i] = reinterpret_cast<const uint4*>(xq)[i];
    for (int64_t i = threadIdx.x; i < nb * xs / 32; i += PT)
        reinterpret_cast<uint2*>(lm)[i] = reinterpret_cast<const uint2*>(xm)[i];
    __syncthreads();
    Batch bt{nullptr, nullptr, 0, nb, -1};
    bt.lx = lx, bt.lm = lm;
    run_phase<NB>(PhaseRange{t, 0, 1},
                  WaveCtx{(int64_t)blockIdx.x * PW + threadIdx.x / WAVE, (int64_t)gridDim.x * PW,
                          (int)(threadIdx.x % WAVE), nullptr, tabs},
                  bt);
}

// 4 weight bytes -> 4 f16 (s * q + b): bytes by v_perm into the mantissa of 1024 (exact), then packed -1024
// (-1152 when signed: bytes ^ 0x80 beforehand) and fma with scale and offset
template <bool US>
__device__ __forceinline__ void cvt4(uint32_t wd, half2v sc, half2v bo, _Float16* o) {
    if (!US) wd ^= 0x80808080u;
    const uint32_t p0 = __builtin_amdgcn_perm(0x64646464u, wd, 0x04010400u);
    const uint32_t p1 = __builtin_amdgcn_perm(0x64646464u, wd, 0x04030402u);
    const half2v off = {(_Float16)(US ? 1024.0f : 1152.0f), (_Float16)(US ? 1024.0f : 1152.0f)};
    half2v h0 = __builtin_bit_cast(half2v, p0) - off, h1 = __builtin_bit_cast(half2v, p1) - off;
    h0 = h0 * sc + bo, h1 = h1 * sc + bo;
    o[0] = h0[0], o[1] = h0[1], o[2] = h1[0], o[3] = h1[1];
}

// Lane group g decodes sub-block sub + g. Of its 8 words it keeps 4 (own) and sends the 4 the other lane group needs
// for its own sub-block: g = 0 keeps words 0, 1, 4, 5 and sends 2, 3, 6, 7; g = 1 the other way round.
__device__ __forceinline__ void split_words(const tona::gpu::Dec& o, int g, uint32_t* own, uint32_t* snd) {
    const int keep = g ? 2 : 0, send = g ? 0 : 2;
    own[0] = (uint32_t)o.w[keep], own[1] = (uint32_t)o.w[keep + 1];
    own[2] = (uint32_t)o.w[keep + 4], own[3] = (uint32_t)o.w[keep + 5];
    snd[0] = (uint32_t)o.w[send], snd[1] = (uint32_t)o.w[send + 1];
    snd[2] = (uint32_t)o.w[send + 4], snd[3] = (uint32_t)o.w[send + 5];
}

// The words and scales of the other lane group arrive by permlanex16 (lane l reads lane l ^ 16)
__device__ __forceinline__ void exchange_lanes(const uint32_t* snd, const float* sc4, uint32_t* got, float* gs) {
#pragma unroll
    for (int k = 0; k < 4; k++)
        got[k] = (uint32_t)__builtin_amdgcn_permlanex16((int)snd[k], (int)snd[k], 0x76543210, 0xfedcba98, false, false);
#pragma unroll
    for (int k = 0; k < 4; k++)
        gs[k] = __builtin_bit_cast(
            float, __builtin_amdgcn_permlanex16(__builtin_bit_cast(int, sc4[k]), __builtin_bit_cast(int, sc4[k]),
                                                0x76543210, 0xfedcba98, false, false));
}

// One sub-block: unpack 16 weights to f16 and two WMMAs (the two 16-halves of the 32 values)
template <bool US>
__device__ __forceinline__ float8 mma_subblock(float8 acc, const uint32_t* wv, const float* sv, const _Float16* xs) {
    const half2v sl = {(_Float16)sv[0], (_Float16)sv[0]}, bl = {(_Float16)sv[2], (_Float16)sv[2]};
    const half2v sh = {(_Float16)sv[1], (_Float16)sv[1]}, bh = {(_Float16)sv[3], (_Float16)sv[3]};
    _Float16 t[16];
    cvt4<US>(wv[0], sl, bl, t);
    cvt4<US>(wv[1], sl, bl, t + 4);
    cvt4<US>(wv[2], sh, bh, t + 8);
    cvt4<US>(wv[3], sh, bh, t + 12);
    half8 b0, b1;
#pragma unroll
    for (int j = 0; j < 8; j++) b0[j] = t[j], b1[j] = t[8 + j];
    const half8 a0 = *reinterpret_cast<const half8*>(xs);
    const half8 a1 = *reinterpret_cast<const half8*>(xs + 16);
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a0, b0, acc);
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a1, b1, acc);
}

struct WideArgs {
    const uint8_t* w;
    int64_t row_bytes;
    const _Float16* x;
    int nb;
    int in;
    int64_t rows;
    float* y;
};

// Wave per 16 rows: B = weights (column = row l % 16, K-eighth l / 16), A = x of the tokens (row = token l % 16),
// D: lane l holds row l % 16 for the tokens 8 * (l / 16) ... +7. Two WMMAs per sub-block (16-halves).
template <uint32_t TY>
__global__ void __launch_bounds__(PT) k_wide(WideArgs a) {
    using namespace tona::gpu;
    TABS_DECL;
    constexpr bool US = q8_unsigned<TY>();
    const int lane = threadIdx.x % WAVE, r = lane & 15, g = lane >> 4;
    const int64_t W = (int64_t)gridDim.x * PW;
    const _Float16* xr = a.x + (int64_t)min(r, a.nb - 1) * a.in + 8 * (int64_t)g;
    for (int64_t grp = (int64_t)blockIdx.x * PW + threadIdx.x / WAVE; grp * 16 < a.rows; grp += W) {
        const uint8_t* wr = a.w + min<int64_t>(grp * 16 + r, a.rows - 1) * a.row_bytes;
        float8 acc = {};
        // two sub-blocks at a time: lane group g unpacks sub-block sub + g and receives the other group's halves
        for (int sub = 0; sub < a.in / 32; sub += 2) {
            Dec o;
            decode_q8<TY>(wr, sub + g, tabs, o);
            uint32_t own[4], snd[4], got[4];
            split_words(o, g, own, snd);
            const float sc4[4] = {o.s_lo, o.s_hi, o.b_lo, o.b_hi};
            float gs[4];
            exchange_lanes(snd, sc4, got, gs);
            // sub-block sub: g = 0 own words and scales, g = 1 received; sub + 1 the other way round
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const bool mine = (q == 0) == (g == 0);
                acc = mma_subblock<US>(acc, mine ? own : got, mine ? sc4 : gs, xr + (int64_t)(sub + q) * 32);
            }
        }
        const int64_t row = grp * 16 + r;
        if (row < a.rows)
#pragma unroll
            for (int e = 0; e < 8; e++)
                if (8 * g + e < a.nb) a.y[(int64_t)(8 * g + e) * a.rows + row] = acc[e];
    }
}

}  // namespace
