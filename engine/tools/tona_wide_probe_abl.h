// Ablation kernels of tona-wide-probe --abl: the row loop of an expert matvec (L = 8 lanes per row, 4 rows per wave, 32
// blocks x 16 waves as k_mega) with parts left out, to see which part limits the bandwidth of a format:
//   mode 0  the loads of the decode only (same addresses and widths, XOR of the words)
//   mode 1  the real decode (tables in LDS, sign masks) into a Dec, words summed, no x and no dot
//   mode 2  decode and the dot products with x from registers (no LDS read of x)
//   mode 3  row_q8 as in the engine (x from LDS)
//   mode 4  as mode 1 for IQ2_S and IQ3_S, but the sign masks from the sign byte by arithmetic instead of the table in
//   LDS
//           (the words differ from the table's only in the unused lanes; output meaningless)
// Output is meaningless except for mode 3. Included once into tona_wide_probe.hip.
#pragma once

#include "gpu/matvec_dev.h"
#include "tona_wide_probe_kernels.h"

namespace {

// The weight loads of decode_q8<T>, in the same widths, with their words XORed
template <uint32_t T>
__device__ __forceinline__ int load_only(const uint8_t* row, int64_t sub) {
    using namespace tona::gpu;
    if constexpr (T == 22) {
        const uint8_t* b = row + (sub / 8) * 82;
        const int64_t ib = sub % 8;
        return ld4_a2(b + 2 + 4 * ib) ^ ld4_a2(b + 34 + 4 * ib) ^ b[66 + ib] ^ b[74 + ib] ^ u16(b);
    } else if constexpr (T == 18) {
        const uint8_t* b = row + (sub / 8) * 98;
        const int64_t ib = sub % 8;
        return ld4_a2(b + 2 + 8 * ib) ^ ld4_a2(b + 6 + 8 * ib) ^ ld4_a2(b + 66 + 4 * ib) ^ u16(b);
    } else if constexpr (T == 21) {
        const uint8_t* b = row + (sub / 8) * 110;
        const int64_t ib = sub % 8;
        return ld4_a2(b + 2 + 8 * ib) ^ ld4_a2(b + 6 + 8 * ib) ^ b[66 + ib] ^ ld4_a2(b + 74 + 4 * ib) ^
               b[106 + ib / 2] ^ u16(b);
    } else if constexpr (T == 23) {
        const uint8_t* b = row + (sub / 8) * 136;
        const int64_t ib = sub % 8;
        const uint8_t* qs = b + 8 + 16 * ib;
        return b[4 + ib / 2] ^ u16(b + 2) ^ ld4(qs) ^ ld4(qs + 4) ^ ld4(qs + 8) ^ ld4(qs + 12);
    } else if constexpr (T == 20) {
        const uint8_t* b = row + sub * 18;
        return u16(b) ^ ld4_a2(b + 2) ^ ld4_a2(b + 6) ^ ld4_a2(b + 10) ^ ld4_a2(b + 14);
    } else {  // 42
        const uint8_t* b = row + (sub / 2) * 18;
        const uint8_t* q = b + 2 + (sub & 1) * 8;
        return u16(b) ^ ld4_a2(q) ^ ld4_a2(q + 4);
    }
}

// Mask bytes (0xFF where the bit is set) of the four low bits of s, without a table
__device__ __forceinline__ uint32_t sign_mask4(uint32_t s) {
    const uint32_t spread = ((s & 0xFu) * 0x204081u) & 0x01010101u;
    return __builtin_amdgcn_perm(0u, 0u, spread | 0x0C0C0C0Cu);
}

// decode_q8<22> and <21> with sign_mask4 in place of the m8 table
template <uint32_t T>
__device__ __forceinline__ void decode_arith(const uint8_t* row, int64_t sub, const tona::gpu::Tabs& tb,
                                             tona::gpu::Dec& o) {
    using namespace tona::gpu;
    if constexpr (T == 22) {
        const uint8_t* b = row + (sub / 8) * 82;
        const int64_t ib = sub % 8;
        const int qs4 = ld4_a2(b + 2 + 4 * ib), sg4 = ld4_a2(b + 34 + 4 * ib);
        const int qh = b[66 + ib], sc = b[74 + ib];
#pragma unroll
        for (int64_t l = 0; l < 4; l++) {
            const int idx = ((qs4 >> 8 * l) & 0xff) | ((qh << (8 - 2 * l)) & 0x300);
            const uint64_t g = tb.iq2s[idx];
            const uint32_t sb = (sg4 >> 8 * l) & 0xff;
            o.w[2 * l] = apply_sign((uint32_t)g, sign_mask4(sb));
            o.w[2 * l + 1] = apply_sign((uint32_t)(g >> 32), sign_mask4(sb >> 4));
        }
        const float d = h2f(b) * 0.125f;
        o.s_lo = d * (float)(1 + 2 * (sc & 0xf)), o.s_hi = d * (float)(1 + 2 * (sc >> 4));
    } else {
        decode_q8<T>(row, sub, tb, o);
    }
}

constexpr int kAblL = 8, kAblG = 2;  // lanes per row, sub-blocks per round (ROWQ8_G)

// Decode of the sub-blocks of one round into o (modes 1, 2, 4)
template <uint32_t T, int MODE>
__device__ __forceinline__ void abl_decode(const uint8_t* row, int64_t sub0, int subs, const tona::gpu::Tabs& tabs,
                                           tona::gpu::Dec (&o)[kAblG]) {
    for (int64_t g = 0; g < kAblG; g++) {
        if (sub0 + g * kAblL >= subs) continue;
        if constexpr (MODE == 4)
            decode_arith<T>(row, sub0 + g * kAblL, tabs, o[g]);
        else
            tona::gpu::decode_q8<T>(row, sub0 + g * kAblL, tabs, o[g]);
    }
}

// What a round does with the decoded sub-block: modes 1 and 4 sum the words, mode 2 does the dot products with the
// x words in registers
template <uint32_t T, int MODE, int NB>
__device__ __forceinline__ int abl_use(const tona::gpu::Dec& o, const int* xr, float (&s)[NB]) {
    int acc = 0;
    if constexpr (MODE == 1 || MODE == 4) {
        for (int k = 0; k < 8; k++) acc += o.w[k];
        s[0] += o.s_lo + o.s_hi;
    } else {
        for (int b = 0; b < NB; b++) s[b] = tona::gpu::apply_q8<T>(o, xr, tona::gpu::Q8Meta{1.0f, 0, 0}, s[b]);
    }
    return acc;
}

// Modes 0, 1, 2 and 4 for one row: the sub-blocks j, j + 8, ... of the lane; returns the integer part of the result
template <uint32_t T, int MODE, int NB>
__device__ __forceinline__ int abl_row(const uint8_t* row, int in, int j, const tona::gpu::Tabs& tabs, const int* xr,
                                       float (&s)[NB]) {
    const int subs = in / 32;
    int acc = 0;
    for (int b = 0; b < NB; b++) s[b] = 0;
    for (int sub0 = j; sub0 < subs; sub0 += kAblG * kAblL) {
        if constexpr (MODE == 0) {
            for (int64_t g = 0; g < kAblG && sub0 + g * kAblL < subs; g++) acc ^= load_only<T>(row, sub0 + g * kAblL);
        } else {
            tona::gpu::Dec o[kAblG];
            abl_decode<T, MODE>(row, sub0, subs, tabs, o);
            for (int64_t g = 0; g < kAblG && sub0 + g * kAblL < subs; g++) acc += abl_use<T, MODE, NB>(o[g], xr, s);
        }
    }
    return acc;
}

struct AblArgs {
    const uint8_t* w;
    int64_t rows, rb;
    int in;
    const int8_t* xq;
    const tona::gpu::Q8Meta* xm;
    float* y;
};

template <uint32_t T, int MODE, int NB>
__global__ void __launch_bounds__(PT) k_abl(const AblArgs a) {
    using namespace tona::gpu;
    TABS_DECL;
    __shared__ __attribute__((aligned(16))) int8_t lx[8 * 2560 + 8 * 80 * sizeof(Q8Meta)];
    const int64_t xs = a.in;
    Q8Meta* lm = reinterpret_cast<Q8Meta*>(lx + NB * xs);
    for (int64_t i = threadIdx.x; i < NB * xs / 16; i += PT)
        reinterpret_cast<uint4*>(lx)[i] = reinterpret_cast<const uint4*>(a.xq)[i];
    for (int64_t i = threadIdx.x; i < NB * xs / 32; i += PT)
        reinterpret_cast<uint2*>(lm)[i] = reinterpret_cast<const uint2*>(a.xm)[i];
    __syncthreads();
    const int lane = threadIdx.x % WAVE, j = lane % kAblL;
    const int64_t gw = (int64_t)blockIdx.x * PW + threadIdx.x / WAVE, GW = (int64_t)gridDim.x * PW;
    int xr[8];  // mode 2: x of the first sub-block of this lane, kept in registers
    const int* xp = reinterpret_cast<const int*>(lx + (int64_t)j * 32);
    for (int k = 0; k < 8; k++) xr[k] = xp[k];
    for (int64_t u = gw; u * (WAVE / kAblL) < a.rows; u += GW) {
        const int64_t r = u * (WAVE / kAblL) + lane / kAblL;
        if (r >= a.rows) continue;
        const uint8_t* row = a.w + r * a.rb;
        float s[NB];
        int acc = 0;
        if constexpr (MODE == 3) {
            row_q8<T, NB>(Q8RowArgs{row, lx, lm, a.in, j, kAblL, NB, xs, 0xffu}, tabs, s);
        } else {
            acc = abl_row<T, MODE, NB>(row, a.in, j, tabs, xr, s);
        }
        float tot = s[0] + (float)acc;
        for (int b = 1; b < NB; b++) tot += s[b];
        for (int off = kAblL / 2; off > 0; off /= 2) tot += __shfl_xor(tot, off, WAVE);
        if (j == 0) a.y[r] = tot;
    }
}

}  // namespace
