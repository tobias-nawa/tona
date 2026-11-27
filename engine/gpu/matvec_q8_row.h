// int8 path: row function. L lanes of a wave share one row, 32/L rows per wave. Part of matvec_dev.h; do not
// include on its own.
#pragma once

#include <algorithm>

#include "gpu/matvec_q8_core.h"
#include "gpu/matvec_q8_iq.h"
#include "gpu/matvec_q8_kquants.h"
#include "gpu/matvec_q8_nvfp4.h"
#include "gpu/matvec_q8_tile.h"

namespace tona::gpu {
namespace {

// One row for row_q8: NB vectors x (stride xs values, metadata xs/32), of which nb < NB are computed, and only those
// with bit b set in mask. j is the lane within the row's L lanes.
struct Q8RowArgs {
    const uint8_t* row;
    const int8_t* xq;
    const Q8Meta* meta;
    int in, j, L;
    int nb;
    int64_t xs;
    uint32_t mask;
};

// Unpack the G sub-blocks a lane handles in one round of row_q8
template <uint32_t T, int G>
__device__ __forceinline__ void decode_group(const Q8RowArgs a, int sub0, const Tabs& tb, Dec (&o)[G]) {
    const int subs = a.in / 32;
#pragma unroll
    for (int g = 0; g < G; g++)
        if (sub0 + g * a.L < subs) decode_row<T>(a.row, a.in, sub0 + g * a.L, tb, o[g]);
}

// Sum over the L lanes of a row
template <int NB>
__device__ __forceinline__ void reduce_lanes(int L, float (&s)[NB]) {
#pragma unroll
    for (int b = 0; b < NB; b++)
        for (int off = L / 2; off > 0; off /= 2) s[b] += __shfl_xor(s[b], off, WAVE);
}

// Dot products of one row (a sum over its L lanes) with up to NB vectors.
// Load G = 2 sub-blocks per lane at once, then compute in the same order (bit-identical to one by one): the load
// counters count in order, while loading one by one makes each lane wait a full latency per sub-block. Pass -2.8 %
// (1 token), -1.0 % (4), -2.2 % (8); G = 3 and 4 are slower (research/measurements/2026-10-25-hc-matvec.md).
#ifndef ROWQ8_G
#define ROWQ8_G 2
#endif
// Sub-blocks per round of one format: IQ2_XXS sub-blocks are 8 bytes, so a lane has the fewest bytes in flight of all
// formats and a deeper round pays (research/measurements/2026-11-19-bolt-opt1.md); the others keep ROWQ8_G
#ifndef ROWQ8_G16
#define ROWQ8_G16 ROWQ8_G
#endif
template <uint32_t T>
constexpr int rowq8_g() {
    if constexpr (T == 16) return ROWQ8_G16;
    return ROWQ8_G;
}
// The vectors B0..B1-1 of one round: the unpacked sub-blocks o times x of each vector, added to s[b]
template <uint32_t T, int NB, int G, int B0, int B1>
__device__ __forceinline__ void row_q8_vectors(const Q8RowArgs a, int sub0, const Dec (&o)[G], float (&s)[NB]) {
    const int subs = a.in / 32;
#pragma unroll
    for (int b = B0; b < B1; b++) {
        if (b >= a.nb || !(a.mask >> b & 1)) continue;
        uint4 xa[G], xc[G];
        Q8Meta mm[G];
#pragma unroll
        for (int g = 0; g < G; g++)
            if (sub0 + g * a.L < subs) {
                const uint4* xp = reinterpret_cast<const uint4*>(a.xq + b * a.xs + (sub0 + g * a.L) * 32);
                xa[g] = xp[0], xc[g] = xp[1], mm[g] = a.meta[b * a.xs / 32 + sub0 + g * a.L];
            }
#pragma unroll
        for (int g = 0; g < G; g++)
            if (sub0 + g * a.L < subs) {
                const int xi[8] = {(int)xa[g].x, (int)xa[g].y, (int)xa[g].z, (int)xa[g].w,
                                   (int)xc[g].x, (int)xc[g].y, (int)xc[g].z, (int)xc[g].w};
                s[b] = apply_q8<T>(o[g], xi, mm[g], s[b]);
            }
    }
}

// All vectors of the pass for one round. More than 4 vectors go in two groups behind a scheduling barrier: with the
// loads of 8 vectors in flight at once the kernel for 8 vectors spilled VGPRs here and ran about 4 ms per pass slower
// than the one for 4 (research/measurements/2026-11-12-verify-pass.md). The weights are unpacked once.
template <uint32_t T, int NB, int G>
__device__ __forceinline__ void row_q8_round(const Q8RowArgs a, int sub0, const Dec (&o)[G], float (&s)[NB]) {
    constexpr int H = NB > 4 ? 4 : NB;
    row_q8_vectors<T, NB, G, 0, H>(a, sub0, o, s);
    if constexpr (NB > H) {
        if (a.nb > H) {
            __builtin_amdgcn_sched_barrier(0);
            row_q8_vectors<T, NB, G, H, NB>(a, sub0, o, s);
        }
    }
}

// Pipelined variant (Raw<T>::kOn): the raw bytes of the next round are requested before this round is computed
template <uint32_t T, int NB, int G>
__device__ __forceinline__ void row_q8_pipe(const Q8RowArgs a, const Tabs& tb, float (&s)[NB]) {
    const int subs = a.in / 32;
    Raw<T> cur[G], nxt[G];
#pragma unroll
    for (int g = 0; g < G; g++)
        if (a.j + g * a.L < subs) load_raw(a.row, a.j + g * a.L, cur[g]);
    for (int sub0 = a.j; sub0 < subs; sub0 += G * a.L) {
        const int nx = sub0 + G * a.L;
#pragma unroll
        for (int g = 0; g < G; g++)
            if (nx + g * a.L < subs) load_raw(a.row, nx + g * a.L, nxt[g]);
        Dec o[G];
#pragma unroll
        for (int g = 0; g < G; g++)
            if (sub0 + g * a.L < subs) decode_raw(cur[g], tb, o[g]);
        row_q8_round<T, NB, G>(a, sub0, o, s);
#pragma unroll
        for (int g = 0; g < G; g++) cur[g] = nxt[g];
    }
}

template <uint32_t T, int NB>
__device__ __forceinline__ void row_q8(const Q8RowArgs a, const Tabs& tb, float (&s)[NB]) {
    constexpr int G = rowq8_g<T>();
#pragma unroll
    for (int b = 0; b < NB; b++) s[b] = 0;
    // up to 4 vectors only: at 8 the pipeline's second set of loads costs registers and was 0.08 ms per pass slower
    if constexpr (Raw<T>::kOn && NB <= 4) {
        row_q8_pipe<T, NB, G>(a, tb, s);
        reduce_lanes<NB>(a.L, s);
        return;
    }
    const int subs = a.in / 32;
    for (int sub0 = a.j; sub0 < subs; sub0 += G * a.L) {
        Dec o[G];
        decode_group<T, G>(a, sub0, tb, o);
        row_q8_round<T, NB, G>(a, sub0, o, s);
    }
    reduce_lanes<NB>(a.L, s);
}

// Largest power of two <= 32 that divides the number of sub-blocks
int lanes_per_row(int64_t in) {
    const int64_t subs = in / 32;
    return (int)std::min<int64_t>(WAVE, subs & -subs);
}

}  // namespace
}  // namespace tona::gpu
