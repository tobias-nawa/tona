// int8 path: TONA_Q6_K_T (type 1019, docs/use/tona-types.md), Q6_K matrices stored as tiles of 8 rows. Part of
// matvec_dev.h; do not include on its own.
//
// A "row pointer" of this type is the tile base plus the row within the tile (0..7) in the low bits: tiles start at a
// multiple of 16 bytes (8 rows of k / 256 * 210 bytes, GGUF data alignment 32), so the pointer carries the row and the
// loops of the other formats stay as they are. The row length k is not in the pointer: decode_row passes it.
// One decode reads 2 x 8 + 2 x 4 + 2 + 2 bytes instead of 32 x 2-byte pieces of a Q6_K block (research/measurements/
// 2026-11-13-pc-q6k-tile.md).
#pragma once

#include "gpu/matvec_q8_core.h"
#include "gpu/matvec_q8_kquants.h"

namespace tona::gpu {
namespace {

constexpr uint32_t kQ6KTile = 1019;

// Row r of a matrix of row_bytes per row
template <uint32_t T>
__device__ __forceinline__ const uint8_t* row_at(const uint8_t* w, int64_t r, int64_t row_bytes) {
    if constexpr (T == kQ6KTile)
        return w + (r >> 3) * 8 * row_bytes + (r & 7);
    else
        return w + r * row_bytes;
}

// The 2-bit fields of a 16-bit tile word at bit 0, 4, 8, 12 to the bits 4-5 of the four bytes
__device__ __forceinline__ uint32_t tile_spread_hi(uint32_t x) {
    uint32_t z = (x & 0x0033u) | ((x & 0x3300u) << 8);
    z = (z & 0x00030003u) | ((z & 0x00300030u) << 4);
    return z << 4;
}

// Sub-block (chunk of 32 weights) sub of the row in a tile of 8 rows with k weights each. Lane of the 8x8 fragment of
// row r8, group q: ((r8 & 4) << 2) | ((r8 & 3) << 1) | ((q & 2) << 2) | (q & 1) (engine/model/quant_tile.cpp). The
// groups q = 0, 1 are adjacent entries, as are q = 2, 3 eight entries on. A tile word holds 8 weights as nibbles in
// order; the even and odd ones are separated by masks and one v_perm interleaves them back, so x needs no permutation.
__device__ __forceinline__ void decode_tile(const uint8_t* row, int k, int sub, Dec& o) {
    const uintptr_t p = reinterpret_cast<uintptr_t>(row);
    const int r8 = (int)(p & 15);
    const uint8_t* tile = reinterpret_cast<const uint8_t*>(p - r8);
    const int e0 = 32 * sub + (((r8 & 4) << 2) | ((r8 & 3) << 1));
    const uint2 a = *reinterpret_cast<const uint2*>(tile + 4 * e0);
    const uint2 b = *reinterpret_cast<const uint2*>(tile + 4 * (e0 + 8));
    const uint32_t h0 = *reinterpret_cast<const uint32_t*>(tile + 4 * k + 2 * e0);
    const uint32_t h1 = *reinterpret_cast<const uint32_t*>(tile + 4 * k + 2 * (e0 + 8));
    const int8_t* sc = reinterpret_cast<const int8_t*>(tile + 6 * k) + 16 * sub + 8 * (r8 >> 2) + (r8 & 3);
    const uint32_t ql[4] = {a.x, a.y, b.x, b.y};
    const uint32_t qh[4] = {h0 & 0xFFFFu, h0 >> 16, h1 & 0xFFFFu, h1 >> 16};
#pragma unroll
    for (int q = 0; q < 4; q++) {
        const uint32_t ev = (ql[q] & 0x0F0F0F0Fu) | tile_spread_hi(qh[q]);
        const uint32_t od = ((ql[q] >> 4) & 0x0F0F0F0Fu) | tile_spread_hi(qh[q] >> 2);
        o.w[2 * q] = (int)__builtin_amdgcn_perm(od, ev, 0x05010400u);
        o.w[2 * q + 1] = (int)__builtin_amdgcn_perm(od, ev, 0x07030602u);
    }
    const float d = h2f(tile + 6 * k + k / 2 + 2 * (8 * (sub / 8) + r8));
    o.s_lo = d * sc[0], o.s_hi = d * sc[4];
    o.b_lo = -32 * o.s_lo, o.b_hi = -32 * o.s_hi;
}

// Sub-block sub of a row of any int8-path format; in = row length (only the tile type needs it)
template <uint32_t T>
__device__ __forceinline__ void decode_row(const uint8_t* row, int in, int sub, const Tabs& tb, Dec& o) {
    if constexpr (T == kQ6KTile)
        decode_tile(row, in, sub, o);
    else
        decode_q8<T>(row, sub, tb, o);
}

}  // namespace
}  // namespace tona::gpu
