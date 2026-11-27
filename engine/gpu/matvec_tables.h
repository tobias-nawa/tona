// Constants, lookup tables in device memory and small helpers shared by all matvec device code, plus the host-side
// upload of the tables. Part of matvec_dev.h; do not include on its own.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>

#include "model/quant_tables.h"

namespace tona::gpu {
namespace {

constexpr int WAVE = 32;
constexpr int ROWS_PER_BLOCK = 8;  // 8 waves = 256 threads

// Small enough to stay in cache. Every translation unit has its own copies and uploads them itself.
__device__ __attribute__((aligned(16))) uint64_t d_iq2s_grid[1024];
__device__ __attribute__((aligned(16))) uint32_t d_iq3xxs_grid[256];
__device__ __attribute__((aligned(16))) uint32_t d_iq3s_grid[512];
__device__ uint8_t d_ksigns[128];
__device__ int8_t d_kvalues[16];
// Byte j = 0xFF if bit j is set (sign mask of the IQ formats)
__device__ __attribute__((aligned(16))) uint64_t d_kmask8[256];
// d_kmask8[ksigns_iq2xs[i]] for IQ3_XXS
__device__ __attribute__((aligned(16))) uint64_t d_kmask3[128];

// IQ2_XXS grid (2 KB), constant-initialized in the code object of every unit, so it needs no upload and a unit that
// never uses it carries nothing of it. The int8 path reads it from LDS: a kernel with a loop for IQ2_XXS calls
// load_iq2xxs_lds() once per block, before its first barrier (matvec_q8_iq.h, decode_q8<16>).
struct Iq2xxsGrid {
    uint64_t v[256];
};
constexpr Iq2xxsGrid make_iq2xxs_grid() noexcept {
    Iq2xxsGrid g{};
    for (int i = 0; i < 256; i++) g.v[i] = tona::tables::iq2xxs_grid[i];
    return g;
}
__constant__ constinit Iq2xxsGrid d_iq2xxs_grid = make_iq2xxs_grid();
__shared__ __attribute__((aligned(16))) uint64_t lds_iq2xxs[256];
// Every thread of the block; the caller's next barrier makes the table visible
__device__ __forceinline__ void load_iq2xxs_lds() {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) lds_iq2xxs[i] = d_iq2xxs_grid.v[i];
}

__device__ __forceinline__ float h2f(const uint8_t* p) {
    _Float16 h;
    __builtin_memcpy(&h, p, 2);
    return (float)h;
}
__device__ __forceinline__ float h2f_bits(uint32_t bits) { return (float)__builtin_bit_cast(_Float16, (uint16_t)bits); }
__device__ __forceinline__ uint16_t u16(const uint8_t* p) { return (uint16_t)(p[0] | (p[1] << 8)); }

__device__ __forceinline__ void scale_min_k4(int j, const uint8_t* q, int& d, int& m) {
    if (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

__device__ __forceinline__ float wave_sum(float s) {
    for (int off = WAVE / 2; off > 0; off /= 2) s += __shfl_xor(s, off, WAVE);
    return s;
}

// Elements and bytes per block of a format; the sub-blocks of 32 values live inside (see dot_sub). F32, F16 and BF16
// are listed with the block of 32 that dot_sub reads; their rows are sized by row_bytes_of directly.
struct BlockFmt {
    uint32_t type;
    int elems, bytes;
};
constexpr BlockFmt kBlockFmts[] = {
    {0, 32, 128},   {1, 32, 64},    {30, 32, 64},   {8, 32, 34},      {20, 32, 18},  {42, 64, 18},
    {12, 256, 144}, {13, 256, 176}, {14, 256, 210}, {1019, 256, 210}, {22, 256, 82}, {18, 256, 98},
    {21, 256, 110}, {23, 256, 136}, {16, 256, 66},  {40, 64, 36},
};
constexpr BlockFmt block_fmt(uint32_t type) {
    for (const BlockFmt& f : kBlockFmts)
        if (f.type == type) return f;
    return {type, 0, 0};
}
template <uint32_t T>
constexpr int blk_bytes() {
    return block_fmt(T).bytes;
}
template <uint32_t T>
constexpr int blk_elems() {
    return block_fmt(T).elems ? block_fmt(T).elems : 256;
}

int64_t row_bytes_of(uint32_t type, int64_t in) {
    if (type == 0) return in * 4;
    if (type == 1 || type == 30) return in * 2;
    const BlockFmt f = block_fmt(type);
    return f.elems ? in / f.elems * f.bytes : -1;
}

// Byte masks (0xFF per set bit) of the IQ sign table, as the unpacking of the int8 path applies them.
void make_mask_tables(uint64_t (&m8)[256], uint64_t (&m3)[128]) {
    for (int i = 0; i < 256; i++) {
        m8[i] = 0;
        for (int j = 0; j < 8; j++)
            if (i >> j & 1) m8[i] |= 0xffull << (8 * j);
    }
    for (int i = 0; i < 128; i++) m3[i] = m8[tona::tables::ksigns_iq2xs[i]];
}

void upload_tables() {
    static bool done = false;
    if (done) return;
    using namespace tona::tables;
    auto up = [](const void* sym, const void* src, size_t n) {
        if (hipMemcpyToSymbol(sym, src, n) != hipSuccess) throw std::runtime_error("hipMemcpyToSymbol failed");
    };
    up(HIP_SYMBOL(d_iq2s_grid), iq2s_grid, sizeof iq2s_grid);
    up(HIP_SYMBOL(d_iq3xxs_grid), iq3xxs_grid, sizeof iq3xxs_grid);
    up(HIP_SYMBOL(d_iq3s_grid), iq3s_grid, sizeof iq3s_grid);
    up(HIP_SYMBOL(d_ksigns), ksigns_iq2xs, sizeof ksigns_iq2xs);
    up(HIP_SYMBOL(d_kvalues), kvalues_iq4nl, sizeof kvalues_iq4nl);
    uint64_t m8[256], m3[128];
    make_mask_tables(m8, m3);
    up(HIP_SYMBOL(d_kmask8), m8, sizeof m8);
    up(HIP_SYMBOL(d_kmask3), m3, sizeof m3);
    done = true;
}

}  // namespace
}  // namespace tona::gpu
