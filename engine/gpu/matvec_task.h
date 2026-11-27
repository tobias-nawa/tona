// Task of the matvec chain as the device sees it (DevTask), the per-pass state (Batch) and the small helpers that
// resolve a unit of a task. Part of matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_q8_core.h"
#include "gpu/matvec_tables.h"

namespace tona::gpu {
namespace {

#ifndef CHAIN_WAVES
#define CHAIN_WAVES 16
#endif

// One task of the chain; base = position of its first row in the flat row sequence of the phase.
// int8 path (xq != null): unit = group of 32/L rows that one wave computes at once; otherwise one row.
// Variable per token (megakernel): w += slots[slot_e] * slot_stride (expert slot), y += pos * y_pos_stride
// (KV cache). act_scale != 0: y = silu(y * act_scale).
struct DevTask {
    const uint8_t* w;
    const float* x;
    float* y;
    const int8_t* xq;
    const Q8Meta* xm;
    int64_t rows, units, row_bytes, base;
    uint32_t type;
    int32_t in, L;
    int32_t slot_e = -1;
    int64_t slot_stride = 0, y_pos_stride = 0;
    float act_scale = 0;
    // Group of n_exp experts of the same shape: units [e * upe, (e + 1) * upe) belong to expert e with slot
    // slots[slot_e + e]; x, xq, xm and y move by the strides per expert (elements resp. bytes)
    int32_t n_exp = 1, upe = 0;
    int64_t x_e = 0, y_e = 0;
    // several vectors (verify pass): vector b reads x + b * xb (int8 likewise, metadata xb/32), writes y + b * yb
    int64_t xb = 0, yb = 0;
    // float path with few rows: unit = (row, token), units = rows * tsplit (tokens per pass). Otherwise one wave
    // computed all tokens of a row one after the other, which took 43 us in a single block for the 4 rows of 10,240
    // of the inject in gr.down.
    int32_t tsplit = 0;
    // 1: dense int8 task with a valid float x (x, stride xb): on a wide pass (Batch::wide) it runs through
    // wide_loop (f16 WMMA)
    int32_t wide = 0;
};

// Variable per launch: position, number of vectors, expert slots and masks (bit b: vector b uses expert e)
struct Batch {
    const int32_t* slots = nullptr;
    const uint32_t* masks = nullptr;
    int64_t pos = 0;
    int nb = 1;
    // >= 0: expert groups compute only their first n_u experts (units at run time), possibly none
    int n_u = -1;
    // != null: x of all int8 tasks of the phase is in LDS (same strides as in device memory)
    const int8_t* lx = nullptr;
    const Q8Meta* lm = nullptr;
    // Pass wide enough for wide_loop (nb >= TONA_WIDE_MIN): tasks with DevTask::wide compute 16 rows per wave by WMMA
    bool wide = false;
};

// Resolve the pointers of unit u of the task (expert group); returns the unit within the matrix and in mask the
// vectors that need it
__device__ __forceinline__ int64_t resolve_unit(DevTask& t, int64_t u, const Batch& bt, uint32_t& mask) {
    mask = ~0u;
    if (t.n_exp <= 1) return u;
    const int32_t* slots = bt.slots;
    const int e = (int)((uint32_t)u / (uint32_t)t.upe);
    if (bt.masks) mask = bt.masks[t.slot_e + e];
    t.w += (int64_t)slots[t.slot_e + e] * t.slot_stride;
    t.x += e * t.x_e;
    if (t.xq) t.xq += e * t.x_e, t.xm += e * t.x_e / 32;
    t.y += e * t.y_e;
    return u - (int64_t)e * t.upe;
}

__device__ __forceinline__ float epilogue(const DevTask& t, float s) {
    if (t.act_scale != 0.0f) {
        const float v = s * t.act_scale;
        return v / (1.0f + __expf(-v));
    }
    return s;
}

// All grid tables once per block into LDS (14 KB), for the int8 tasks of the chain
#define TABS_DECL                                                                                     \
    __shared__ __attribute__((aligned(16))) uint64_t lds_iq2s[1024], lds_m8[256], lds_m3[128];        \
    __shared__ __attribute__((aligned(16))) uint32_t lds_iq3xxs[256], lds_iq3s[512];                  \
    {                                                                                                 \
        auto cp = [](auto* dst, const auto* src, int n) {                                             \
            const uint4* s4 = reinterpret_cast<const uint4*>(src);                                    \
            uint4* d4 = reinterpret_cast<uint4*>(dst);                                                \
            for (int i = threadIdx.x; i < n * (int)sizeof(*src) / 16; i += blockDim.x) d4[i] = s4[i]; \
        };                                                                                            \
        cp(lds_iq2s, d_iq2s_grid, 1024), cp(lds_m8, d_kmask8, 256), cp(lds_m3, d_kmask3, 128);        \
        cp(lds_iq3xxs, d_iq3xxs_grid, 256), cp(lds_iq3s, d_iq3s_grid, 512);                           \
        __syncthreads();                                                                              \
    }                                                                                                 \
    const Tabs tabs { lds_iq2s, lds_iq3xxs, lds_iq3s, lds_m8, lds_m3 }

}  // namespace
}  // namespace tona::gpu
