// Unit loops of a task: one loop per format, run by the waves of a block over the units of the task. Part of
// matvec_dev.h; do not include on its own.
#pragma once

#include "gpu/matvec_q8_row.h"
#include "gpu/matvec_rows_float.h"
#include "gpu/matvec_task.h"

namespace tona::gpu {
namespace {

// The format is chosen once per task, then each format has its own row loop. With a switch per row the compiler
// held the registers of all formats at the same time: 218 VGPRs, occupancy 6 instead of 16 waves per SIMD.

// Float path, one row times one vector. Type 31 stands for BF16 rows whose length is not a multiple of 8.
template <uint32_t T>
__device__ __forceinline__ float row_dot(const uint8_t* row, const float* x, int in, int lane, uint8_t* stage) {
    if constexpr (T == 14)
        return row_q6k(row, x, in, lane);
    else if constexpr (T == 30)
        return row_bf16(row, x, in, lane);
    else if constexpr (T == 31)
        return row_generic<30>(row, x, in, lane, stage);
    else
        return row_generic<T>(row, x, in, lane, stage);
}

// Float path with few rows: unit = (row, token)
template <uint32_t T>
__device__ __forceinline__ void rows_loop_tsplit(const DevTask& t0, int64_t r0, int64_t W, int lane, uint8_t* stage,
                                                 const Batch& bt) {
    for (; r0 < t0.units; r0 += W) {
        const int64_t r = r0 / t0.tsplit;
        const int b = (int)(r0 % t0.tsplit);
        if (b >= bt.nb) continue;
        const uint8_t* row = t0.w + r * t0.row_bytes;
        const float s = row_dot<T>(row, t0.x + b * t0.xb, t0.in, lane, stage);
        if (lane == 0) t0.y[b * t0.yb + r] = epilogue(t0, s);
    }
}

// Float path with several vectors: the row is read again per vector (the weights then come from cache)
template <uint32_t T, int NB>
__device__ __forceinline__ void rows_loop_vectors(const DevTask& t0, int64_t r0, int64_t W, int lane, uint8_t* stage,
                                                  const Batch& bt) {
    for (; r0 < t0.units; r0 += W) {
        DevTask t = t0;
        uint32_t mask;
        const int64_t r = resolve_unit(t, r0, bt, mask);
        const uint8_t* row = t.w + r * t.row_bytes;
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b >= bt.nb || !(mask >> b & 1)) continue;
            const float s = row_dot<T>(row, t.x + b * t.xb, t.in, lane, stage);
            if (lane == 0) t.y[b * t.yb + r] = epilogue(t, s);
        }
    }
}

template <uint32_t T, int NB>
__device__ __forceinline__ void rows_loop(const DevTask& t0, int64_t r0, int64_t W, int lane, uint8_t* stage,
                                          const Batch& bt) {
    if (t0.tsplit)
        rows_loop_tsplit<T>(t0, r0, W, lane, stage, bt);
    else
        rows_loop_vectors<T, NB>(t0, r0, W, lane, stage, bt);
}

// int8 path. LX: x comes from LDS (Batch::lx) instead of device memory.
template <uint32_t T, int NB, bool LX = false>
__device__ __forceinline__ void units_loop_q8(const DevTask& t0, int64_t u0, int64_t W, int lane, const Tabs& tb,
                                              const Batch& bt) {
    const int per = WAVE / t0.L, j = lane % t0.L;
    for (; u0 < t0.units; u0 += W) {
        DevTask t = t0;
        uint32_t mask;
        const int64_t u = resolve_unit(t, u0, bt, mask);
        const int64_t r = u * per + lane / t.L;
        if (r < t.rows) {  // whole L-groups drop out together, the sum stays within the group
            float s[NB];
            const Q8RowArgs a{
                row_at<T>(t.w, r, t.row_bytes), LX ? bt.lx : t.xq, LX ? bt.lm : t.xm, t.in, j, t.L, bt.nb, t.xb, mask};
            row_q8<T, NB>(a, tb, s);
            if (j == 0) {
#pragma unroll
                for (int b = 0; b < NB; b++)
                    if (b < bt.nb && (mask >> b & 1)) t.y[b * t.yb + r] = epilogue(t, s[b]);
            }
        }
    }
}

}  // namespace
}  // namespace tona::gpu
