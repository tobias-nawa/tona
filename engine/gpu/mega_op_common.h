// Shared pieces of the megakernel step functions: block size, reductions, the step context.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/lds_sync.h"
#include "gpu/matvec_dev.h"
#include "gpu/mega_dev.h"
#include "gpu/mega_ops.h"
#include "gpu/mega_types.h"

namespace tona::gpu {
namespace {

constexpr int THREADS = WAVE * MWAVES;
// LDS layout of op_attn_wmma: K, V^T and P as f16 for 16 cells each, plus row values, in the same LDS as sm.
// The LDS of the kernel must not grow: with 61.6 KB not all 32 blocks fitted next to the desktop and the barriers
// aborted (see also mega_ops.h, SM_FLOATS).
namespace aw {
using half8 = _Float16 __attribute__((ext_vector_type(8)));
using float8 = float __attribute__((ext_vector_type(8)));
constexpr int HD = 256, CC = 16, LQ = HD + 8;
constexpr int BYTES = CC * LQ * 2 + HD * CC * 2 + 16 * CC * 2 + 48 * 4;
}  // namespace aw
static_assert(aw::BYTES <= SM_FLOATS * 4, "op_attn_wmma must fit in sm");

// Sum or maximum over all waves of the block; red: MWAVES floats in LDS
__device__ __forceinline__ float block_sum(float v, float* red) {
    v = wave_sum(v);
    lds_sync();
    if (threadIdx.x % WAVE == 0) red[threadIdx.x / WAVE] = v;
    lds_sync();
    float t = 0;
    for (int i = 0; i < MWAVES; i++) t += red[i];
    return t;
}
__device__ __forceinline__ float block_max(float v, float* red) {
    v = wave_max(v);
    lds_sync();
    if (threadIdx.x % WAVE == 0) red[threadIdx.x / WAVE] = v;
    lds_sync();
    float t = -INFINITY;
    for (int i = 0; i < MWAVES; i++) t = fmaxf(t, red[i]);
    return t;
}

// 32 consecutive values (one per lane, index i) stored as int8, as k_quantize_q8 does
__device__ __forceinline__ void wave_q8(float v, int64_t i, int8_t* q, Q8Meta* meta) {
    const int lane = threadIdx.x % WAVE;
    const float d = wave_max(fabsf(v)) / 127.0f;
    const int qi = d > 0 ? (int)rintf(v / d) : 0;
    int s = qi;
    for (int off = 8; off > 0; off /= 2) s += __shfl_xor(s, off, WAVE);
    q[i] = (int8_t)qi;
    if (lane == 0) meta[i / 32].d = d, meta[i / 32].s0 = (int16_t)s;
    if (lane == 16) meta[i / 32].s1 = (int16_t)s;
}

struct Ctx {
    int64_t gw, GW, gt, GT, pos;
    int lane, nb;
    int commit, parity, r0;  // DeltaNet commit count and buffer half; first row of the sequence in the pass (n_cells)
    MegaParams* P;
    float* sm;          // LDS: MAX_CELLS + 256 + 64 floats
    uint64_t deadline;  // wall_clock64 by which the whole pass must be done (~0: no deadline)
};

// What the steps need besides the op and the context: the task array, the grid tables in LDS, the LDS buffer of the
// kernel, the parameters of the pass, the barrier buffer, the wall clock rate and the number of the current step.
struct StepEnv {
    const DevTask* tasks;
    Tabs tabs;
    float* sm;
    MegaParams* P;
    unsigned* bar;
    int wall_khz, o;
};

// QSA selects pool blocks as soon as there are more blocks than top_k holds (i[5] = block size, i[7] = top_k)
__device__ __forceinline__ bool qsa_select(const MegaOp& op, int64_t pos) {
    const int blk = op.i[5];
    const int64_t n_pool = (pos + 1) / blk, n_top = std::min<int64_t>(n_pool, op.i[7] / blk);
    return n_pool > n_top;
}

}  // namespace
}  // namespace tona::gpu
