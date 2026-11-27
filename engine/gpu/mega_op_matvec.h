// The matvec step of the megakernel program: stage x in LDS where the phase allows it, then run the tasks of the phase.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// x quantized to int8 in LDS from float (p[2], i[6] values per token), as OP_QUANT does
__device__ __forceinline__ void matvec_quantize_x(const MegaOp& op, const Ctx& c, int8_t* lx, Q8Meta* lm) {
    const int64_t xs = op.i[0];
    const float* xf = (const float*)op.p[2];
    const int g = op.i[6] / 32;
    for (int u = threadIdx.x / WAVE; u < c.nb * g; u += MWAVES) {
        const int64_t i = (u / g) * xs + (u % g) * 32 + c.lane;
        wave_q8(xf[i], i, lx, lm);
    }
}

// x (p[0]) and its metadata (p[1]) copied from device memory into LDS
__device__ __forceinline__ void matvec_copy_x(const MegaOp& op, const Ctx& c, int8_t* lx, Q8Meta* lm) {
    const int64_t xs = op.i[0];
    const uint4* xsrc = (const uint4*)op.p[0];
    for (int64_t i = threadIdx.x; i < c.nb * xs / 16; i += THREADS) reinterpret_cast<uint4*>(lx)[i] = xsrc[i];
    const uint2* msrc = (const uint2*)op.p[1];
    for (int64_t i = threadIdx.x; i < c.nb * xs / 32; i += THREADS) reinterpret_cast<uint2*>(lm)[i] = msrc[i];
}

// i[0] > 0: all int8 tasks read the same x (p[0], metadata p[1], i[0] bytes per token) and it goes into LDS.
// i[6] > 0: x is quantized here from float instead (saves the OP_QUANT step and its barrier; also with one token).
// The host decides for PNB tokens; if nb tokens do not fit in LDS the pass reads x from memory (i[6]: fits for BMAX,
// checked at build time).
__device__ __forceinline__ void matvec_stage_x(const MegaOp& op, const Ctx& c, float* sm, Batch& bt) {
    if (!(op.i[0] > 0 && (c.nb > 1 || op.i[6] > 0) &&
          c.nb * (op.i[0] + op.i[0] / 32 * (int64_t)sizeof(Q8Meta)) <= XSTAGE_BYTES))
        return;
    int8_t* lx = reinterpret_cast<int8_t*>(sm);
    Q8Meta* lm = reinterpret_cast<Q8Meta*>(lx + c.nb * (int64_t)op.i[0]);
    if (op.i[6] > 0)
        matvec_quantize_x(op, c, lx, lm);
    else
        matvec_copy_x(op, c, lx, lm);
    lds_sync();
    bt.lx = lx, bt.lm = lm;
}

template <int NB, Q6Layout L, ExtraQuants X>
__device__ __forceinline__ void run_matvec(const MegaOp& op, const Ctx& c, const StepEnv env) {
    // i[7] = 1: without block 0 (it waits for the CPU in host.wait before)
    if (op.i[7] == 1 && blockIdx.x == 0) return;
    // n_u = 0 (all experts of the layer on the CPU): no unit, otherwise the group would read an old slot
    Batch bt{env.P->slot, env.P->umask, c.pos, c.nb, env.P->n_u};
    bt.wide = env.P->wide_min > 0 && c.nb >= env.P->wide_min && op.cond != 1;
    matvec_stage_x(op, c, env.sm, bt);
    if (op.i[7] == 1)
        run_phase<NB, L, X>(PhaseRange{env.tasks, op.a, op.b},
                            WaveCtx{c.gw - MWAVES, c.GW - MWAVES, c.lane, nullptr, env.tabs}, bt);
    else
        run_phase<NB, L, X>(PhaseRange{env.tasks, op.a, op.b}, WaveCtx{c.gw, c.GW, c.lane, nullptr, env.tabs}, bt);
}

}  // namespace
}  // namespace tona::gpu
