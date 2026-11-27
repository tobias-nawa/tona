// Step functions of the latent MoE (Nemotron-H / Bolt): the experts are ungated (up and down, squared ReLU) and work in
// a latent space; routing (mega_op_route_sig.h), the matvecs (latent down and up, expert groups, shared expert) and the
// expert copies are the shared steps. Four steps are new:
//   OP_ACT_RELU2     act = relu(u)^2 over the occupied expert places and the shared expert, int8 (the Coder's
//                    OP_ACT_QUANT computes silu(g) * u from two inputs);
//   OP_MOE_SUM_LAT   the fixed-point sum of w_e * out_e in the latent space (bit-identical whichever side computes
//   which
//                    expert, as OP_MOE_SUM), float and int8 for the matvec of lat_up; no shared expert, no gate;
//   OP_QUANT_LAT     the latent input (float) to int8 for the expert matvecs (OP_QUANT lives in the Coder's step
//   group); OP_MOE_ADD       y = a + b: the projected routed experts plus the shared expert, as the oracle adds them.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// u, act, q: i[0] = n = UMAX * ff + sff values per token, i[1] = ff, i[2] = distance of the tokens in the buffers (the
// buffers are sized for the widest layer; 0: n); only the n_u occupied places and the shared expert.
// p: u, act (float out), q, meta
__device__ __noinline__ void op_act_relu2(const MegaOp& op, const Ctx& c) {
    const float* u = (const float*)op.p[0];
    float* out = (float*)op.p[1];
    int8_t* q = (int8_t*)op.p[2];
    Q8Meta* meta = (Q8Meta*)op.p[3];
    const int n = op.i[0], ff = op.i[1];
    const int64_t stride = op.i[2] ? op.i[2] : n;
    const int64_t used = (int64_t)c.P->n_u * ff / 32, per = used + (n - UMAX * ff) / 32;
    for (int64_t w = c.gw; w < c.nb * per; w += c.GW) {
        const int64_t b = w / per, r = w % per;
        const int64_t i = b * stride + (r < used ? r * 32 : (int64_t)UMAX * ff + (r - used) * 32) + c.lane;
        const float v = u[i];
        const float a = v > 0.0f ? v * v : 0.0f;  // as the oracle's relu_sq
        out[i] = a;
        wave_q8(a, i, q, meta);
    }
}

// p: ed (UMAX * lat per token), y (float, may be null), q, meta, cpu (fixed-point sums of the CPU's experts, or null);
// i[0] = UMAX, i[1] = lat (a multiple of 32). The weights of the pass are P->ew; experts that were not selected (weight
// 0) have no valid output and are skipped.
__device__ __noinline__ void op_moe_sum_lat(const MegaOp& op, const Ctx& c) {
    const float* ed = (const float*)op.p[0];
    float* y = (float*)op.p[1];
    int8_t* q = (int8_t*)op.p[2];
    Q8Meta* meta = (Q8Meta*)op.p[3];
    const int lat = op.i[1], nu = c.P->n_u;
    float* w_all = c.sm;  // the weights once per block into LDS
    for (int k = threadIdx.x; k < c.nb * UMAX; k += THREADS) w_all[k] = c.P->ew[k];
    lds_sync();
    for (int64_t t = c.gt; t < (int64_t)c.nb * lat; t += c.GT) {  // whole waves: nb * lat and GT are multiples of 32
        const int b = (int)(t / lat);
        const int64_t i = t % lat;
        const float* w = w_all + b * UMAX;
        int64_t a = c.P->n_cpu ? ((const int64_t*)op.p[4])[t] : 0;
        for (int e = 0; e < nu; e++)
            if (w[e] != 0.0f) a += tona::det::to_fix(w[e] * ed[((int64_t)b * UMAX + e) * lat + i]);
        const float v = tona::det::from_fix(a);
        if (y) y[t] = v;
        wave_q8(v, t, q, meta);
    }
}

// p: x, q, meta; i[0] = values per token (a multiple of 32, also the distance of the tokens in x)
__device__ __noinline__ void op_quant_lat(const MegaOp& op, const Ctx& c) {
    const float* x = (const float*)op.p[0];
    int8_t* q = (int8_t*)op.p[1];
    Q8Meta* meta = (Q8Meta*)op.p[2];
    const int64_t n = (int64_t)c.nb * op.i[0];
    for (int64_t w = c.gw; w * 32 < n; w += c.GW) {
        const int64_t i = w * 32 + c.lane;
        wave_q8(x[i], i, q, meta);
    }
}

// p: a, b, y; i[0] = d values per token
__device__ __noinline__ void op_moe_add(const MegaOp& op, const Ctx& c) {
    const float* a = (const float*)op.p[0];
    const float* b = (const float*)op.p[1];
    float* y = (float*)op.p[2];
    for (int64_t t = c.gt; t < (int64_t)c.nb * op.i[0]; t += c.GT) y[t] = a[t] + b[t];
}

// Runs the step if it is a latent MoE step; false for the other groups.
__device__ __forceinline__ bool dispatch_moe_latent(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_ACT_RELU2:
            op_act_relu2(op, c);
            return true;
        case OP_MOE_SUM_LAT:
            op_moe_sum_lat(op, c);
            return true;
        case OP_QUANT_LAT:
            op_quant_lat(op, c);
            return true;
        case OP_MOE_ADD:
            op_moe_add(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
