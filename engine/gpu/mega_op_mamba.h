// Step functions of the Mamba-2 layers (Bolt): causal conv with SiLU, selective scan with the replay commit, gated
// group norm. The in-projection and out-projection are ordinary matvec phases around them. Part of the kernel
// translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); only the profiles of architectures with Mamba-2 include it.
//
// Layout of one token's in-projection output (zxbcdt, `in_dim` floats): [z (inner) | x B C (conv_dim) | dt (heads)],
// with conv_dim = inner + 2 * groups * N. Conv output: [x (inner) | B (groups * N) | C (groups * N)].
// The NVFP4 per-tensor scale of the in-projection (in_scale) is applied where zxbcdt is read, that of the
// out-projection (out_scale) before the int8 quantization of the norm output (the matvec is linear, so the product is
// the same); both are 1 for the other types.
#pragma once

#include "gpu/mamba_dev.h"
#include "gpu/mega_op_common.h"

namespace tona::gpu {
namespace {

// Causal depthwise conv + bias + SiLU as dn_conv1 (mega_op_dn.h): the ring holds H + BMAX input positions (index
// position mod R), so tokens of a pass that are later discarded cannot overwrite inputs the next pass still reads, and
// the conv state needs no commit. Positions before 0 read zero, so a new sequence needs no reset of the ring.
// p: 0 zxbcdt, 1 ring [R][conv_dim], 2 weight [conv_dim][K], 3 bias, 4 out [nb][conv_dim].
// i: 0 conv_dim, 1 K, 2 in_dim, 3 offset of the conv input in a row (inner). f[0]: in_scale.
__device__ __noinline__ void op_mamba_conv(const MegaOp& op, const Ctx& c) {
    const float* zx = (const float*)op.p[0];
    float* ring = (float*)op.p[1];
    const float *w = (const float*)op.p[2], *bias = (const float*)op.p[3];
    float* out = (float*)op.p[4];
    const int ch = op.i[0], K = op.i[1], in_dim = op.i[2], off = op.i[3];
    const int H = K - 1, R = H + BMAX;
    const float sc = op.f[0];
    for (int64_t t = c.gt; t < (int64_t)c.nb * ch; t += c.GT) {
        const int b = (int)(t / ch), cc = (int)(t % ch);
        const int64_t pos = c.pos + b;
        float acc = 0;
        for (int k = 0; k < H; k++) {
            const int back = H - k;
            const float x = back <= b         ? zx[(int64_t)(b - back) * in_dim + off + cc] * sc
                            : pos - back >= 0 ? ring[((pos - back) % R) * ch + cc]
                                              : 0.0f;
            acc += w[(int64_t)cc * K + k] * x;
        }
        const float cur = zx[(int64_t)b * in_dim + off + cc] * sc;
        acc += w[(int64_t)cc * K + H] * cur;
        ring[(pos % R) * ch + cc] = cur;
        out[t] = silu_f(acc + bias[cc]);
    }
}

// Where the scan finds its inputs and keeps its state
struct MambaScanIn {
    const float *zx, *conv, *dt_bias, *A, *D;
    float *state, *pdec, *pxdt, *pb, *y;
    int heads, P, groups, in_dim, dt_off, conv_dim, inner;
    float in_scale;
};

__device__ __forceinline__ MambaScanIn mamba_scan_in(const MegaOp& op) {
    return {(const float*)op.p[0],
            (const float*)op.p[1],
            (const float*)op.p[2],
            (const float*)op.p[3],
            (const float*)op.p[4],
            (float*)op.p[5],
            (float*)op.p[6],
            (float*)op.p[7],
            (float*)op.p[8],
            (float*)op.p[9],
            op.i[0],
            op.i[1],
            op.i[3],
            op.i[4],
            op.i[5],
            op.i[6],
            op.i[7],
            op.f[0]};
}

// What a wave works on: head h, rows i0 ... i0 + MB_R - 1, group g of the head
struct MambaUnit {
    int h, i0, g, lane;
};

// One token of a unit: its inputs, the update of the rows, the output, and what the replay stores (decay by the first
// row group of a head, B by the first head of a group)
__device__ __forceinline__ void mamba_scan_token(const MambaScanIn& a, const MambaUnit& u, MbRows& r, int b, int par) {
    const bool writes_b = u.i0 == 0 && u.h % (a.heads / a.groups) == 0;
    const float* cv = a.conv + (int64_t)b * a.conv_dim;
    const MbTok tk = mb_token(a.zx[(int64_t)b * a.in_dim + a.dt_off + u.h] * a.in_scale, a.dt_bias[u.h], a.A[u.h]);
    float xv[MB_R], xdt[MB_R], B[MB_K], C[MB_K];
#pragma unroll
    for (int j = 0; j < MB_R; j++) xv[j] = cv[u.h * a.P + u.i0 + j], xdt[j] = xv[j] * tk.dt;
#pragma unroll
    for (int k = 0; k < MB_K; k++) {
        B[k] = cv[a.inner + u.g * MB_N + u.lane + 32 * k];
        C[k] = cv[a.inner + a.groups * MB_N + u.g * MB_N + u.lane + 32 * k];
    }
    mb_update(r, tk.decay, xdt, B);
    const float ysum = mb_read(r, C);
#pragma unroll
    for (int j = 0; j < MB_R; j++) {
        if (u.lane == mb_row_lane(j)) a.y[(int64_t)b * a.inner + u.h * a.P + u.i0 + j] = ysum + xv[j] * a.D[u.h];
        if (u.lane == j) a.pxdt[((int64_t)par * BMAX + b) * a.inner + u.h * a.P + u.i0 + j] = xdt[j];
    }
    if (u.lane == 8 && u.i0 == 0) a.pdec[((int64_t)par * BMAX + b) * a.heads + u.h] = tk.decay;
    if (writes_b) {
#pragma unroll
        for (int k = 0; k < MB_K; k++)
            a.pb[(((int64_t)par * BMAX + b) * a.groups + u.g) * MB_N + u.lane + 32 * k] = B[k];
    }
}

// Selective scan for the nb tokens of the pass. The stored state lags one pass behind: this pass first commits the
// `commit` accepted tokens of the previous pass from what it stored (decay per head, x * dt per row, B per group; half
// parity ^ 1), then runs its own tokens from registers and stores their inputs in the half `parity`. Every token takes
// the same steps wherever it sits in a pass and however many tokens were committed before it.
// p: 0 zxbcdt, 1 conv out, 2 dt bias, 3 A, 4 D, 5 state [heads][P][N], 6 pending decay [2][BMAX][heads],
//    7 pending x * dt [2][BMAX][inner], 8 pending B [2][BMAX][groups * N], 9 y out [nb][inner].
// i: 0 heads, 1 P (a multiple of MB_R), 3 groups, 4 in_dim, 5 offset of dt in a row, 6 conv_dim, 7 inner; N = 128.
// f[0]: in_scale. Unit = (head, MB_R rows).
template <int NB>
__device__ __noinline__ void op_mamba_scan(const MegaOp& op, const Ctx& c) {
    const MambaScanIn a = mamba_scan_in(op);
    const int rg = a.P / MB_R, per_group = a.heads / a.groups;
    const int m = c.commit, par = c.parity, nb = c.nb;
    for (int64_t un = c.gw; un < (int64_t)a.heads * rg; un += c.GW) {
        const MambaUnit u{(int)(un / rg), (int)(un % rg) * MB_R, (int)(un / rg) / per_group, c.lane};
        float* st = a.state + ((int64_t)u.h * a.P + u.i0) * MB_N;
        MbRows r;
        mb_load(r, st, c.lane);
        if (m > 0) {
            const MbPend pend{a.pdec, a.pxdt, a.pb, a.heads, a.inner, a.groups, BMAX};
            mb_commit(pend, r, {u.h * a.P + u.i0, u.h, u.g}, m, par ^ 1);
            mb_store(r, st, c.lane);
        }
        constexpr int kUnroll = NB > 4 ? 1 : NB;  // as op_dn_recur: unrolled 8 times the loads of all tokens spill
#pragma unroll kUnroll
        for (int b = 0; b < NB; b++) {
            if (b >= nb) continue;
            mamba_scan_token(a, u, r, b, par);
        }
    }
}

// Gate and group norm: v = y * silu(z), per group of inner / groups values v / sqrt(mean(v^2) + eps) * w, times
// out_scale, int8 for the out-projection (and as float when o is given). One block per (token, group) at a time: the
// group is a multiple of 32 values, the sum of squares is a block sum.
// p: 0 y [nb][inner], 1 zxbcdt, 2 norm weight [inner], 3 int8, 4 meta, 5 float out (may be null).
// i: 0 inner, 1 groups, 2 in_dim. f: 0 eps, 1 in_scale, 2 out_scale.
__device__ __noinline__ void op_mamba_norm(const MegaOp& op, const Ctx& c) {
    const float *y = (const float*)op.p[0], *zx = (const float*)op.p[1], *w = (const float*)op.p[2];
    int8_t* q8 = (int8_t*)op.p[3];
    Q8Meta* meta = (Q8Meta*)op.p[4];
    float* o = (float*)op.p[5];
    const int inner = op.i[0], G = op.i[1], in_dim = op.i[2], gs = inner / G;
    for (int64_t u = blockIdx.x; u < (int64_t)c.nb * G; u += gridDim.x) {
        const int b = (int)(u / G), g = (int)(u % G);
        const float* yb = y + (int64_t)b * inner + g * gs;
        const float* zb = zx + (int64_t)b * in_dim + g * gs;
        float ss = 0;
        for (int i = threadIdx.x; i < gs; i += THREADS) {
            const float v = yb[i] * silu_f(zb[i] * op.f[1]);
            ss += v * v;
        }
        const float scale = rsqrtf(block_sum(ss, c.sm) / gs + op.f[0]);
        for (int i = threadIdx.x; i < gs; i += THREADS) {
            const float v = yb[i] * silu_f(zb[i] * op.f[1]);
            const float ov = v * scale * w[g * gs + i] * op.f[2];
            const int64_t at = (int64_t)b * inner + g * gs + i;
            if (o) o[at] = ov;
            wave_q8(ov, at, q8, meta);
        }
    }
}

// Runs the step if it is a Mamba-2 step; false for the other groups.
template <int NB>
__device__ __forceinline__ bool dispatch_mamba(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_MAMBA_CONV:
            op_mamba_conv(op, c);
            return true;
        case OP_MAMBA_SCAN:
            op_mamba_scan<NB>(op, c);
            return true;
        case OP_MAMBA_NORM:
            op_mamba_norm(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
