// Step functions of the MTP head: input stage and pick.
// Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"
#include "gpu/mega_op_mtp_dyn.h"

namespace tona::gpu {
namespace {

// MTP input: one block per (token b, stream g). p: e [nb][d], enorm [d], hnorm [hc*d], x [nb][hc][2d], int8, meta;
// h: hc*d per token from P->mtp_h (rows of the target or of the last MTP step), null: 0 (position 0)
__device__ __noinline__ void op_mtp_in(const MegaOp& op, const Ctx& c) {
    const float* e = (const float*)op.p[0];
    const float* wn_e = (const float*)op.p[1];
    const float* wn_h = (const float*)op.p[2];
    float* x = (float*)op.p[3];
    int8_t* q = (int8_t*)op.p[4];
    Q8Meta* meta = (Q8Meta*)op.p[5];
    const int hc = op.i[0], d = op.i[1];
    const float* H = c.P->mtp_h;
    for (int k = blockIdx.x; k < c.nb * hc; k += gridDim.x) {
        const int b = k / hc, g = k % hc;
        const float* eb = e + (int64_t)b * d;
        const float* hb = H ? H + ((int64_t)b * hc + g) * d : nullptr;
        float se = 0, sh = 0;
        for (int j = threadIdx.x; j < d; j += THREADS) {
            se += eb[j] * eb[j];
            if (hb) sh += hb[j] * hb[j];
        }
        const float fe = rsqrtf(block_sum(se, c.sm) / d + op.f[0]);
        const float fh = rsqrtf(block_sum(sh, c.sm) / d + op.f[0]);
        float* xk = x + (int64_t)k * 2 * d;
        for (int j = threadIdx.x; j < 2 * d; j += THREADS) {
            const float v = j < d ? eb[j] * fe * wn_e[j] : (hb ? hb[j - d] * fh * wn_h[(int64_t)g * d + j - d] : 0.0f);
            xk[j] = v;
            wave_q8(v, (int64_t)k * 2 * d + j, q, meta);
        }
    }
}

// Argmax (smaller row on a tie) over all threads of the block: bv/br in, the block's best out. red/redi: one slot per
// wave in LDS.
__device__ __forceinline__ void pick_reduce(float& bv, int& br, float* red, int* redi) {
    const int t = threadIdx.x;
    for (int off = WAVE / 2; off > 0; off /= 2) {
        const float ov = __shfl_xor(bv, off, WAVE);
        const int orr = __shfl_xor(br, off, WAVE);
        if (ov > bv || (ov == bv && orr < br)) bv = ov, br = orr;
    }
    lds_sync();
    if (t % WAVE == 0) red[t / WAVE] = bv, redi[t / WAVE] = br;
    lds_sync();
    bv = red[0], br = redi[0];
    for (int w = 1; w < MWAVES; w++)
        if (red[w] > bv || (red[w] == bv && redi[w] < br)) bv = red[w], br = redi[w];
}

// Step i[2] = 0: every block over its slice of the rows -> {row, max, sum of exp(. - max)} into part[block]. The
// logits are clamped (NaN = -FLT_MAX, +-inf = +-FLT_MAX), so the first valid row always wins at the least and the row
// is a real one even if every logit is NaN; the NaN count of the block goes into bit 30 of the row.
__device__ __forceinline__ void mtp_pick_part(const MegaOp& op, const Ctx& c, float* part) {
    const int t = threadIdx.x;
    const float* lg = (const float*)op.p[0] + (int64_t)(c.nb - 1) * op.i[1];
    const int32_t* map = (const int32_t*)op.p[1];
    const int n = op.i[0], per = (n + gridDim.x - 1) / gridDim.x, r0 = blockIdx.x * per, r1 = min(n, r0 + per);
    float bv = -INFINITY, nan_n = 0;
    int br = n;
    for (int r = r0 + t; r < r1; r += THREADS) {
        if (map && map[r] < 0) continue;
        const float v = lg[r];
        nan_n += v != v;
        if (logit_clamp(v) > bv) bv = logit_clamp(v), br = r;
    }
    pick_reduce(bv, br, c.sm, (int*)(c.sm + MWAVES));
    float s = 0;
    for (int r = r0 + t; r < r1; r += THREADS)
        if (!map || map[r] >= 0) s += __expf(logit_clamp(lg[r]) - bv);
    s = block_sum(s, c.sm + 2 * MWAVES);
    const float nan_b = block_sum(nan_n, c.sm + 3 * MWAVES);
    if (t == 0)
        part[3 * blockIdx.x] = __int_as_float(br | (nan_b > 0 ? 1 << 30 : 0)), part[3 * blockIdx.x + 1] = bv,
                 part[3 * blockIdx.x + 2] = s;
}

// Step i[2] = 1: block 0 merges the parts of all blocks into out {row, max, sum, blocks with NaN (int)}
__device__ __forceinline__ void mtp_pick_merge(const MegaOp& op, const Ctx& c, const float* part) {
    const int t = threadIdx.x, G = gridDim.x;
    float bv = -INFINITY;
    int br = 1 << 30, nan_flag = 0;
    if (t < G) {
        const int raw = __float_as_int(part[3 * t]);
        bv = part[3 * t + 1], br = raw & ((1 << 30) - 1), nan_flag = raw >> 30;
    }
    pick_reduce(bv, br, c.sm, (int*)(c.sm + MWAVES));
    float s = t < G && part[3 * t + 1] > -INFINITY ? part[3 * t + 2] * __expf(part[3 * t + 1] - bv) : 0.0f;
    s = block_sum(s, c.sm + 2 * MWAVES);
    const float nan_blocks = block_sum((float)nan_flag, c.sm + 3 * MWAVES);
    float* out = (float*)op.p[3];
    // i[3] = 1: the map is written by the program (dynamic rows), the host cannot translate the row, so out[0] is the
    // token
    if (op.i[3]) br = max(((const int32_t*)op.p[1])[br < op.i[0] ? br : 0], 0);
    if (t == 0) out[0] = __int_as_float(br), out[1] = bv, out[2] = s, ((int*)out)[3] = (int)nan_blocks;
}

// MTP draft, two steps (i[2]): the blocks reduce their slices of the rows, then block 0 merges. p: logits
// [nb][stride], token per row (null: row = token; < 0: free), part, out; i: rows, stride, step, token out. On a tie the
// smaller row wins (as std::max_element). A single block over all rows was the bottleneck at 41K rows.
__device__ __noinline__ void op_mtp_pick(const MegaOp& op, const Ctx& c) {
    float* part = (float*)op.p[2];
    if (op.i[2] == 0) {
        mtp_pick_part(op, c, part);
        return;
    }
    if (blockIdx.x != 0) return;
    mtp_pick_merge(op, c, part);
}

// Runs the step if it is an MTP step; false for the other groups.
__device__ __forceinline__ bool dispatch_mtp(const MegaOp& op, const Ctx& c) {
    switch (op.kind) {
        case OP_MTP_IN:
            op_mtp_in(op, c);
            return true;
        case OP_MTP_PICK:
            op_mtp_pick(op, c);
            return true;
        case OP_MTP_PROJ:
            op_mtp_proj(op, c);
            return true;
        case OP_MTP_DYN:
            op_mtp_dyn(op, c);
            return true;
        default:
            return false;
    }
}

}  // namespace
}  // namespace tona::gpu
