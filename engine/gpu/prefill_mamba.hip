// Prefill kernels of the Mamba-2 layers (prefill_mamba.h). The scan runs the tokens one after the other with the state
// of 8 rows in the registers of a wave (the form of the decode step), not as the chunked matrix form of the SSD paper:
// the sequential form gives the same bits for any cut of the prompt and for the decode path
// (research/measurements/2026-11-19-bolt-b4.md has the cost and the decision).
#include <format>
#include <stdexcept>

#include "gpu/mamba_dev.h"
#include "gpu/prefill_dev.h"
#include "gpu/prefill_mamba.h"

namespace tona::gpu {

namespace {

// Conv input of position p of the chunk (earlier: ring, before 0: zero)
__device__ __forceinline__ float mamba_conv_in(const MambaConvArgs& a, int64_t p, int64_t c) {
    const int R = a.K - 1 + BMAX;
    if (p >= a.pos0) return a.zx[(p - a.pos0) * a.in_dim + a.off + c] * a.in_scale;
    return p >= 0 ? a.ring[(p % R) * a.ch + c] : 0.0f;
}

// Same order of the sum as op_mamba_conv: oldest tap first, the current token last
// Grid: channels along x, tokens along y (no 64-bit division per element)
__global__ void __launch_bounds__(TPB) k_mamba_conv(MambaConvArgs a) {
    const int64_t c = (int64_t)blockIdx.x * TPB + threadIdx.x, tok = blockIdx.y;
    if (c >= a.ch) return;
    const int64_t t = tok * a.ch + c, pos = a.pos0 + tok;
    float acc = 0;
    for (int k = 0; k < a.K - 1; k++) acc += a.w[c * a.K + k] * mamba_conv_in(a, pos - (a.K - 1 - k), c);
    acc += a.w[c * a.K + a.K - 1] * mamba_conv_in(a, pos, c);
    a.out[t] = silu_f(acc + a.bias[c]);
}

__global__ void __launch_bounds__(TPB) k_mamba_ring(MambaRingArgs a) {
    const int R = a.K - 1 + BMAX;
    const int cnt = a.n < R ? a.n : R;
    const int64_t c = (int64_t)blockIdx.x * TPB + threadIdx.x, tok = a.n - cnt + blockIdx.y;
    if (c >= a.ch) return;
    a.ring[((a.pos0 + tok) % R) * a.ch + c] = a.zx[tok * a.in_dim + a.off + c] * a.in_scale;
}

// Inputs of one token for a wave: the raw dt of the head, x of the rows, B and C of the lane's state columns
struct MbIn {
    float decay, dt, xm, x[MB_R], B[MB_K], C[MB_K];  // xm: x of the row lane / 4 (the lane that stores its output)
};
struct MbAt {
    int h, i0, g, lane;
};
__device__ __forceinline__ MbIn mb_in(const MambaScanArgs& a, int b, const MbAt p) {
    const float* cv = a.conv + (int64_t)b * a.conv_dim;
    MbIn t;
    t.decay = a.decay[(int64_t)b * a.heads + p.h], t.dt = a.dt[(int64_t)b * a.heads + p.h];
    t.xm = cv[p.h * a.P + p.i0 + p.lane / 4];
#pragma unroll
    for (int j = 0; j < MB_R; j++) t.x[j] = cv[p.h * a.P + p.i0 + j];
#pragma unroll
    for (int k = 0; k < MB_K; k++) {
        t.B[k] = cv[a.inner + p.g * MB_N + p.lane + 32 * k];
        t.C[k] = cv[a.inner + a.groups * MB_N + p.g * MB_N + p.lane + 32 * k];
    }
    return t;
}

// One thread per (token, head)
__global__ void __launch_bounds__(TPB) k_mamba_dt(MambaDtArgs a) {
    const int64_t t = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (t >= (int64_t)a.n * a.heads) return;
    const int64_t tok = t / a.heads, h = t % a.heads;
    const MbTok tk = mb_token(a.zx[tok * a.in_dim + a.dt_off + h] * a.in_scale, a.dt_bias[h], a.A[h]);
    a.decay[t] = tk.decay, a.dt[t] = tk.dt;
    if (a.da) a.da[t] = tk.dt * a.A[h];
}

// One wave per (head, 8 rows); 8 waves per block
__global__ void __launch_bounds__(TPB) k_mamba_scan(MambaScanArgs a) {
    const int lane = threadIdx.x % WAVE;
    const int rg = a.P / MB_R, per_group = a.heads / a.groups;
    const int64_t un = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    if (un >= (int64_t)a.heads * rg) return;
    const int h = (int)(un / rg), i0 = (int)(un % rg) * MB_R, g = h / per_group;
    float* st = a.state + ((int64_t)h * a.P + i0) * MB_N;
    MbRows r;
    mb_load(r, st, lane);
    if (a.m > 0) {
        const MbPend pend{a.pdec, a.pxdt, a.pb, a.heads, a.inner, a.groups, BMAX};
        mb_commit(pend, r, {h * a.P + i0, h, g}, a.m, a.half);
    }
    const float Dh = a.D[h];
    // The inputs of the next token are loaded while this one computes: the loads are the latency of the chain otherwise
    MbIn cur{}, next{};
    if (a.n > 0) cur = mb_in(a, 0, {h, i0, g, lane});
    for (int b = 0; b < a.n; b++) {
        if (b + 1 < a.n) next = mb_in(a, b + 1, {h, i0, g, lane});
        float xdt[MB_R];
#pragma unroll
        for (int j = 0; j < MB_R; j++) xdt[j] = cur.x[j] * cur.dt;
        mb_update(r, cur.decay, xdt, cur.B);
        const float ysum = mb_read(r, cur.C);
        if (lane % 4 == 0) a.y[(int64_t)b * a.inner + h * a.P + i0 + lane / 4] = ysum + cur.xm * Dh;
        cur = next;
    }
    mb_store(r, st, lane);
}

// One wave per (token, group)
__global__ void __launch_bounds__(TPB) k_mamba_norm(MambaNormArgs a) {
    const int lane = threadIdx.x % WAVE;
    const int64_t u = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    if (u >= (int64_t)a.n * a.groups) return;
    const int b = (int)(u / a.groups), g = (int)(u % a.groups), gs = a.inner / a.groups;
    const float* yb = a.y + (int64_t)b * a.inner + g * gs;
    const float* zb = a.zx + (int64_t)b * a.in_dim + g * gs;
    float ss = 0;
    for (int i = lane; i < gs; i += WAVE) {
        const float v = yb[i] * silu_f(zb[i] * a.in_scale);
        ss += v * v;
    }
    const float scale = rsqrtf(wave_sum(ss) / gs + a.eps);
    for (int i = lane; i < gs; i += WAVE) {
        const float v = yb[i] * silu_f(zb[i] * a.in_scale);
        a.o[(int64_t)b * a.inner + g * gs + i] = (_Float16)(v * scale * a.w[g * gs + i] * a.out_scale);
    }
}

}  // namespace

void launch_mamba_conv(const MambaConvArgs& a, hipStream_t st) {
    if (a.n > 0) k_mamba_conv<<<dim3(blocks_for(a.ch), a.n), TPB, 0, st>>>(a);
}
void launch_mamba_ring(const MambaRingArgs& a, hipStream_t st) {
    const int R = a.K - 1 + BMAX;
    if (a.n > 0) k_mamba_ring<<<dim3(blocks_for(a.ch), a.n < R ? a.n : R), TPB, 0, st>>>(a);
}
void launch_mamba_dt(const MambaDtArgs& a, hipStream_t st) {
    if (a.n > 0) k_mamba_dt<<<blocks_for((int64_t)a.n * a.heads), TPB, 0, st>>>(a);
}
void launch_mamba_scan(const MambaScanArgs& a, hipStream_t st) {
    k_mamba_scan<<<blocks_for((int64_t)a.heads * (a.P / MB_R) * WAVE), TPB, 0, st>>>(a);
}
void launch_mamba_norm(const MambaNormArgs& a, hipStream_t st) {
    if (a.n > 0) k_mamba_norm<<<blocks_for((int64_t)a.n * a.groups * WAVE), TPB, 0, st>>>(a);
}

void mamba_check_shapes(int heads, int P, int state, int groups, int conv) {
    if (state != MB_N || P % MB_R != 0 || groups <= 0 || heads % groups != 0 || ((heads * P) / groups) % WAVE != 0 ||
        conv < 2)
        throw std::runtime_error(std::format(
            "gpu: the Mamba-2 kernels need state {}, head rows a multiple of {}, a group of a multiple of {} values "
            "and a conv of 2 taps or more; got heads {}, rows {}, state {}, groups {}, conv {}",
            MB_N, MB_R, WAVE, heads, P, state, groups, conv));
}

}  // namespace tona::gpu
