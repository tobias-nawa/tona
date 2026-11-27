// Prefill kernels of the DeltaNet layers: causal conv, L2 norm of q/k, the delta rule (two forms) and the output norm.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"
#include "gpu/prefill_split.h"

namespace tona::gpu {

namespace {

// Causal conv + SiLU of channel c at chunk token t. The order of the sum is fixed (oldest tap first, the newest last):
// the fused kernel below and the cut invariance (docs/use/prompt-cache.md) rely on the same value for the same input.
__device__ __forceinline__ float dn_conv_at(const DnConvArgs& a, int64_t t, int64_t c) {
    const int64_t pos = a.pos0 + t, R = a.K - 1 + BMAX;
    float acc = 0;
    for (int k = 0; k < a.K - 1; k++) {
        const int64_t p = pos - (a.K - 1 - k);
        const float x = p >= a.pos0 ? a.qkv[(p - a.pos0) * a.ch + c] : p >= 0 ? a.ring[(p % R) * a.ch + c] : 0.0f;
        acc += a.w[c * a.K + k] * x;
    }
    acc += a.w[c * a.K + a.K - 1] * a.qkv[t * a.ch + c];
    return silu_f(acc);
}

// Causal conv, and for the q and k heads (the first `heads` x DS channels) the L2 norm right after it, in
// registers: the unfused pair wrote the conv values and read them back (a full pass over q and k, 98 % of the memory
// rate, research/measurements/2026-11-03-dn-fuse.md). One wave per (token, q/k head) with the lane mapping of the
// separate norm (channels lane + 32 k), so the sum of squares has the same order; one wave per (token, 32 channels)
// for v.
__global__ void k_dn_conv_l2(DnConvArgs a) {
    const int64_t u = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    const int vw = (a.ch - a.heads * DS) / WAVE, per = a.heads + vw;
    if (u >= (int64_t)(a.m < 0 ? a.n : a.m) * per) return;
    const int64_t t = a.t0 + u / per;
    const int r = u % per;
    if (r >= a.heads) {
        const int64_t c = (int64_t)a.heads * DS + (r - a.heads) * WAVE + lane;
        a.out[t * a.ch + c] = dn_conv_at(a, t, c);
        return;
    }
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] = dn_conv_at(a, t, (int64_t)r * DS + lane + 32 * k), ss += v[k] * v[k];
    const float scale = rsqrtf(wave_sum(ss) / DS + a.eps / DS) * rsqrtf((float)DS);
#pragma unroll
    for (int k = 0; k < 4; k++) a.out[t * a.ch + (int64_t)r * DS + lane + 32 * k] = v[k] * scale;
}

// Position of a thread in the state: V head j, K head kh, v column vi, wave w (rows 16w ... 16w + 15)
struct DnPos {
    int j, kh, vi, w;
};

// Commits the m tokens still open from the last decode pass, token by token as in the megakernel (op_dn_recur): the
// same state whoever commits
__device__ __forceinline__ void dn_commit_open(const DnRecurArgs a, const DnPos p, float (&S)[16]) {
    const int vd = a.nv * DS;
    const float* ok = a.pk + (int64_t)a.half * BMAX * a.kd;
    const float* od = a.pd + (int64_t)a.half * BMAX * vd;
    const float* oa = a.pa + (int64_t)a.half * BMAX * a.nv;
    for (int i = 0; i < a.m; i++) {
        const float al = oa[i * a.nv + p.j], d = od[(int64_t)i * vd + p.j * DS + p.vi];
#pragma unroll
        for (int r = 0; r < 16; r++) S[r] = fmaf(ok[(int64_t)i * a.kd + p.kh * DS + 16 * p.w + r], d, al * S[r]);
    }
}

// Delta rule over the tokens of the chunk, one after the other. Block per (V head j, 32 v columns), lane = column,
// wave w holds the k rows 16w ... 16w + 15 of the state (S[ki][vi] as in the megakernel). First commits the m
// tokens still open from the last decode pass (half `half` of the intermediate buffers, S = alpha S + k delta^T per
// token as in the megakernel), then per token delta = beta (v - alpha S^T k), S = alpha S + k delta^T,
// o = S^T q / sqrt(128).
__global__ void __launch_bounds__(TPB) k_dn_recur(DnRecurArgs a) {
    const auto& [conv, ba, dt_bias, A, state, pk, pd, pa, m, half, raw, n, nk, nv, kd, ch] = a;
    const int j = blockIdx.x, lane = threadIdx.x % WAVE, w = threadIdx.x / WAVE;
    const int vi = blockIdx.y * 32 + lane, kh = j % nk, vd = nv * DS;
    float* st = state + (int64_t)j * DS * DS + vi;
    float S[16];
#pragma unroll
    for (int r = 0; r < 16; r++) S[r] = st[(16 * w + r) * DS];
    if (m > 0) dn_commit_open(a, {j, kh, vi, w}, S);
    // Load the values of the next token while the running one computes; k and q alternate between two LDS buffers.
    // Two syncs per token instead of three. 2.9 us per token before (load latency in the chain).
    __shared__ float kq[2][2][DS];  // [buffer][k | q][DS]
    __shared__ float red[2][TPB / WAVE][32];
    const float dtb = dt_bias[j], Aj = A[j];
    const int off_kq = threadIdx.x < DS ? kd + kh * DS + threadIdx.x : kh * DS + threadIdx.x - DS;
    auto fetch = [&](int t, float& kqv, float& bl, float& al, float& v) {
        const float* c = conv + (int64_t)t * ch;
        kqv = c[off_kq];
        bl = ba[(int64_t)t * 2 * nv + j];
        al = ba[(int64_t)t * 2 * nv + nv + j];
        v = c[2 * kd + j * DS + vi];
    };
    float kqv = 0, bl = 0, al = 0, v = 0;
    if (n > 0) fetch(0, kqv, bl, al, v);
    (&kq[0][0][0])[threadIdx.x] = kqv;
    __syncthreads();
    for (int t = 0; t < n; t++) {
        const int b = t & 1;
        const float beta = sigm(bl);
        const float xa = al + dtb;
        const float alpha = expf((xa > 20.0f ? xa : log1pf(expf(xa))) * Aj);
        const float vt = v;
        if (t + 1 < n) fetch(t + 1, kqv, bl, al, v);
        float part = 0;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            S[r] *= alpha;
            part += S[r] * kq[b][0][16 * w + r];
        }
        red[0][w][lane] = part;
        __syncthreads();
        float sk = 0;
#pragma unroll
        for (int i = 0; i < TPB / WAVE; i++) sk += red[0][i][lane];
        const float dl = beta * (vt - sk);
        float po = 0;
#pragma unroll
        for (int r = 0; r < 16; r++) {
            S[r] += kq[b][0][16 * w + r] * dl;
            po += S[r] * kq[b][1][16 * w + r];
        }
        red[1][w][lane] = po;
        if (t + 1 < n) (&kq[b ^ 1][0][0])[threadIdx.x] = kqv;
        __syncthreads();
        if (w == 0) {
            float o = 0;
#pragma unroll
            for (int i = 0; i < TPB / WAVE; i++) o += red[1][i][lane];
            raw[(int64_t)t * vd + j * DS + vi] = o * rsqrtf((float)DS);
        }
    }
#pragma unroll
    for (int r = 0; r < 16; r++) st[(16 * w + r) * DS] = S[r];
}

// Spread the sums of 16 values over the 32 lanes (reduce-scatter): afterwards lane l holds the full sum of value
// (l >> 1) & 15 (in v[0]), lanes 2c and 2c + 1 equal
template <int M, int H>
__device__ __forceinline__ void scatter_step(float (&v)[16], int lane) {
    const bool up = lane & M;
#pragma unroll
    for (int i = 0; i < H; i++) {
        const float send = up ? v[i] : v[i + H], keep = up ? v[i + H] : v[i];
        v[i] = keep + __shfl_xor(send, M);
    }
}
__device__ __forceinline__ float wave_scatter16(float (&v)[16], int lane) {
    scatter_step<16, 8>(v, lane);
    scatter_step<8, 4>(v, lane);
    scatter_step<4, 2>(v, lane);
    scatter_step<2, 1>(v, lane);
    return v[0] + __shfl_xor(v[0], 1);
}

// Delta rule token by token without a block-wide sync: one wave per (V head j, 16 v columns), lane l holds the
// k rows 4l ... 4l + 3 of these columns (64 values), sums over k by lane exchange, delta to all lanes by readlane.
// Modelled on llama.cpp PR 20340, where such a fused kernel was faster than the chunked form for Qwen3.5.
// k_dn_recur (n = 0) commits the open tokens of the last decode pass beforehand.
__global__ void __launch_bounds__(64) k_dn_wave(DnWaveArgs a) {
    const auto& [conv, ba, dt_bias, A, state, raw, n, nk, nv, kd, ch] = a;
    constexpr int NC = 16;
    const int lane = threadIdx.x & 31, gw = blockIdx.x * 2 + threadIdx.x / 32, slices = DS / NC;
    if (gw >= nv * slices) return;
    const int j = gw / slices, c0 = (gw % slices) * NC, kh = j % nk, vd = nv * DS, my = (lane >> 1) & 15;
    float* st = state + (int64_t)j * DS * DS + c0;
    float S[4][NC];
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < NC; c++) S[r][c] = st[(4 * lane + r) * DS + c];
    const float dtb = dt_bias[j], Aj = A[j];
    float4 kn = {}, qn = {};
    float bn = 0, an = 0, vn = 0;
    auto fetch = [&](int t) {
        const float* p = conv + (int64_t)t * ch;
        kn = *reinterpret_cast<const float4*>(p + kd + kh * DS + 4 * lane);
        qn = *reinterpret_cast<const float4*>(p + kh * DS + 4 * lane);
        vn = p[2 * kd + j * DS + c0 + my];
        bn = ba[(int64_t)t * 2 * nv + j];
        an = ba[(int64_t)t * 2 * nv + nv + j];
    };
    if (n > 0) fetch(0);
    for (int t = 0; t < n; t++) {
        const float k[4] = {kn.x, kn.y, kn.z, kn.w}, q[4] = {qn.x, qn.y, qn.z, qn.w};
        const float beta = sigm(bn), xa = an + dtb, v = vn;
        const float alpha = expf((xa > 20.0f ? xa : log1pf(expf(xa))) * Aj);
        if (t + 1 < n) fetch(t + 1);
        float p[NC];
#pragma unroll
        for (int c = 0; c < NC; c++) {
            float s = 0;
#pragma unroll
            for (int r = 0; r < 4; r++) S[r][c] *= alpha, s += S[r][c] * k[r];
            p[c] = s;
        }
        const float dl = beta * (v - wave_scatter16(p, lane));
#pragma unroll
        for (int c = 0; c < NC; c++) {
            const float dc = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, dl), 2 * c));
            float s = 0;
#pragma unroll
            for (int r = 0; r < 4; r++) S[r][c] += k[r] * dc, s += S[r][c] * q[r];
            p[c] = s;
        }
        const float o = wave_scatter16(p, lane);
        if (!(lane & 1)) raw[(int64_t)t * vd + j * DS + c0 + my] = o * rsqrtf((float)DS);
    }
#pragma unroll
    for (int r = 0; r < 4; r++)
#pragma unroll
        for (int c = 0; c < NC; c++) st[(4 * lane + r) * DS + c] = S[r][c];
}

__global__ void k_dn_norm(DnNormArgs a) {
    const auto& [raw, wn, z, o, n, nv, eps] = a;
    const int64_t u = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (u >= (int64_t)n * nv) return;
    const int64_t base = u * DS;  // token u / nv, head u % nv: nv * DS values per token, contiguous
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] = raw[base + lane + 32 * k], ss += v[k] * v[k];
    const float scale = rsqrtf(wave_sum(ss) / DS + eps);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = lane + 32 * k;
        o[base + i] = (_Float16)(v[k] * scale * wn[i] * sigm(z[base + i]));
    }
}

// Ornith's readout gate is silu(z). A copy of k_dn_norm instead of a template parameter: the Coder's kernel stays the
// text it was, and with it its machine code (isa-diff showed a changed instruction order with a template).
__global__ void k_dn_norm_silu(DnNormArgs a) {
    const auto& [raw, wn, z, o, n, nv, eps] = a;
    const int64_t u = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (u >= (int64_t)n * nv) return;
    const int64_t base = u * DS;
    float v[4], ss = 0;
#pragma unroll
    for (int k = 0; k < 4; k++) v[k] = raw[base + lane + 32 * k], ss += v[k] * v[k];
    const float scale = rsqrtf(wave_sum(ss) / DS + eps);
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = lane + 32 * k;
        o[base + i] = (_Float16)(v[k] * scale * wn[i] * silu_f(z[base + i]));
    }
}
}  // namespace

void launch_dn_conv_l2(const DnConvArgs& a, hipStream_t st) {
    const int64_t per = a.heads + (a.ch - a.heads * DS) / WAVE;
    for_row_pieces(a.n, [&](int t0, int m) {
        DnConvArgs b = a;
        b.t0 = t0, b.m = m;
        k_dn_conv_l2<<<blocks_for((int64_t)m * per * WAVE), TPB, 0, st>>>(b);
    });
}
void launch_dn_recur(const DnRecurArgs& a, hipStream_t st) { k_dn_recur<<<dim3(a.nv, DS / 32), TPB, 0, st>>>(a); }
void launch_dn_wave(const DnWaveArgs& a, hipStream_t st) { k_dn_wave<<<a.nv * DS / 16 / 2, 64, 0, st>>>(a); }
void launch_dn_norm(const DnNormArgs& a, hipStream_t st) {
    k_dn_norm<<<blocks_for((int64_t)a.n * a.nv * WAVE), TPB, 0, st>>>(a);
}
void launch_dn_norm_silu(const DnNormArgs& a, hipStream_t st) {
    k_dn_norm_silu<<<blocks_for((int64_t)a.n * a.nv * WAVE), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
