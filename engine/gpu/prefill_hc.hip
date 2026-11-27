// Prefill kernels for the residual streams (hyper-connections) and the gated residual: broadcast, norm, mix, combine,
// conversions and the ring store.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"
#include "gpu/prefill_split.h"

namespace tona::gpu {

namespace {

__global__ void k_bcast(BcastArgs a) {
    const auto& [e, R, n, hc, d] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x, nd = (int64_t)hc * d;
    if (i >= n * nd) return;
    R[i] = e[i / nd * 2 * d + i % nd % d];
}

__global__ void __launch_bounds__(TPB) k_norm16(Norm16Args a) {
    const auto& [R, w, xn, sc, hc, d, eps] = a;
    __shared__ float red[TPB / WAVE][4], s[4];
    const int t = blockIdx.x, lane = threadIdx.x % WAVE, wv = threadIdx.x / WAVE;
    const int64_t nd = (int64_t)hc * d;
    const float* r = R + t * nd;
    for (int g = 0; g < hc; g++) {
        float ss = 0;
        for (int i = threadIdx.x; i < d; i += TPB) ss += r[g * d + i] * r[g * d + i];
        ss = wave_sum(ss);
        if (lane == 0) red[wv][g] = ss;
    }
    __syncthreads();
    if ((int)threadIdx.x < hc) {
        float tt = 0;
        for (int i = 0; i < TPB / WAVE; i++) tt += red[i][threadIdx.x];
        s[threadIdx.x] = rsqrtf(tt / d + eps);
        sc[t * 4 + threadIdx.x] = s[threadIdx.x];
    }
    __syncthreads();
    for (int64_t j = threadIdx.x; j < nd; j += TPB) xn[t * nd + j] = (_Float16)(r[j] * s[j / d] * w[j]);
}

__global__ void __launch_bounds__(TPB) k_mtp_in16(MtpIn16Args a) {
    const auto& [e, R, hfirst, wn_e, wn_h, x16, t0, hc, d, eps] = a;
    __shared__ float red[TPB / WAVE];
    const int k = blockIdx.x, i = k / hc, g = k % hc, t = t0 + i;
    const int64_t hcd = (int64_t)hc * d;
    const float* eb = e + (int64_t)i * d;
    const float* hb = t == 0 ? (hfirst ? hfirst + (int64_t)g * d : nullptr) : R + (t - 1) * hcd + (int64_t)g * d;
    float se = 0, sh = 0;
    for (int j = threadIdx.x; j < d; j += TPB) se += eb[j] * eb[j], sh += hb ? hb[j] * hb[j] : 0.0f;
    const float fe = rsqrtf(block_sum256(se, red) / d + eps), fh = rsqrtf(block_sum256(sh, red) / d + eps);
    _Float16* o = x16 + (int64_t)k * 2 * d;
    for (int j = threadIdx.x; j < d; j += TPB) {
        o[j] = (_Float16)(eb[j] * fe * wn_e[j]);
        o[d + j] = (_Float16)(hb ? hb[j] * fh * wn_h[(int64_t)g * d + j] : 0.0f);
    }
}

// Saves reading R from VRAM for the norm: the R just written is still in the L2
__global__ void __launch_bounds__(TPB) k_combine_norm16(CombineNorm16Args a) {
    const auto& [R, y, inj, w, xn, sc, hc, d, eps] = a;
    __shared__ float red[TPB / WAVE][4], s[4];
    const int t = blockIdx.x, lane = threadIdx.x % WAVE, wv = threadIdx.x / WAVE;
    const int64_t nd = (int64_t)hc * d;
    float* r = R + t * nd;
    const float* yt = y + (int64_t)t * d;
    for (int g = 0; g < hc; g++) {
        const float f = 2.0f * sigm(inj[t * 4 + g] / hc);
        float ss = 0;
        for (int i = threadIdx.x; i < d; i += TPB) {
            const float v = r[g * d + i] + yt[i] * f;
            r[g * d + i] = v;
            ss += v * v;
        }
        ss = wave_sum(ss);
        if (lane == 0) red[wv][g] = ss;
    }
    __syncthreads();
    if ((int)threadIdx.x < hc) {
        float tt = 0;
        for (int i = 0; i < TPB / WAVE; i++) tt += red[i][threadIdx.x];
        s[threadIdx.x] = rsqrtf(tt / d + eps);
        sc[t * 4 + threadIdx.x] = s[threadIdx.x];
    }
    __syncthreads();
    for (int64_t j = threadIdx.x; j < nd; j += TPB) xn[t * nd + j] = (_Float16)(r[j] * s[j / d] * w[j]);
}

__global__ void k_act16(Act16Args a) {
    const auto& [x, y, n, scale] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i < n) y[i] = (_Float16)silu_f(x[i] * scale);
}

__global__ void k_mix(MixArgs a) {
    const auto& [R, sc, w, gate, mixed, mixed16, n, hc, d] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * d) return;
    const int64_t t = i / d, k = i % d, nd = (int64_t)hc * d;
    float s = 0;
    for (int g = 0; g < hc; g++) {
        const int64_t j = t * nd + g * d + k;
        s += R[j] * sc[t * 4 + g] * w[g * d + k] * sigm(gate[j]);
    }
    const float m = s / hc;
    mixed[i] = m;
    mixed16[i] = (_Float16)m;
}

__global__ void k_combine(CombineArgs a) {
    const auto& [R, y, inj, n, hc, d] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x, nd = (int64_t)hc * d;
    if (i >= n * nd) return;
    const int64_t t = i / nd, j = i % nd;
    R[i] += y[t * d + j % d] * 2.0f * sigm(inj[t * 4 + j / d] / hc);
}

__global__ void k_to16(To16Args a) {
    const auto& [x, ldx, y, ldy, rows, cols] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)rows * cols) return;
    y[i / cols * ldy + i % cols] = (_Float16)x[i / cols * ldx + i % cols];
}

__global__ void k_hc_mean(HcMeanArgs a) {
    const auto& [R, F, n, hc, d, ldf, col0] = a;
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)n * d) return;
    const int64_t t = i / d, k = i % d;
    float s = 0;
    for (int g = 0; g < hc; g++) s += R[t * hc * d + g * d + k];
    F[t * ldf + col0 + k] = s * (1.0f / hc);
}

__global__ void k_ring_store(RingStoreArgs a) {
    const auto& [src, ring, n, pos0, cols, nring] = a;
    const int keep = min(n, nring);
    const int64_t i = (int64_t)blockIdx.x * TPB + threadIdx.x;
    if (i >= (int64_t)keep * cols) return;
    const int64_t t = n - keep + i / cols, c = i % cols;
    ring[((pos0 + t) % nring) * cols + c] = src[t * cols + c];
}

}  // namespace

void launch_bcast(const BcastArgs& a, hipStream_t st) {
    k_bcast<<<blocks_for((int64_t)a.n * a.hc * a.d), TPB, 0, st>>>(a);
}
void launch_norm16(const Norm16Args& a, int n_tokens, hipStream_t st) {
    const int64_t nd = (int64_t)a.hc * a.d;
    for_row_pieces(n_tokens, [&](int t0, int m) {
        Norm16Args b = a;
        b.R += t0 * nd, b.xn += t0 * nd, b.sc += t0 * 4;
        k_norm16<<<m, TPB, 0, st>>>(b);
    });
}
void launch_mtp_in16(const MtpIn16Args& a, int blocks, hipStream_t st) { k_mtp_in16<<<blocks, TPB, 0, st>>>(a); }
void launch_combine_norm16(const CombineNorm16Args& a, int n_tokens, hipStream_t st) {
    const int64_t nd = (int64_t)a.hc * a.d;
    for_row_pieces(n_tokens, [&](int t0, int m) {
        CombineNorm16Args b = a;
        b.R += t0 * nd, b.y += (int64_t)t0 * a.d, b.inj += t0 * 4, b.xn += t0 * nd, b.sc += t0 * 4;
        k_combine_norm16<<<m, TPB, 0, st>>>(b);
    });
}
void launch_act16(const Act16Args& a, hipStream_t st) { k_act16<<<blocks_for(a.n), TPB, 0, st>>>(a); }
void launch_mix(const MixArgs& a, hipStream_t st) { k_mix<<<blocks_for((int64_t)a.n * a.d), TPB, 0, st>>>(a); }
void launch_combine(const CombineArgs& a, hipStream_t st) {
    const int64_t nd = (int64_t)a.hc * a.d;
    for_row_pieces(a.n, [&](int t0, int m) {
        CombineArgs b = a;
        b.R += t0 * nd, b.y += (int64_t)t0 * a.d, b.inj += t0 * 4, b.n = m;
        k_combine<<<blocks_for((int64_t)m * nd), TPB, 0, st>>>(b);
    });
}
void launch_to16(const To16Args& a, hipStream_t st) { k_to16<<<blocks_for((int64_t)a.rows * a.cols), TPB, 0, st>>>(a); }
void launch_hc_mean(const HcMeanArgs& a, hipStream_t st) {
    k_hc_mean<<<blocks_for((int64_t)a.n * a.d), TPB, 0, st>>>(a);
}
void launch_ring_store(const RingStoreArgs& a, hipStream_t st) {
    k_ring_store<<<blocks_for((int64_t)std::min(a.n, a.nring) * a.cols), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
