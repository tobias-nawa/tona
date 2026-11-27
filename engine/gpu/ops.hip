#include <cmath>

#include "gpu/ops.h"
#include "gpu/yarn_dev.h"

namespace tona::gpu::ops {

namespace {

constexpr int WAVE = 32;

__device__ __forceinline__ float sigm(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + __expf(-x)); }

// Sum over a block (blockDim.x a multiple of 32, at most 1024)
__device__ float block_sum(float v) {
    __shared__ float part[32];
    for (int off = WAVE / 2; off > 0; off /= 2) v += __shfl_xor(v, off, WAVE);
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    __syncthreads();
    if (lane == 0) part[w] = v;
    __syncthreads();
    float t = 0;
    for (int i = 0; i < (int)(blockDim.x / WAVE); i++) t += part[i];
    return t;
}
__device__ float block_max(float v) {
    __shared__ float part[32];
    for (int off = WAVE / 2; off > 0; off /= 2) v = fmaxf(v, __shfl_xor(v, off, WAVE));
    const int w = threadIdx.x / WAVE, lane = threadIdx.x % WAVE;
    __syncthreads();
    if (lane == 0) part[w] = v;
    __syncthreads();
    float t = -INFINITY;
    for (int i = 0; i < (int)(blockDim.x / WAVE); i++) t = fmaxf(t, part[i]);
    return t;
}

__device__ void rope_dev(float* x, int n_rot, int64_t pos, float base) {
    // one thread per pair; the caller synchronizes
    const int half = n_rot / 2;
    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        float c, s;
        rope_cs(i, n_rot, pos, base, c, s);
        const float x0 = x[i], x1 = x[i + half];
        x[i] = x0 * c - x1 * s;
        x[i + half] = x0 * s + x1 * c;
    }
}

__global__ void k_rms_norm_groups(const float* x, const float* w, float* out, int d, float eps) {
    const float* xg = x + (int64_t)blockIdx.x * d;
    float ss = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss += xg[i] * xg[i];
    const float scale = rsqrtf(block_sum(ss) / d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const int64_t j = (int64_t)blockIdx.x * d + i;
        out[j] = xg[i] * scale * (w ? w[j] : 1.0f);
    }
}

__global__ void k_silu_scale(float* v, int n, float scale) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) v[i] = silu_f(v[i] * scale);
}

__global__ void k_hc_mix(const float* xn, const float* gate, float* mixed, int hc, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= d) return;
    float s = 0;
    for (int c = 0; c < hc; c++) s += xn[c * d + i] * sigm(gate[c * d + i]);
    mixed[i] = s / hc;
}

__global__ void k_hc_combine(float* R, const float* y, const float* inject, int hc, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= hc * d) return;
    R[i] += y[i % d] * 2.0f * sigm(inject[i / d] / hc);
}

__global__ void k_broadcast(float* R, const float* e, int hc, int d) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < hc * d) R[i] = e[i % d];
}

__global__ void k_dn_conv(DnConvArgs a) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= a.ch) return;
    const int K = a.K, H = K - 1;
    float acc = 0;
    // Tap k (0..K-2) reads K-1-k positions back; the slot of position p is p mod H
    for (int k = 0; k < H; k++) {
        const int64_t p = a.pos - (H - k);
        acc += a.w[c * K + k] * (p >= 0 ? a.state[((p % H + H) % H) * a.ch + c] : 0.0f);
    }
    acc += a.w[c * K + H] * a.qkv[c];
    a.state[(a.pos % H) * a.ch + c] = a.qkv[c];
    a.out[c] = silu_f(acc);
}

__global__ void k_l2norm_heads(float* x, int dim, float eps) {
    float* h = x + (int64_t)blockIdx.x * dim;
    float ss = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += h[i] * h[i];
    const float scale = rsqrtf(block_sum(ss) / dim + eps / dim) * rsqrtf((float)dim);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) h[i] *= scale;
}

// Arguments of k_dn_recur: the wrapper's DnRecurArgs without nv (the grid size)
struct DnRecurKernelArgs {
    const float *q, *k, *v, *b, *a, *dt_bias, *A, *w_norm, *z;
    float *state, *o;
    int nk, S;
    float eps;
};

// One block per V head j, one thread per v row; state st[k*S + v] (contiguous over v: coalesced accesses)
__global__ void k_dn_recur(DnRecurKernelArgs p) {
    extern __shared__ float sh[];
    const int S = p.S;
    float* ks = sh;
    float* qs = sh + S;
    const int j = blockIdx.x, vi = threadIdx.x;
    const int kh = j % p.nk;  // GGUF: V head j uses K head j mod nk
    if (vi < S) {
        ks[vi] = p.k[kh * S + vi];
        qs[vi] = p.q[kh * S + vi];
    }
    __syncthreads();
    const float beta = sigm(p.b[j]);
    const float xa = p.a[j] + p.dt_bias[j];
    const float sp = xa > 20.0f ? xa : log1pf(expf(xa));
    const float alpha = expf(sp * p.A[j]);
    float* st = p.state + (int64_t)j * S * S;
    float out = 0;
    if (vi < S) {
        float sk = 0;
        for (int ki = 0; ki < S; ki++) {
            const float s0 = st[ki * S + vi] * alpha;
            st[ki * S + vi] = s0;
            sk += s0 * ks[ki];
        }
        const float dlt = (p.v[j * S + vi] - sk) * beta;
        for (int ki = 0; ki < S; ki++) {
            const float s1 = st[ki * S + vi] + ks[ki] * dlt;
            st[ki * S + vi] = s1;
            out += s1 * qs[ki];
        }
        out *= rsqrtf((float)S);
    }
    const float ss = block_sum(vi < S ? out * out : 0.0f);
    if (vi < S) {
        const float scale = rsqrtf(ss / S + p.eps);
        p.o[j * S + vi] = out * scale * p.w_norm[vi] * sigm(p.z[j * S + vi]);
    }
}

__global__ void k_qsa_split(const float* qfull, float* q, float* gate, int hd) {
    const int h = blockIdx.x;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        q[h * hd + i] = qfull[h * 2 * hd + i];
        gate[h * hd + i] = qfull[h * 2 * hd + hd + i];
    }
}

// Arguments of k_head_norm_rope: the wrapper's HeadNormRopeArgs without heads (the grid size)
struct HeadNormRopeKernelArgs {
    float* x;
    const float* w;
    int hd, n_rot;
    int64_t pos;
    float base, eps;
};

__global__ void k_head_norm_rope(HeadNormRopeKernelArgs a) {
    const int hd = a.hd;
    float* h = a.x + (int64_t)blockIdx.x * hd;
    float ss = 0;
    for (int i = threadIdx.x; i < hd; i += blockDim.x) ss += h[i] * h[i];
    const float scale = rsqrtf(block_sum(ss) / hd + a.eps);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) h[i] = h[i] * scale * a.w[i];
    __syncthreads();
    rope_dev(h, a.n_rot, a.pos, a.base);
}

__global__ void k_idx_pool(IdxPoolArgs a) {
    __shared__ float buf[1024];
    const int dim = a.dim, blk = a.blk;
    float ss = 0;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float m = 0;
        for (int r = 0; r < blk; r++) m += a.raw[(a.first + r) * dim + i];
        m *= 1.0f / blk;
        buf[i] = m;
        ss += m * m;
    }
    const float scale = rsqrtf(block_sum(ss) / dim + a.eps);
    for (int i = threadIdx.x; i < dim; i += blockDim.x) buf[i] = buf[i] * scale * a.w[i];
    __syncthreads();
    rope_dev(buf, a.n_rot, a.first, a.base);
    __syncthreads();
    for (int i = threadIdx.x; i < dim; i += blockDim.x) a.out[i] = buf[i];
}

// One wave per block b
__global__ void k_idx_score(const float* q, const float* pool, float* score, int heads, int dim, int64_t n_pool) {
    const int64_t b = (int64_t)blockIdx.x * (blockDim.x / WAVE) + threadIdx.x / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (b >= n_pool) return;
    float total = 0;
    for (int h = 0; h < heads; h++) {
        float dot = 0;
        for (int i = lane; i < dim; i += WAVE) dot += q[h * dim + i] * pool[b * dim + i];
        for (int off = WAVE / 2; off > 0; off /= 2) dot += __shfl_xor(dot, off, WAVE);
        total += fmaxf(dot, 0.0f);
    }
    if (lane == 0) score[b] = total * rsqrtf((float)dim);
}

// One block per Q head; at most 4096 cells (QSA selects at most 2051)
__global__ void k_attention(AttentionArgs a) {
    __shared__ float sc[4096];
    __shared__ float qh[512];
    const int hd = a.hd, nkv = a.nkv, n_cells = a.n_cells;
    const int h = blockIdx.x, kvh = h / (a.nh / nkv);
    for (int i = threadIdx.x; i < hd; i += blockDim.x) qh[i] = a.q[h * hd + i];
    __syncthreads();
    // One wave per cell for the dot product
    const int wv = threadIdx.x / WAVE, lane = threadIdx.x % WAVE, nw = blockDim.x / WAVE;
    for (int c = wv; c < n_cells; c += nw) {
        const int64_t cell = a.cells ? a.cells[c] : c;
        const float* kc = a.K + (cell * nkv + kvh) * hd;
        float dot = 0;
        for (int i = lane; i < hd; i += WAVE) dot += qh[i] * kc[i];
        for (int off = WAVE / 2; off > 0; off /= 2) dot += __shfl_xor(dot, off, WAVE);
        if (lane == 0) sc[c] = dot * a.scale;
    }
    __syncthreads();
    float mx = -INFINITY;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) mx = fmaxf(mx, sc[c]);
    mx = block_max(mx);
    float sum = 0;
    for (int c = threadIdx.x; c < n_cells; c += blockDim.x) {
        const float e = __expf(sc[c] - mx);
        sc[c] = e;
        sum += e;
    }
    sum = block_sum(sum);
    __syncthreads();
    for (int i = threadIdx.x; i < hd; i += blockDim.x) {
        float acc = 0;
        for (int c = 0; c < n_cells; c++) {
            const int64_t cell = a.cells ? a.cells[c] : c;
            acc += sc[c] * a.V[(cell * nkv + kvh) * hd + i];
        }
        a.o[h * hd + i] = acc / sum;
    }
}

__global__ void k_mul_sigmoid(float* x, const float* g, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= sigm(g[i]);
}
__global__ void k_silu_mul(float* g, const float* u, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) g[i] = silu_f(g[i]) * u[i];
}
__global__ void k_axpy(float* y, const float* x, float w, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += w * x[i];
}
__global__ void k_axpy_sigmoid(float* y, const float* x, const float* gate, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += sigm(*gate) * x[i];
}

struct PleGateKernelArgs {
    const float *R, *key, *val, *w_key, *w_query, *w_conv;
    float *gated, *normed;
    int d;
    float eps;
};

// One block per stream c: normalize key/query, gate, gated = val * gate_c, then normalize with w_conv
__global__ void k_ple_gate(PleGateKernelArgs a) {
    const int c = blockIdx.x, d = a.d;
    const float* kc = a.key + (int64_t)c * d;
    const float* rc = a.R + (int64_t)c * d;
    float sk = 0, sq = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) sk += kc[i] * kc[i], sq += rc[i] * rc[i];
    const float fk = rsqrtf(block_sum(sk) / d + a.eps), fq = rsqrtf(block_sum(sq) / d + a.eps);
    float dot = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        dot += (kc[i] * fk * a.w_key[c * d + i]) * (rc[i] * fq * a.w_query[c * d + i]);
    const float s = block_sum(dot) * rsqrtf((float)d);
    const float mag = sqrtf(fmaxf(fabsf(s), 1e-6f));
    const float sgn = (float)((s > 0) - (s < 0));
    const float gate = sigm(sgn * mag);
    float sg = 0;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const float g = a.val[i] * gate;
        a.gated[c * d + i] = g;
        sg += g * g;
    }
    const float fg = rsqrtf(block_sum(sg) / d + a.eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x)
        a.normed[c * d + i] = a.gated[c * d + i] * fg * a.w_conv[c * d + i];
}

struct PleConvKernelArgs {
    float* R;
    const float *gated, *normed, *w;
    float* hist;
    int n, K, dil;
    int64_t pos;
};

// One channel per thread: taps at p, p-dil, ..., p-(K-1)*dil; ring buffer hist with (K-1)*dil slots, slot p mod H
__global__ void k_ple_conv(PleConvKernelArgs a) {
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= a.n) return;
    const int K = a.K, H = (K - 1) * a.dil;
    float acc = 0;
    for (int k = 0; k < K; k++) {
        const int back = (K - 1 - k) * a.dil;
        float x;
        if (back == 0)
            x = a.normed[ch];
        else {
            const int64_t p = a.pos - back;
            x = p >= 0 ? a.hist[(p % H) * (int64_t)a.n + ch] : 0.0f;
        }
        acc += a.w[ch * K + k] * x;
    }
    a.hist[(a.pos % H) * (int64_t)a.n + ch] = a.normed[ch];  // slot of pos - H was read above
    a.R[ch] += a.gated[ch] + silu_f(acc);
}

int blocks(int n, int t = 256) { return (n + t - 1) / t; }

}  // namespace

void rms_norm_groups(const RmsNormArgs& a, hipStream_t s) {
    hipLaunchKernelGGL(k_rms_norm_groups, dim3(a.groups), dim3(256), 0, s, a.x, a.w, a.out, a.d, a.eps);
}
void silu_scale(float* v, int n, float scale, hipStream_t s) {
    hipLaunchKernelGGL(k_silu_scale, dim3(blocks(n)), dim3(256), 0, s, v, n, scale);
}
void hc_mix(const float* xn, const float* gate, float* mixed, int hc, int d, hipStream_t s) {
    hipLaunchKernelGGL(k_hc_mix, dim3(blocks(d)), dim3(256), 0, s, xn, gate, mixed, hc, d);
}
void hc_combine(float* R, const float* y, const float* inject, int hc, int d, hipStream_t s) {
    hipLaunchKernelGGL(k_hc_combine, dim3(blocks(hc * d)), dim3(256), 0, s, R, y, inject, hc, d);
}
void broadcast_rows(float* R, const float* e, int hc, int d, hipStream_t s) {
    hipLaunchKernelGGL(k_broadcast, dim3(blocks(hc * d)), dim3(256), 0, s, R, e, hc, d);
}
void dn_conv(const DnConvArgs& a, hipStream_t s) {
    hipLaunchKernelGGL(k_dn_conv, dim3(blocks(a.ch)), dim3(256), 0, s, a);
}
void l2norm_heads(float* x, int heads, int dim, float eps, hipStream_t s) {
    hipLaunchKernelGGL(k_l2norm_heads, dim3(heads), dim3(128), 0, s, x, dim, eps);
}
void dn_recur(const DnRecurArgs& a, hipStream_t s) {
    const int threads = (a.S + WAVE - 1) / WAVE * WAVE;
    const DnRecurKernelArgs p{a.q, a.k, a.v, a.b, a.a, a.dt_bias, a.A, a.w_norm, a.z, a.state, a.o, a.nk, a.S, a.eps};
    hipLaunchKernelGGL(k_dn_recur, dim3(a.nv), dim3(threads), 2 * a.S * sizeof(float), s, p);
}
void qsa_split(const float* qfull, float* q, float* gate, int heads, int hd, hipStream_t s) {
    hipLaunchKernelGGL(k_qsa_split, dim3(heads), dim3(256), 0, s, qfull, q, gate, hd);
}
void set_yarn(const rope::YarnTable& t) { upload_yarn(t); }
void check_yarn(const rope::YarnTable& t) { gpu::check_yarn(t, "ops.hip"); }
void head_norm_rope(const HeadNormRopeArgs& a, hipStream_t s) {
    const HeadNormRopeKernelArgs p{a.x, a.w, a.hd, a.n_rot, a.pos, a.base, a.eps};
    hipLaunchKernelGGL(k_head_norm_rope, dim3(a.heads), dim3(256), 0, s, p);
}
void idx_pool(const IdxPoolArgs& a, hipStream_t s) { hipLaunchKernelGGL(k_idx_pool, dim3(1), dim3(128), 0, s, a); }
void idx_score(const IdxScoreArgs& a, hipStream_t s) {
    hipLaunchKernelGGL(k_idx_score, dim3((unsigned)((a.n_pool + 7) / 8)), dim3(256), 0, s, a.q, a.pool, a.score,
                       a.heads, a.dim, a.n_pool);
}
void attention(const AttentionArgs& a, hipStream_t s) {
    hipLaunchKernelGGL(k_attention, dim3(a.nh), dim3(256), 0, s, a);
}
void mul_sigmoid(float* x, const float* g, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_mul_sigmoid, dim3(blocks(n)), dim3(256), 0, s, x, g, n);
}
void silu_mul(float* g, const float* u, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_silu_mul, dim3(blocks(n)), dim3(256), 0, s, g, u, n);
}
void axpy(float* y, const float* x, float w, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_axpy, dim3(blocks(n)), dim3(256), 0, s, y, x, w, n);
}
void axpy_sigmoid(float* y, const float* x, const float* gate, int n, hipStream_t s) {
    hipLaunchKernelGGL(k_axpy_sigmoid, dim3(blocks(n)), dim3(256), 0, s, y, x, gate, n);
}
void ple_apply(const PleArgs& a, hipStream_t s) {
    float* gated = a.scratch;
    float* normed = a.scratch + (int64_t)a.hc * a.d;
    const PleGateKernelArgs gate{a.R, a.key, a.val, a.w_key, a.w_query, a.w_conv_norm, gated, normed, a.d, a.eps};
    hipLaunchKernelGGL(k_ple_gate, dim3(a.hc), dim3(256), 0, s, gate);
    const PleConvKernelArgs conv{a.R, gated, normed, a.conv_w, a.hist, a.hc * a.d, a.K, a.dil, a.pos};
    hipLaunchKernelGGL(k_ple_conv, dim3(blocks(a.hc * a.d)), dim3(256), 0, s, conv);
}

}  // namespace tona::gpu::ops
