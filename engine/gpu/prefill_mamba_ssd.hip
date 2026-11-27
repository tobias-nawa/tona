// Chunked state-space duality (SSD, Dao and Gu, arXiv 2405.21060) form of the Mamba-2 prefill scan on the matrix units
// (prefill_mamba.h, launch_mamba_ssd). The prompt is cut into chunks of SQ = 64 tokens on a grid of absolute positions
// (a chunk never straddles a multiple of 64, so a prompt cut at a multiple of 64 gives the same bits as the whole).
// Per head, over the chunks one after the other, with S^T (N x P) in the accumulators of 4 waves (16 rows P each):
//   y_t = sum_{s <= t} (C_t . B_s) exp(cs_t - cs_s) dt_s x_s      intra-chunk:  M (t x s) times Xdt (s x p)
//         + exp(cs_t) C_t . S_start                               inter-chunk:  C (t x n) times S^T (n x p)
//         + D x_t
//   S_end = exp(cs_Q) S_start + sum_s exp(cs_Q - cs_s) dt_s x_s (x) B_s          Bt (n x s) times Xw (s x p)
// cs is the running sum of dt * A over the chunk. Operands of the matrix products are f16, the accumulators f32 and the
// state stays f32 between chunks and calls; this reorders the sums of the sequential scan (prefill_mamba.hip), so the
// results differ from it by rounding, and the state a decode pass continues from is not the bit-equal one
// (research/measurements/2026-11-19-bolt-opt1.md has the tolerance).
// Two kernels. k_ssd_prep, per (chunk, group): the scores C . B^T (the same for the 16 heads of a group), and B^T and C
// as f16 in the layouts the matrix units read them (the scan kernel loads them straight from L2 as operands).
// k_ssd_scan, per head: x, the scores and the running sum of the next chunk are requested before the matrix products of
// this one (the loads of the first version were 30 us per chunk of latency in a row), and the LDS holds only what the 4
// waves of a head share, so 4 blocks run per WGP.
#include <format>
#include <stdexcept>

#include "gpu/mamba_dev.h"
#include "gpu/prefill_dev.h"
#include "gpu/prefill_mamba.h"

namespace tona::gpu {

namespace {

constexpr int SQ = 64;            // tokens per chunk
constexpr int SN = MB_N;          // state size
constexpr int SP = 64;            // rows per head (the kernels are written for this)
constexpr int SWAVES = SP / 16;   // waves per block, one 16-row slice of P each
constexpr int SNT = SWAVES * 32;  // threads per block
constexpr int LDB = SQ + 8;       // LDS row strides in halves (+8: against bank conflicts)
constexpr int LDP = SN + 8;

using half4 = _Float16 __attribute__((ext_vector_type(4)));

// The chunk grid: chunk 0 ends at the next multiple of SQ of the absolute position
struct Chunk {
    int first, len;
};
__device__ __forceinline__ Chunk chunk_of(int c, int64_t pos0, int n) {
    const int head = SQ - (int)(pos0 % SQ);
    const int first = c == 0 ? 0 : head + (c - 1) * SQ;
    return {first, min(c == 0 ? head : SQ, n - first)};
}

// What k_ssd_prep leaves per (chunk, group): the scores (SQ x SQ floats), C (SQ x SN halves, row t) and B^T (SN x SQ
// halves, row n)
struct PrepPtrs {
    float* cb;
    _Float16 *c16, *bt16;
};
__device__ __forceinline__ PrepPtrs prep_ptrs(const SsdArgs& a, int c, int g) {
    const int64_t i = (int64_t)c * a.groups + g;
    return {a.cb + i * SQ * SQ, a.c16 + i * SQ * SN, a.bt16 + i * SN * SQ};
}

// B and C of the chunk (all threads) into the LDS as f16, rows s and t (stride LDP), zero past the chunk
__device__ void prep_stage(const SsdArgs& a, const Chunk ck, int g, _Float16* Bs, _Float16* Cs) {
    const float* Bb = a.conv + a.inner + (int64_t)g * SN;
    const float* Cb = a.conv + a.inner + (int64_t)a.groups * SN + (int64_t)g * SN;
    constexpr int ITER = SQ * SN / 4 / SNT;
#pragma unroll
    for (int b0 = 0; b0 < ITER; b0 += 4) {
        float4 vb[4], vc[4];
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = threadIdx.x + (b0 + k) * SNT, s = i / (SN / 4), n4 = i % (SN / 4);
            vb[k] = vc[k] = float4{};
            if (s < ck.len) {
                vb[k] = *reinterpret_cast<const float4*>(Bb + (int64_t)(ck.first + s) * a.conv_dim + 4 * n4);
                vc[k] = *reinterpret_cast<const float4*>(Cb + (int64_t)(ck.first + s) * a.conv_dim + 4 * n4);
            }
        }
#pragma unroll
        for (int k = 0; k < 4; k++) {
            const int i = threadIdx.x + (b0 + k) * SNT, s = i / (SN / 4), n4 = i % (SN / 4);
            *reinterpret_cast<half4*>(&Bs[s * LDP + 4 * n4]) =
                half4{(_Float16)vb[k].x, (_Float16)vb[k].y, (_Float16)vb[k].z, (_Float16)vb[k].w};
            *reinterpret_cast<half4*>(&Cs[s * LDP + 4 * n4]) =
                half4{(_Float16)vc[k].x, (_Float16)vc[k].y, (_Float16)vc[k].z, (_Float16)vc[k].w};
        }
    }
}

// Scores of the 16 rows t = 16 w ... of the chunk: C (t x n) times B^T (n x s), operands from the LDS
__device__ void prep_scores(const _Float16* Bs, const _Float16* Cs, float* cb) {
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    float8 acc[SQ / 16] = {};
#pragma unroll
    for (int k = 0; k < SN / 16; k++) {
        const half8 av = *reinterpret_cast<const half8*>(&Cs[(16 * w + r16) * LDP + 16 * k + kh]);
#pragma unroll
        for (int j = 0; j < SQ / 16; j++) {
            const half8 bv = *reinterpret_cast<const half8*>(&Bs[(16 * j + r16) * LDP + 16 * k + kh]);
            acc[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(av, bv, acc[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < SQ / 16; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) cb[(16 * w + kh + e) * SQ + 16 * j + r16] = acc[j][e];
}

// One block per (chunk, group): the scores, C and B^T for the scan
__global__ void __launch_bounds__(SNT) k_ssd_prep(SsdArgs a) {
    __shared__ _Float16 Bs[SQ * LDP], Cs[SQ * LDP];
    const int c = blockIdx.x, g = blockIdx.y;
    const Chunk ck = chunk_of(c, a.pos0, a.n);
    const PrepPtrs out = prep_ptrs(a, c, g);
    prep_stage(a, ck, g, Bs, Cs);
    __syncthreads();
    prep_scores(Bs, Cs, out.cb);
    for (int i = threadIdx.x; i < SQ * SN / 8; i += SNT) {  // C as it is, 8 halves at a time
        const int t = i / (SN / 8), n8 = i % (SN / 8);
        *reinterpret_cast<half8*>(&out.c16[t * SN + 8 * n8]) = *reinterpret_cast<const half8*>(&Cs[t * LDP + 8 * n8]);
    }
    for (int i = threadIdx.x; i < SN * SQ / 8; i += SNT) {  // B^T: row n, 8 consecutive s
        const int n = i / (SQ / 8), s8 = i % (SQ / 8);
        half8 v;
        for (int j = 0; j < 8; j++) v[j] = Bs[(8 * s8 + j) * LDP + n];
        *reinterpret_cast<half8*>(&out.bt16[n * SQ + 8 * s8]) = v;
    }
}

// What one block keeps in the LDS for a chunk of a head: M = (C . B^T) exp(cs_t - cs_s) lower triangle (t x s), X dt
// and the raw X transposed (p x s), the running sum cs, dt and the weights of the state update
struct SsdLds {
    _Float16 Ms[SQ * LDB], xT[SP * LDB], xr[SP * LDB];
    float cs[SQ], dts[SQ], wgt[SQ];
};

// What the threads request for a chunk before they need it: x of the head (8 float4 per thread), the scores (8 float4)
// and, in wave 0, dt * A and dt of two tokens
struct Pending {
    float4 vx[SQ * SP / 4 / SNT], vcb[SQ * SQ / 4 / SNT];
    float da[2], dt[2];
};
__device__ void request(const SsdArgs& a, const Chunk ck, int c, int h, Pending& r) {
    const int g = h / (a.heads / a.groups);
    const float* cb = prep_ptrs(a, c, g).cb;
#pragma unroll
    for (int k = 0; k < SQ * SP / 4 / SNT; k++) {
        const int i = threadIdx.x + k * SNT, s = i / (SP / 4), p4 = i % (SP / 4);
        r.vx[k] = float4{};
        if (s < ck.len)
            r.vx[k] = *reinterpret_cast<const float4*>(a.conv + (int64_t)(ck.first + s) * a.conv_dim + h * SP + 4 * p4);
    }
#pragma unroll
    for (int k = 0; k < SQ * SQ / 4 / SNT; k++)
        r.vcb[k] = *reinterpret_cast<const float4*>(cb + 4 * (threadIdx.x + k * SNT));
    for (int i = 0; i < 2; i++) {
        const int s = threadIdx.x % 32 + 32 * i;
        r.da[i] = r.dt[i] = 0.0f;
        if (threadIdx.x < 32 && s < ck.len)
            r.da[i] = a.da[(int64_t)(ck.first + s) * a.heads + h],
            r.dt[i] = a.dt[(int64_t)(ck.first + s) * a.heads + h];
    }
}

// cs = inclusive sum of dt * A over the chunk (wave 0, from the requested values), and dt, into the LDS
__device__ void store_cs(const Pending& r, SsdLds& L) {
    if (threadIdx.x >= 32) return;
    const int lane = threadIdx.x;
    float d0 = r.da[0], d1 = r.da[1];
#pragma unroll
    for (int off = 1; off < 32; off *= 2) {
        const float u0 = __shfl_up(d0, off, 32), u1 = __shfl_up(d1, off, 32);
        if (lane >= off) d0 += u0, d1 += u1;
    }
    d1 += __shfl(d0, 31, 32);
    L.cs[lane] = d0, L.cs[lane + 32] = d1, L.dts[lane] = r.dt[0], L.dts[lane + 32] = r.dt[1];
}

// x, the masked scores and the weights of the state update into the LDS (cs and dt are there)
__device__ void store_chunk(const Pending& r, const Chunk ck, SsdLds& L) {
    if (threadIdx.x < SQ) L.wgt[threadIdx.x] = __expf(L.cs[ck.len - 1] - L.cs[threadIdx.x]);
#pragma unroll
    for (int k = 0; k < SQ * SP / 4 / SNT; k++) {
        const int i = threadIdx.x + k * SNT, s = i / (SP / 4), p4 = i % (SP / 4);
        const float xd = L.dts[s];
        const float v[4] = {r.vx[k].x, r.vx[k].y, r.vx[k].z, r.vx[k].w};
#pragma unroll
        for (int j = 0; j < 4; j++)
            L.xT[(4 * p4 + j) * LDB + s] = (_Float16)(v[j] * xd), L.xr[(4 * p4 + j) * LDB + s] = (_Float16)v[j];
    }
#pragma unroll
    for (int k = 0; k < SQ * SQ / 4 / SNT; k++) {
        const int i = 4 * (threadIdx.x + k * SNT), t = i / SQ, s0 = i % SQ;
        const float e[4] = {r.vcb[k].x, r.vcb[k].y, r.vcb[k].z, r.vcb[k].w};
        _Float16 m[4];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int s = s0 + j;
            m[j] = (s <= t && t < ck.len) ? (_Float16)(e[j] * __expf(L.cs[t] - L.cs[s])) : (_Float16)0.0f;
        }
        *reinterpret_cast<half4*>(&L.Ms[t * LDB + s0]) = half4{m[0], m[1], m[2], m[3]};
    }
}

// The state as f16 for the matrix product (saturating: a state beyond the f16 range would turn into inf)
__device__ __forceinline__ half8 to_half8(const float8& v) {
    half8 h;
#pragma unroll
    for (int e = 0; e < 8; e++) h[e] = (_Float16)fminf(fmaxf(v[e], -65504.0f), 65504.0f);
    return h;
}

// The two matrix products of the 16 rows t = 16 ti ...: yi intra-chunk (M times X dt), yo inter-chunk (C times the
// state)
__device__ void chunk_products(const SsdLds& L, const _Float16* c16, const half8 (&sf)[SN / 16], int ti, float8& yi,
                               float8& yo) {
    const int lane = threadIdx.x % 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int p = threadIdx.x / 32 * 16 + r16;
    for (int ks = 0; ks <= ti; ks++) {
        const half8 am = *reinterpret_cast<const half8*>(&L.Ms[(16 * ti + r16) * LDB + 16 * ks + kh]);
        const half8 bx = *reinterpret_cast<const half8*>(&L.xT[p * LDB + 16 * ks + kh]);
        yi = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(am, bx, yi);
    }
#pragma unroll
    for (int j = 0; j < SN / 16; j++) {
        const half8 ac = *reinterpret_cast<const half8*>(&c16[(16 * ti + r16) * SN + 16 * j + kh]);
        yo = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ac, sf[j], yo);
    }
}

// A chunk and the index of the 16 rows t = 16 ti ... in it
struct RowTile {
    Chunk ck;
    int ti;
};

// y of the 16 rows t = 16 ti ... of this wave's 16 columns p: intra-chunk + inter-chunk + skip
__device__ void chunk_output(const SsdArgs& a, const RowTile rt, int h, const SsdLds& L, const _Float16* c16,
                             const half8 (&sf)[SN / 16]) {
    const Chunk ck = rt.ck;
    const int ti = rt.ti;
    const int lane = threadIdx.x % 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int p = threadIdx.x / 32 * 16 + r16;
    float8 yi = {}, yo = {};
    chunk_products(L, c16, sf, ti, yi, yo);
    const float Dh = a.D[h];
    const half8 xs = *reinterpret_cast<const half8*>(&L.xr[p * LDB + 16 * ti + kh]);
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const int t = 16 * ti + kh + e;
        if (t >= ck.len) continue;
        a.y[(int64_t)(ck.first + t) * a.inner + h * SP + p] = yi[e] + __expf(L.cs[t]) * yo[e] + Dh * (float)xs[e];
    }
}

// S^T = exp(cs_Q) S^T + Bt Xw over the chunk
__device__ void chunk_state(const SsdLds& L, const _Float16* bt16, float scale, float8 (&st)[SN / 16]) {
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
#pragma unroll
    for (int j = 0; j < SN / 16; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) st[j][e] *= scale;
#pragma unroll
    for (int ks = 0; ks < SQ / 16; ks++) {
        const half8 bx = *reinterpret_cast<const half8*>(&L.xT[(w * 16 + r16) * LDB + 16 * ks + kh]);
        const float4 w0 = *reinterpret_cast<const float4*>(&L.wgt[16 * ks + kh]);
        const float4 w1 = *reinterpret_cast<const float4*>(&L.wgt[16 * ks + kh + 4]);
        const half8 bw =
            half8{(_Float16)((float)bx[0] * w0.x), (_Float16)((float)bx[1] * w0.y), (_Float16)((float)bx[2] * w0.z),
                  (_Float16)((float)bx[3] * w0.w), (_Float16)((float)bx[4] * w1.x), (_Float16)((float)bx[5] * w1.y),
                  (_Float16)((float)bx[6] * w1.z), (_Float16)((float)bx[7] * w1.w)};
#pragma unroll
        for (int j = 0; j < SN / 16; j++) {
            const half8 ab = *reinterpret_cast<const half8*>(&bt16[(16 * j + r16) * SQ + 16 * ks + kh]);
            st[j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(ab, bw, st[j]);
        }
    }
}

// The state in the accumulators: S^T[n][p] of this wave's 16 columns, lane holds rows n = 16 j + kh ... + 7
__device__ void state_io(const SsdArgs& a, int h, float8 (&st)[SN / 16], bool store) {
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    float* row = a.state + ((int64_t)h * SP + w * 16 + r16) * SN;
#pragma unroll
    for (int j = 0; j < SN / 16; j++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            if (store)
                row[16 * j + kh + e] = st[j][e];
            else
                st[j][e] = row[16 * j + kh + e];
        }
}

// One block per head: the chunks one after the other
__global__ void __launch_bounds__(SNT) k_ssd_scan(SsdArgs a) {
    __shared__ SsdLds L;
    const int h = blockIdx.x, g = h / (a.heads / a.groups);
    float8 st[SN / 16];
    state_io(a, h, st, false);
    Pending pend;
    request(a, chunk_of(0, a.pos0, a.n), 0, h, pend);
    for (int c = 0; c < a.nc; c++) {
        const Chunk ck = chunk_of(c, a.pos0, a.n);
        store_cs(pend, L);
        __syncthreads();
        store_chunk(pend, ck, L);
        __syncthreads();
        if (c + 1 < a.nc) request(a, chunk_of(c + 1, a.pos0, a.n), c + 1, h, pend);  // in flight during the products
        const PrepPtrs pp = prep_ptrs(a, c, g);
        half8 sf[SN / 16];
#pragma unroll
        for (int j = 0; j < SN / 16; j++) sf[j] = to_half8(st[j]);
        for (int ti = 0; ti < SQ / 16; ti++)
            if (16 * ti < ck.len) chunk_output(a, {ck, ti}, h, L, pp.c16, sf);
        chunk_state(L, pp.bt16, __expf(L.cs[ck.len - 1]), st);
        __syncthreads();
    }
    state_io(a, h, st, true);
}

}  // namespace

bool ssd_supported(int heads, int P, int state, int groups) {
    return P == SP && state == SN && groups > 0 && heads % groups == 0;
}

int ssd_chunks(int64_t pos0, int n) {
    if (n <= 0) return 0;
    const int head = SQ - (int)(pos0 % SQ);
    return n <= head ? 1 : 1 + (n - head + SQ - 1) / SQ;
}

void launch_mamba_ssd(SsdArgs a, hipStream_t st) {
    const int nc = a.nc = ssd_chunks(a.pos0, a.n);
    if (nc == 0) return;
    if (!ssd_supported(a.heads, a.P, MB_STATE, a.groups))
        throw std::runtime_error(
            std::format("gpu: the chunked Mamba-2 scan is written for {} rows and state {}", SP, SN));
    const int64_t cells = (int64_t)nc * a.groups;
    a.c16 = reinterpret_cast<_Float16*>(a.cb + cells * SQ * SQ);
    a.bt16 = a.c16 + cells * SQ * SN;
    k_ssd_prep<<<dim3(nc, a.groups), SNT, 0, st>>>(a);
    k_ssd_scan<<<a.heads, SNT, 0, st>>>(a);
}

}  // namespace tona::gpu
