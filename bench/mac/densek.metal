// Dense-part kernels for timing (M1): a generic mat-vec over any decoded type and the DeltaNet state step.
// First cut, not the backend's kernels. The host prepends the IQ tables and decode.metal.

struct MvParams {
    uint rows, k, row_bytes;
};

// y[b][r] = W_r . x_b; each simdgroup R consecutive rows
template <typename T, int B, int R>
kernel void mv(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
               constant MvParams& P [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
               uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
               uint lane [[thread_index_in_simdgroup]]) {
    uint r0 = (tg * nsg + sg) * R;
    if (r0 >= P.rows) return;
    float a[R][B];
    rows_dot<T, B, R>(w + ulong(r0) * P.row_bytes, P.row_bytes, x, P.k, lane, a);
    if (lane == 0)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < B; ++b) y[b * P.rows + r0 + r] = a[r][b];
}

// clang-format off
#define MV(T, B, R)                                                                                                 \
    template [[host_name("mv_" #T "_" #B "_r" #R)]] kernel void mv<T, B, R>(                                        \
        device const uchar*, device const float*, device float*, constant MvParams&, uint, uint, uint, uint);
// clang-format on
// clang-format off
#define MV_ALL(T) MV(T, 1, 1) MV(T, 2, 1) MV(T, 3, 1) MV(T, 4, 1) MV(T, 1, 4) MV(T, 2, 4) MV(T, 3, 4) MV(T, 4, 4) \
    MV(T, 1, 2) MV(T, 2, 2) MV(T, 3, 2) MV(T, 4, 2) MV(T, 1, 8) MV(T, 2, 8) MV(T, 3, 8) MV(T, 4, 8)
// clang-format on
MV_ALL(Q4K)
MV_ALL(Q5K)
MV_ALL(Q6K)
MV_ALL(Q80)
MV_ALL(IQ4XS)
MV_ALL(IQ4NL)
MV_ALL(BF16)

// Diagnostic only (wrong results): the loads of Q6K::dec without its arithmetic, to separate memory from ALU time
struct Q6KL {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 210;
        uint j = g % 32, n = j / 16, v0 = 8 * (j % 16), qq = v0 / 32, l0 = v0 % 32;
        uint2 ql = load8_a2(b + 64 * n + (qq & 1) * 32 + l0);
        uint2 qh = load8_a2(b + 128 + 32 * n + l0);
        float d = h2f(b + 208);
        float4 f = float4(as_type<uchar4>(ql.x ^ qh.y)) * d, h = float4(as_type<uchar4>(ql.y ^ qh.x));
        w[0] = f.x, w[1] = f.y, w[2] = f.z, w[3] = f.w, w[4] = h.x, w[5] = h.y, w[6] = h.z, w[7] = h.w;
    }
};
MV_ALL(Q6KL)

// Diagnostic only: as Q6KL, but every 8-byte load aligned down to 8 bytes (one uint2 load)
static inline uint2 load8_down(device const uchar* p) { return *(device const uint2*)((ulong)p & ~7ul); }
struct Q6KA {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 210;
        uint j = g % 32, n = j / 16, v0 = 8 * (j % 16), qq = v0 / 32, l0 = v0 % 32;
        uint2 ql = load8_down(b + 64 * n + (qq & 1) * 32 + l0);
        uint2 qh = load8_down(b + 128 + 32 * n + l0);
        float d = h2f(b + 208);
        float4 f = float4(as_type<uchar4>(ql.x ^ qh.y)) * d, h = float4(as_type<uchar4>(ql.y ^ qh.x));
        w[0] = f.x, w[1] = f.y, w[2] = f.z, w[3] = f.w, w[4] = h.x, w[5] = h.y, w[6] = h.z, w[7] = h.w;
    }
};
MV_ALL(Q6KA)

// The same mat-vec with x staged in threadgroup memory once per threadgroup (all its simdgroups read it from there);
// only for B * k <= 8192 floats (32 KB). Same products in the same order as mv: bit-identical results.
template <typename T, int B, int R>
kernel void mvt(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                constant MvParams& P [[buffer(3)]], threadgroup float* xs [[threadgroup(0)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                uint tid [[thread_index_in_threadgroup]], uint nth [[threads_per_threadgroup]]) {
    for (uint i = tid; i < B * P.k / 4; i += nth) ((threadgroup float4*)xs)[i] = ((device const float4*)x)[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint r0 = (tg * nsg + sg) * R;
    if (r0 >= P.rows) return;
    device const uchar* row = w + ulong(r0) * P.row_bytes;
    float a[R][B];
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) a[r][b] = 0.f;
    for (uint g = lane; g < P.k / 8; g += 32) {
        float4 xa[B], xc[B];
        for (int b = 0; b < B; ++b) {
            threadgroup const float4* xv = (threadgroup const float4*)(xs + b * P.k + g * 8);
            xa[b] = xv[0], xc[b] = xv[1];
        }
        for (int r = 0; r < R; ++r) {
            float wv[8];
            T::dec(row + r * P.row_bytes, g, wv);
            float4 wa = float4(wv[0], wv[1], wv[2], wv[3]), wc = float4(wv[4], wv[5], wv[6], wv[7]);
            for (int b = 0; b < B; ++b) a[r][b] += dot(wa, xa[b]) + dot(wc, xc[b]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) a[r][b] = simd_sum(a[r][b]);
    if (lane == 0)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < B; ++b) y[b * P.rows + r0 + r] = a[r][b];
}

// clang-format off
#define MVT(T, B, R)                                                                                                \
    template [[host_name("mvt_" #T "_" #B "_r" #R)]] kernel void mvt<T, B, R>(                                      \
        device const uchar*, device const float*, device float*, constant MvParams&, threadgroup float*, uint, uint, \
        uint, uint, uint, uint);
// clang-format on
#define MVT_ALL(T) \
    MVT(T, 1, 1) MVT(T, 2, 1) MVT(T, 3, 1) MVT(T, 4, 1) MVT(T, 1, 4) MVT(T, 2, 4) MVT(T, 3, 4) MVT(T, 4, 4)
MVT_ALL(Q4K)
MVT_ALL(Q5K)
MVT_ALL(Q6K)
MVT_ALL(Q80)
MVT_ALL(IQ4XS)
MVT_ALL(IQ4NL)
MVT_ALL(BF16)

// Split-K variant for matrices with few rows: the NS simdgroups of a threadgroup share R rows and split the groups of
// 8 among them; partial sums meet in threadgroup memory. NS = 8.
template <typename T, int B, int R>
kernel void mvs(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                constant MvParams& P [[buffer(3)]], uint tg [[threadgroup_position_in_grid]],
                uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NS = 8;
    threadgroup float part[NS][R][B];
    uint r0 = tg * R;
    float acc[R][B];
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    for (uint g = sg * 32 + lane; g < P.k / 8; g += NS * 32) {
        float4 xa[B], xc[B];
        for (int b = 0; b < B; ++b) {
            device const float4* xv = (device const float4*)(x + b * P.k + g * 8);
            xa[b] = xv[0], xc[b] = xv[1];
        }
        for (int r = 0; r < R; ++r) {
            float wv[8];
            T::dec(w + ulong(r0 + r) * P.row_bytes, g, wv);
            float4 wa = float4(wv[0], wv[1], wv[2], wv[3]), wc = float4(wv[4], wv[5], wv[6], wv[7]);
            for (int b = 0; b < B; ++b) acc[r][b] += dot(wa, xa[b]) + dot(wc, xc[b]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) {
            float s = simd_sum(acc[r][b]);
            if (lane == 0) part[sg][r][b] = s;
        }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0 && lane < uint(R * B)) {
        uint r = lane / B, b = lane % B;
        float s = 0.f;
        for (int i = 0; i < NS; ++i) s += part[i][r][b];
        y[b * P.rows + r0 + r] = s;
    }
}

// clang-format off
#define MVS(T, B, R)                                                                                                \
    template [[host_name("mvs_" #T "_" #B "_r" #R)]] kernel void mvs<T, B, R>(                                      \
        device const uchar*, device const float*, device float*, constant MvParams&, uint, uint, uint);
// clang-format on
#define MVS_ALL(T) \
    MVS(T, 1, 1) MVS(T, 2, 1) MVS(T, 3, 1) MVS(T, 4, 1) MVS(T, 1, 4) MVS(T, 2, 4) MVS(T, 3, 4) MVS(T, 4, 4)
MVS_ALL(Q4K)
MVS_ALL(Q5K)
MVS_ALL(Q6K)
MVS_ALL(Q80)
MVS_ALL(IQ4XS)
MVS_ALL(IQ4NL)
MVS_ALL(BF16)

// Gated delta rule for B consecutive tokens of one layer, as engine/cpu/ref.cpp Forward::deltanet: per V head j and
// state row vi: s *= alpha; d = (v - s.k) * beta; s += k * d; o = (s.q) * qscale. 128-wide rows, one simdgroup per
// row (4 values per lane), state kept in registers across the B tokens and written back once.
// Inputs per token b: q, k [nk][128] (V head j uses K head j % nk), v [nv][128], alpha, beta [nv].
struct DnParams {
    uint nk, nv;
    float qscale;
};

template <int B>
kernel void dn_step(device float* state [[buffer(0)]], device const float* q [[buffer(1)]],
                    device const float* k [[buffer(2)]], device const float* v [[buffer(3)]],
                    device const float* alpha [[buffer(4)]], device const float* beta [[buffer(5)]],
                    device float* o [[buffer(6)]], constant DnParams& P [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint j = tg.y, kh = j % P.nk;
    for (uint vi = tg.x * nsg * 8 + sg * 8; vi < tg.x * nsg * 8 + sg * 8 + 8; ++vi) {
        device float4* row = (device float4*)(state + (ulong(j) * 128 + vi) * 128) + lane;
        float4 s = *row;
        for (int b = 0; b < B; ++b) {
            float4 kk = ((device const float4*)(k + (b * P.nk + kh) * 128))[lane];
            float4 qq = ((device const float4*)(q + (b * P.nk + kh) * 128))[lane];
            s *= alpha[b * P.nv + j];
            float sk = simd_sum(dot(s, kk));
            float dl = (v[(b * P.nv + j) * 128 + vi] - sk) * beta[b * P.nv + j];
            s += kk * dl;
            float ov = simd_sum(dot(s, qq));
            if (lane == 0) o[(b * P.nv + j) * 128 + vi] = ov * P.qscale;
        }
        *row = s;
    }
}

template [[host_name("dn_step_1")]] kernel void dn_step<1>(device float*, device const float*, device const float*,
                                                           device const float*, device const float*,
                                                           device const float*, device float*, constant DnParams&,
                                                           uint2, uint, uint, uint);
template [[host_name("dn_step_2")]] kernel void dn_step<2>(device float*, device const float*, device const float*,
                                                           device const float*, device const float*,
                                                           device const float*, device float*, constant DnParams&,
                                                           uint2, uint, uint, uint);
template [[host_name("dn_step_3")]] kernel void dn_step<3>(device float*, device const float*, device const float*,
                                                           device const float*, device const float*,
                                                           device const float*, device float*, constant DnParams&,
                                                           uint2, uint, uint, uint);
template [[host_name("dn_step_4")]] kernel void dn_step<4>(device float*, device const float*, device const float*,
                                                           device const float*, device const float*,
                                                           device const float*, device float*, constant DnParams&,
                                                           uint2, uint, uint, uint);
