// Matrix products of the prefill: Y[t][r] = sum_k W[r][k] X[t][k] for T tokens over any decoded weight type, on the
// 8x8 matrix units (simdgroup_matrix, f32: on Apple7 f16 runs at the same rate). A threadgroup computes 64 weight rows
// × 64 tokens; per K step of 32 it decodes its weight tile into threadgroup memory and loads the token tile beside it.
// Each decoded weight serves up to 64 tokens; a simdgroup whose 16 tokens lie past the end of the tile (an expert with
// few tokens) skips its products, so the work follows the token count in steps of 16.
// Every output element is reduced over K alone, in a fixed order, and never across tokens: a token's result does not
// depend on which other tokens share its tile, so where a prefill is cut does not matter (docs/use/prompt-cache.md).

constant constexpr uint GM = 64;      // weight rows per threadgroup
constant constexpr uint GT = 64;      // tokens per threadgroup
constant constexpr uint GTH = 256;    // threads per threadgroup: 8 simdgroups
constant constexpr uint GK = 32;      // K per step
constant constexpr uint GS = GK + 4;  // row stride in threadgroup memory: rows 32 floats apart share their banks
constant constexpr uint XS = GT + 4;  // the token tile is stored [k][token]

// Threadgroup memory of a tile and the source and target row of each of its tokens (-1: no token)
struct GemmTg {
    threadgroup float* ws;  // [row][k]
    threadgroup float* xs;  // [token][k]
    threadgroup int* xr;
    threadgroup int* yr;
};

struct GemmAt {
    uint r0, tid, sg, lane;
    bool contig;  // token j of the tile is row yr[0] + j for all valid j: direct stores
};

template <typename T>
static inline void gemm_load_w(device const uchar* w, constant GemmParams& P, uint r0, uint k0, uint tid,
                               threadgroup float* ws) {
    for (uint i = tid; i < GM * GK / 8; i += GTH) {
        const uint r = i / (GK / 8), g = i % (GK / 8);
        float v[8];
        if (r0 + r < P.rows)
            RowAt<T>::dec(w, r0 + r, P.row_bytes, k0 / 8 + g, v);
        else
            for (int j = 0; j < 8; ++j) v[j] = 0.f;
        threadgroup float4* dst = (threadgroup float4*)(ws + r * GS + g * 8);
        dst[0] = float4(v[0], v[1], v[2], v[3]);
        dst[1] = float4(v[4], v[5], v[6], v[7]);
    }
}

static inline void gemm_load_x(device const float* x, constant GemmParams& P, uint k0, uint tid, GemmTg m) {
    for (uint i = tid; i < GT * GK / 4; i += GTH) {
        const uint t = i / (GK / 4), q = i % (GK / 4);
        const int row = m.xr[t];
        const float4 v = row >= 0 ? ((device const float4*)(x + ulong(row) * P.k + k0))[q] : float4(0.f);
        for (uint e = 0; e < 4; ++e) m.xs[(q * 4 + e) * XS + t] = v[e];
    }
}

// The 8x8 result block (rows r ... r + 7, tile tokens j ... j + 7) into y[row][r]; blocks on the edge or with scattered
// rows go through threadgroup memory
static inline void gemm_store(simdgroup_float8x8 c, device float* y, constant GemmParams& P, uint2 at, GemmTg m,
                              GemmAt a) {
    const uint r = at.x, j = at.y;
    if (a.contig && r + 8 <= P.rows && m.yr[j + 7] >= 0) {
        simdgroup_store(c, y + ulong(m.yr[j]) * P.rows + r, P.rows, ulong2(0, 0), true);
        return;
    }
    threadgroup float* stage = m.ws + a.sg * 64;
    simdgroup_store(c, stage, 8);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = a.lane; e < 64; e += 32) {
        const int row = m.yr[j + e % 8];
        if (r + e / 8 < P.rows && row >= 0) y[ulong(row) * P.rows + r + e / 8] = stage[e];
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
}

// 8 simdgroups: simdgroup s takes rows (s / 4) * 32 ... + 32 and tokens (s % 4) * 16 ... + 16 of the tile
template <typename T>
static inline void gemm_body(device const uchar* w, device const float* x, device float* y, constant GemmParams& P,
                             GemmTg m, GemmAt a) {
    const uint mr = (a.sg / 4) * 32, mt = (a.sg % 4) * 16;
    simdgroup_float8x8 acc[4][2];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);  // xr, yr written
    const bool busy = m.xr[mt] >= 0;                  // tokens fill a tile from its start
    for (uint k0 = 0; k0 < P.k; k0 += GK) {
        gemm_load_w<T>(w, P, a.r0, k0, a.tid, m.ws);
        gemm_load_x(x, P, k0, a.tid, m);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint kk = 0; busy && kk < GK; kk += 8) {
            simdgroup_float8x8 wa[4], xb[2];
            for (int i = 0; i < 4; ++i) simdgroup_load(wa[i], m.ws + (mr + i * 8) * GS + kk, GS);
            for (int j = 0; j < 2; ++j) simdgroup_load(xb[j], m.xs + kk * XS + mt + j * 8, XS);
            for (int i = 0; i < 4; ++i)
                for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], wa[i], xb[j], acc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (int i = 0; i < 4 && busy; ++i)
        for (int j = 0; j < 2; ++j) gemm_store(acc[i][j], y, P, uint2(a.r0 + mr + i * 8, mt + j * 8), m, a);
}

// Dense: tokens t0 ... t0 + 63 of x and y
template <typename T>
kernel void gemm(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                 device float* y [[buffer(2)]], constant GemmParams& P [[buffer(3)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float ws[GM * GS], xs[GK * XS];
    threadgroup int xr[GT], yr[GT];
    if (tid < GT) xr[tid] = yr[tid] = tg.y * GT + tid < P.tokens ? int(tg.y * GT + tid) : -1;
    gemm_body<T>(w, x, y, P, GemmTg{ws, xs, xr, yr}, GemmAt{tg.x * GM, tid, sg, lane, true});
}

// Routed experts grouped by expert (tg.z): the tile takes slots off[e] + tg.y * 64 ... of the slot list perm
// (moe_group); slot s reads x row s / xdiv and writes y row s, with the weights of expert e
template <typename T>
kernel void gemmx(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                  device float* y [[buffer(2)]], device const int* off [[buffer(3)]],
                  device const int* perm [[buffer(4)]], constant GemmParams& P [[buffer(5)]],
                  uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int first = off[tg.z] + int(tg.y * GT), n = off[tg.z + 1] - first;
    if (n <= 0) return;  // the whole threadgroup: no expert has a tile here
    threadgroup float ws[GM * GS], xs[GK * XS];
    threadgroup int xr[GT], yr[GT];
    if (tid < GT) {
        const int s = int(tid) < n ? perm[first + int(tid)] : -1;
        xr[tid] = s >= 0 ? s / int(P.xdiv) : -1;
        yr[tid] = s;
    }
    gemm_body<T>(w + ulong(tg.z) * P.ebytes, x, y, P, GemmTg{ws, xs, xr, yr}, GemmAt{tg.x * GM, tid, sg, lane, false});
}

// Matrices of fewer than 16 rows (the gated reads' inject, 4 rows): a GEMM tile of 64 rows would idle 94 % of its
// work. Threadgroup per token, simdgroup per row; lane l sums the groups l, l + 32, ... and the lanes add in a fixed
// order, so a token's result does not depend on the other tokens.
template <typename T>
kernel void gemv_narrow(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]],
                        device float* y [[buffer(2)]], constant GemmParams& P [[buffer(3)]],
                        uint t [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    if (sg >= P.rows) return;
    device const float4* xt = (device const float4*)(x + ulong(t) * P.k);
    float acc = 0.f;
    for (uint g = lane; g < P.k / 8; g += 32) {
        float v[8];
        RowAt<T>::dec(w, sg, P.row_bytes, g, v);
        acc += dot(float4(v[0], v[1], v[2], v[3]), xt[2 * g]) + dot(float4(v[4], v[5], v[6], v[7]), xt[2 * g + 1]);
    }
    acc = simd_sum(acc);
    if (lane == 0) y[ulong(t) * P.rows + sg] = acc;
}

// clang-format off
#define GEMM(T) \
    template [[host_name("gemm_" #T)]] kernel void gemm<T>(device const uchar*, device const float*, device float*, \
                                                           constant GemmParams&, uint2, uint, uint, uint); \
    template [[host_name("gemmx_" #T)]] kernel void gemmx<T>(device const uchar*, device const float*, device float*, \
        device const int*, device const int*, constant GemmParams&, uint3, uint, uint, uint); \
    template [[host_name("gemv_narrow_" #T)]] kernel void gemv_narrow<T>(device const uchar*, device const float*, \
        device float*, constant GemmParams&, uint, uint, uint);
GEMM(F32)
GEMM(F16)
GEMM(BF16)
GEMM(Q80)
GEMM(Q4K)
GEMM(Q5K)
GEMM(Q6K)
GEMM(IQ4XS)
GEMM(IQ4NL)
GEMM(IQ2XXS)
GEMM(IQ2S)
GEMM(IQ3XXS)
GEMM(IQ3XXSR)
// tile type: dense matrices only, no expert GEMM
template [[host_name("gemm_Q6KT")]] kernel void gemm<Q6KT>(device const uchar*, device const float*, device float*,
                                                          constant GemmParams&, uint2, uint, uint, uint);
template [[host_name("gemv_narrow_Q6KT")]] kernel void gemv_narrow<Q6KT>(device const uchar*, device const float*,
                                                                        device float*, constant GemmParams&, uint, uint,
                                                                        uint);
GEMM(IQ3S)
GEMM(Q20)
// clang-format on
