// Mat-vecs y = W x over any decoded weight type for B tokens at once (x [B][k], y [B][rows]); every token is computed
// with the same arithmetic as alone (rows_dot keeps the tokens apart), so all widths give bit-identical results.
// Dense matrices run with one slot; routed experts with one slot per chosen expert and B = 1: slot s reads expert
// ids[s] (W offset ids[s] * ebytes), input x + (s / xdiv) * xstride, and writes y + s * rows. Two shapes, as measured
// in research/measurements/2026-11-11-mac-densek.md: rows spread over simdgroups (mv), and split-K for matrices with
// few rows (mvs), where 8 simdgroups share R rows.

static inline device const uchar* mv_weights(device const uchar* w, device const int* ids, constant MvParams& P,
                                             uint slot) {
    return P.use_ids ? w + ulong(ids[slot]) * P.ebytes : w;
}

template <typename T, int R, int B>
kernel void mv(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
               device const int* ids [[buffer(3)]], constant MvParams& P [[buffer(4)]],
               uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
               uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r0 = (tg.x * nsg + sg) * R;
    if (r0 >= P.rows) return;
    device const uchar* ws = mv_weights(w, ids, P, tg.y);
    float a[R][B];
    rows_dot<T, B, R>(ws, r0, P.row_bytes, x + (tg.y / P.xdiv) * P.xstride, P.k, lane, a);
    if (lane == 0)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < B; ++b) y[(tg.y + b) * P.rows + r0 + r] = a[r][b];
}

template <typename T, int R, int B>
kernel void mvs(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                device const int* ids [[buffer(3)]], constant MvParams& P [[buffer(4)]],
                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane [[thread_index_in_simdgroup]]) {
    constexpr int NS = 8;
    threadgroup float part[NS][R][B];
    uint r0 = tg.x * R;
    device const uchar* ws = mv_weights(w, ids, P, tg.y);
    device const float* xs = x + (tg.y / P.xdiv) * P.xstride;
    float acc[R][B];
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    const Rows<T, R> rows{ws, r0, P.row_bytes, P.k};
    for (uint g = sg * 32 + lane; g < P.k / 8; g += NS * 32) {
        float4 xa[B], xc[B];
        for (int b = 0; b < B; ++b) {
            device const float4* xv = (device const float4*)(xs + b * P.k + g * 8);
            xa[b] = xv[0], xc[b] = xv[1];
        }
        const auto f = rows.load(g);
        for (int r = 0; r < R; ++r) {
            float wv[8];
            rows.dec(f, r, wv);
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
        const uint r = lane / B, b = lane % B;
        float s = 0.f;
        for (int i = 0; i < NS; ++i) s += part[i][r][b];
        y[(tg.y + b) * P.rows + r0 + r] = s;
    }
}

// rows_dot with the input of token b at xs[b]: the same loop and order, so a row and token get the same bits
using XRow = device const float*;

template <typename T, int B, int R>
static inline void rows_dot_at(device const uchar* row, uint rs, thread const XRow (&xs)[B], uint k, uint lane,
                               thread float (&acc)[R][B]) {
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    for (uint g = lane; g < k / 8; g += 32) {
        float4 xa[B], xc[B];
        for (int b = 0; b < B; ++b) {
            device const float4* xv = (device const float4*)(xs[b] + g * 8);
            xa[b] = xv[0], xc[b] = xv[1];
        }
        for (int r = 0; r < R; ++r) {
            float w[8];
            T::dec(row + r * rs, g, w);
            float4 wa = float4(w[0], w[1], w[2], w[3]), wc = float4(w[4], w[5], w[6], w[7]);
            for (int b = 0; b < B; ++b) acc[r][b] += dot(wa, xa[b]) + dot(wc, xc[b]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = simd_sum(acc[r][b]);
}

// The buffers of one expert's slots: its matrix, the inputs and outputs of all slots, the slot list
struct MvgIo {
    device const uchar* ws;
    device const float* x;
    device float* y;
    device const int* perm;
};

// B slots of one expert from perm[at]: rows r0 .. r0 + R of the expert's matrix for each
template <typename T, int R, int B>
static inline void mvg_slots(thread const MvgIo& io, int at, uint r0, constant MvParams& P, uint lane) {
    XRow xs[B];
    for (int b = 0; b < B; ++b) xs[b] = io.x + uint(io.perm[at + b]) / P.xdiv * P.xstride;
    float a[R][B];
    rows_dot_at<T, B, R>(io.ws + ulong(r0) * P.row_bytes, P.row_bytes, xs, P.k, lane, a);
    if (lane == 0)
        for (int b = 0; b < B; ++b)
            for (int r = 0; r < R; ++r) io.y[uint(io.perm[at + b]) * P.rows + r0 + r] = a[r][b];
}

// Routed experts of a decode pass grouped by expert (moe_group): threadgroup row j takes the j-th expert the pass uses
// (glist) and all its slots, so an expert several tokens share is decoded once for all of them. Width 1 runs the same
// kernel, so every width gives the bits of width 1. At most 4 slots per weight pass: 5- and 8-slot instances cost 1.8
// and 5 ms per pass at width 4 for their registers (research/measurements/2026-11-13-mac-mv8.md).
template <typename T, int R, int MB>
kernel void mvg(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                device const int* off [[buffer(3)]], device const int* perm [[buffer(4)]],
                device const int* glist [[buffer(5)]], constant MvParams& P [[buffer(6)]],
                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int g = glist[tg.y];
    const uint r0 = (tg.x * nsg + sg) * R;
    if (g < 0 || r0 >= P.rows) return;
    // P.tokens: slots per glist entry (moe_group split), 0: the entry holds all slots of its expert
    const int e = g & 0xffff, split = int(P.tokens);
    const MvgIo io{w + ulong(e) * P.ebytes, x, y, perm};
    int at = off[e] + (g >> 16) * split;
    const int end = split ? min(off[e + 1], at + split) : off[e + 1];
    for (; end - at >= MB; at += MB) mvg_slots<T, R, MB>(io, at, r0, P, lane);
    if (MB > 3 && end - at == 3) mvg_slots<T, R, 3>(io, at, r0, P, lane);
    if (MB > 2 && end - at == 2) mvg_slots<T, R, 2>(io, at, r0, P, lane);
    if (end - at == 1) mvg_slots<T, R, 1>(io, at, r0, P, lane);
}

// clang-format off
#define MV_ONE(T, R, B)                                                                                          \
    template [[host_name("mv_" #T "_r" #R "_b" #B)]] kernel void mv<T, R, B>(                                    \
        device const uchar*, device const float*, device float*, device const int*, constant MvParams&, uint2,   \
        uint, uint, uint);                                                                                       \
    template [[host_name("mvs_" #T "_r" #R "_b" #B)]] kernel void mvs<T, R, B>(                                  \
        device const uchar*, device const float*, device float*, device const int*, constant MvParams&, uint2,   \
        uint, uint);
#define MVG_MB(T, R, MB, NAME)                                                                                   \
    template [[host_name(NAME #T "_r" #R)]] kernel void mvg<T, R, MB>(device const uchar*, device const float*,   \
        device float*, device const int*, device const int*, device const int*, constant MvParams&, uint2, uint,  \
        uint, uint);
#define MVG_ONE(T, R) MVG_MB(T, R, 4, "mvg_") MVG_MB(T, R, 2, "mvg2_")
#define MV_TYPE(T) MVG_ONE(T, 1) MVG_ONE(T, 4) MV_ONE(T, 1, 1) MV_ONE(T, 4, 1) MV_ONE(T, 1, 2) MV_ONE(T, 4, 2) MV_ONE(T, 1, 3) MV_ONE(T, 4, 3) \
    MV_ONE(T, 1, 4) MV_ONE(T, 4, 4) MV_ONE(T, 1, 5) MV_ONE(T, 4, 5)
#define MV_DENSE(T) MV_ONE(T, 4, 1) MV_ONE(T, 4, 2) MV_ONE(T, 4, 3) MV_ONE(T, 4, 4) MV_ONE(T, 4, 5)
// clang-format on
MV_TYPE(F32)
MV_TYPE(F16)
MV_TYPE(BF16)
MV_TYPE(Q80)
MV_TYPE(Q4K)
MV_TYPE(Q5K)
MV_TYPE(Q6K)
MV_TYPE(IQ4XS)
MV_TYPE(IQ4NL)
MV_TYPE(IQ2XXS)
MV_TYPE(IQ2S)
MV_TYPE(IQ3XXS)
MV_TYPE(IQ3XXSR)
MV_DENSE(Q6KT)  // tile type: dense matrices only (rows a multiple of 8, so R = 4)
MV_TYPE(IQ3S)
MV_TYPE(Q20)

// Rows of a weight table into f32 (embedding row, n-gram rows): row r of the output is table row idx[r], written
// `copies` times one after the other (output row r * copies + c). One thread per group of 8.
template <typename T>
kernel void gather(device const uchar* w [[buffer(0)]], device const int* idx [[buffer(1)]],
                   device float* y [[buffer(2)]], constant GatherParams& P [[buffer(3)]],
                   uint2 gid [[thread_position_in_grid]]) {
    uint g = gid.x, r = gid.y;
    if (g >= P.n / 8) return;
    float v[8];
    T::dec(w + ulong(idx[r]) * P.row_bytes, g, v);
    for (uint c = 0; c < P.copies; ++c)
        for (int k = 0; k < 8; ++k) y[(r * P.copies + c) * P.n + g * 8 + k] = v[k];
}

#define GATHER(T)                                                                                                     \
    template [[host_name("gather_" #T)]] kernel void gather<T>(device const uchar*, device const int*, device float*, \
                                                               constant GatherParams&, uint2);
GATHER(F32)
GATHER(F16)
GATHER(BF16)
GATHER(Q80)
GATHER(Q4K)
GATHER(Q5K)
GATHER(Q6K)
GATHER(IQ4XS)
GATHER(IQ4NL)
GATHER(IQ2XXS)
GATHER(IQ2S)
GATHER(IQ3XXS)
GATHER(IQ3XXSR)
GATHER(IQ3S)
GATHER(Q20)
