// Routed experts read through a table of addresses instead of w + expert * ebytes: expert e of the matrix lies at
// tab[e].p. For a model whose experts do not all fit the GPU's working set (the arena, arena.h): the used ones are in
// blocks of their own, the others are read from the SSD into staging slots before the layer runs; a block can be freed
// under memory pressure without moving the others. The same arithmetic as mv, mvg and gemmx, so a pass gives the same
// bits with and without the table.

// A GPU address written by the host (MTLBuffer.gpuAddress plus an offset); a pointer must be wrapped to be a buffer's
// element type
struct ExpertAt {
    device const uchar* p;
};

template <typename T, int R>
kernel void mvt(device const float* x [[buffer(0)]], device float* y [[buffer(1)]], device const int* ids [[buffer(2)]],
                device const ExpertAt* tab [[buffer(3)]], constant MvParams& P [[buffer(4)]],
                uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r0 = (tg.x * nsg + sg) * R;
    if (r0 >= P.rows) return;
    device const uchar* ws = tab[ids[tg.y]].p;
    float a[R][1];
    rows_dot<T, 1, R>(ws, r0, P.row_bytes, x + (tg.y / P.xdiv) * P.xstride, P.k, lane, a);
    if (lane == 0)
        for (int r = 0; r < R; ++r) y[tg.y * P.rows + r0 + r] = a[r][0];
}

template <typename T, int R, int MB>
kernel void mvgt(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                 device const int* off [[buffer(2)]], device const int* perm [[buffer(3)]],
                 device const int* glist [[buffer(4)]], device const ExpertAt* tab [[buffer(5)]],
                 constant MvParams& P [[buffer(6)]], uint2 tg [[threadgroup_position_in_grid]],
                 uint sg [[simdgroup_index_in_threadgroup]], uint nsg [[simdgroups_per_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
    const int g = glist[tg.y];
    const uint r0 = (tg.x * nsg + sg) * R;
    if (g < 0 || r0 >= P.rows) return;
    const int e = g & 0xffff, split = int(P.tokens);
    const MvgIo io{tab[e].p, x, y, perm};
    int at = off[e] + (g >> 16) * split;
    const int end = split ? min(off[e + 1], at + split) : off[e + 1];
    for (; end - at >= MB; at += MB) mvg_slots<T, R, MB>(io, at, r0, P, lane);
    if (MB > 3 && end - at == 3) mvg_slots<T, R, 3>(io, at, r0, P, lane);
    if (MB > 2 && end - at == 2) mvg_slots<T, R, 2>(io, at, r0, P, lane);
    if (end - at == 1) mvg_slots<T, R, 1>(io, at, r0, P, lane);
}

template <typename T>
kernel void gemmxt(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                   device const int* off [[buffer(2)]], device const int* perm [[buffer(3)]],
                   device const ExpertAt* tab [[buffer(4)]], constant GemmParams& P [[buffer(5)]],
                   uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const int first = off[tg.z] + int(tg.y * GT), n = off[tg.z + 1] - first;
    if (n <= 0) return;
    threadgroup float ws[GM * GS], xs[GK * XS];
    threadgroup int xr[GT], yr[GT];
    if (tid < GT) {
        const int s = int(tid) < n ? perm[first + int(tid)] : -1;
        xr[tid] = s >= 0 ? s / int(P.xdiv) : -1;
        yr[tid] = s;
    }
    gemm_body<T>(tab[tg.z].p, x, y, P, GemmTg{ws, xs, xr, yr}, GemmAt{tg.x * GM, tid, sg, lane, false});
}

// clang-format off
#define MOE_TAB_R(T, R)                                                                                           \
    template [[host_name("mvt_" #T "_r" #R)]] kernel void mvt<T, R>(device const float*, device float*,            \
        device const int*, device const ExpertAt*, constant MvParams&, uint2, uint, uint, uint);                \
    template [[host_name("mvgt_" #T "_r" #R)]] kernel void mvgt<T, R, 4>(device const float*, device float*,       \
        device const int*, device const int*, device const int*, device const ExpertAt*, constant MvParams&,    \
        uint2, uint, uint, uint);
#define MOE_TAB(T) MOE_TAB_R(T, 1) MOE_TAB_R(T, 4)                                                               \
    template [[host_name("gemmxt_" #T)]] kernel void gemmxt<T>(device const float*, device float*,                 \
        device const int*, device const int*, device const ExpertAt*, constant GemmParams&, uint3, uint, uint,    \
        uint);
// clang-format on
// The types of the Coder's routed experts
MOE_TAB(IQ2S)
MOE_TAB(IQ3XXS)
MOE_TAB(IQ3S)
MOE_TAB(IQ4NL)
MOE_TAB(IQ4XS)
MOE_TAB(Q20)

// An arena stop: the GPU tells the CPU that the layer's routing is written. The CPU polls this word in shared memory;
// it sees the write about 55 µs after the GPU, where the shared event's signaledValue took milliseconds
// (research/measurements/2026-11-11-mac-event.md, 2026-11-15-mac-coder-arena.md).
kernel void stop_flag(device atomic_uint* flag [[buffer(0)]], constant uint& value [[buffer(1)]]) {
    atomic_store_explicit(flag, value, memory_order_relaxed);
}
