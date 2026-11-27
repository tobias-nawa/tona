// Expert mat-vec kernels for the Coder's expert types, first cut for timing (M0), not the backend's kernels.
// The host prepends the IQ tables from engine/model/quant_tables.h (converted to MSL) and decode.metal (the group
// decoders and rows_dot) before compiling.
// Layout: one simdgroup per R output rows (gate/up: the same rows of both, fused with SiLU); lanes stride over the
// groups of 8 of a row; B tokens share every decoded group, R rows share every input load. R = 1 or 4.

struct Params {
    uint rows;       // output rows per expert
    uint k;          // inner dimension
    uint row_bytes;  // bytes per weight row
    uint n_sel;      // selected experts
    uint x_stride;   // floats between the inputs of two experts (0: one input for all)
};

// h[e][b][r] = silu(gate_r . x_b) * (up_r . x_b); tg.y = selected expert; each simdgroup R consecutive rows
template <typename T, int B, int R>
kernel void gate_up(device const uchar* gate [[buffer(0)]], device const uchar* up [[buffer(1)]],
                    device const float* x [[buffer(2)]], device float* h [[buffer(3)]],
                    device const uint* ids [[buffer(4)]], constant Params& P [[buffer(5)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r0 = (tg.x * nsg + sg) * R;
    if (r0 >= P.rows) return;
    ulong off = (ulong(ids[tg.y]) * P.rows + r0) * P.row_bytes;
    float a[R][B], c[R][B];
    rows_dot<T, B, R>(gate + off, P.row_bytes, x, P.k, lane, a);
    rows_dot<T, B, R>(up + off, P.row_bytes, x, P.k, lane, c);
    if (lane == 0)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < B; ++b) h[(tg.y * B + b) * P.rows + r0 + r] = a[r][b] / (1.f + exp(-a[r][b])) * c[r][b];
}

// y[e][b][r] = down_r . h[e][b]
template <typename T, int B, int R>
kernel void down(device const uchar* w [[buffer(0)]], device const float* x [[buffer(2)]],
                 device float* y [[buffer(3)]], device const uint* ids [[buffer(4)]], constant Params& P [[buffer(5)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                 uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    uint r0 = (tg.x * nsg + sg) * R;
    if (r0 >= P.rows) return;
    ulong off = (ulong(ids[tg.y]) * P.rows + r0) * P.row_bytes;
    float a[R][B];
    rows_dot<T, B, R>(w + off, P.row_bytes, x + tg.y * P.x_stride, P.k, lane, a);
    if (lane == 0)
        for (int r = 0; r < R; ++r)
            for (int b = 0; b < B; ++b) y[(tg.y * B + b) * P.rows + r0 + r] = a[r][b];
}

// clang-format off
#define GU(T, B, R)                                                                                                 \
    template [[host_name("gate_up_" #T "_" #B "_r" #R)]] kernel void gate_up<T, B, R>(                              \
        device const uchar*, device const uchar*, device const float*, device float*, device const uint*,          \
        constant Params&, uint2, uint, uint, uint);
// clang-format on
// clang-format off
#define DN(T, B, R)                                                                                                 \
    template [[host_name("down_" #T "_" #B "_r" #R)]] kernel void down<T, B, R>(device const uchar*, device const float*, \
                                                                      device float*, device const uint*,            \
                                                                      constant Params&, uint2, uint, uint, uint);
// clang-format on
#define ALL_B(M, T) M(T, 1, 1) M(T, 2, 1) M(T, 3, 1) M(T, 4, 1) M(T, 1, 4) M(T, 2, 4) M(T, 3, 4) M(T, 4, 4)
ALL_B(GU, IQ2S)
ALL_B(GU, IQ3XXS)
ALL_B(GU, IQ3S)
ALL_B(GU, IQ4XS)
ALL_B(DN, IQ4NL)
ALL_B(DN, Q20)
