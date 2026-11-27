// Dense mat-vec for 4 to 8 tokens (P.tokens) on the 8x8 simdgroup matrix units, half inputs and weights, float
// accumulators. Not bit-identical with the f32 kernels, so only for chunks of 4 or more tokens and behind
// TONA_METAL_MMA and TONA_METAL_MMA_WIDE (pass.mm, pass.h); the 1-token kernel stays f32 (maintainer decision
// 2026-11-12, quality check in research/measurements/). Output column j depends on input column j only, so a token's
// row is the same whatever the other columns hold: the 8-token dispatch decodes the weights once where two 4-token
// dispatches decoded them twice (research/measurements/2026-11-12-mac-w8-mma.md). Lane l holds elements (fm, fn) and
// (fm, fn + 1) of each 8x8 fragment (ggml-metal's mul_mv_mma lane layout). K is permuted inside each 32-weight chunk:
// fragment column q of step kk is weight (q / 2) * 8 + 2 * kk + q % 2, so a lane's 8 decoded weights (row fm, group fn
// / 2 of the chunk) fill its slots of the 4 steps; the inputs follow the same permutation. A threadgroup takes 8 rows;
// its 8 simdgroups split K (chunk c to simdgroup c % 8) and add their tiles in simdgroup order. With TONA_Q6_K_T a
// lane's group is entry 32 c + l of the tile: the 32 lanes read consecutive words
// (research/measurements/2026-11-12-mac-dense-mma-probes.md).
static inline short mma_fm(ushort l) { return ((l / 4) & 4) + ((l / 2) % 4); }
static inline short mma_fn(ushort l) { return ((l / 4) & 2) * 2 + (l % 2) * 2; }

constexpr constant int kMmaSg = 8;

template <typename T>
kernel void mvm(device const uchar* w [[buffer(0)]], device const float* x [[buffer(1)]], device float* y [[buffer(2)]],
                device const int* ids [[buffer(3)]], constant MvParams& P [[buffer(4)]],
                uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                uint lane_ [[thread_index_in_simdgroup]]) {
    const uint B = P.tokens;
    threadgroup float red[kMmaSg][64];
    const ushort lane = ushort(lane_);
    const uint r0 = tg * 8;
    const short fm = mma_fm(lane), fn = mma_fn(lane);
    // columns fn and fn + 1 of the input tile. Output column j depends on input column j only and columns past B are
    // never stored, so those lanes read column B - 1 again instead of being zeroed: selecting zeros cost 11 of 44 ms
    // (research/measurements/2026-11-12-mac-q6k-tile.md)
    device const float* x0 = x + min(uint(fn), B - 1) * P.k + (fm / 2) * 8 + fm % 2;
    device const float* x1 = x + min(uint(fn) + 1, B - 1) * P.k + (fm / 2) * 8 + fm % 2;
    simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    for (uint c = sg; c < P.k / 32; c += kMmaSg) {
        float v[8];
        RowAt<T>::dec(w, r0 + fm, P.row_bytes, c * 4 + fn / 2, v);
        for (int kk = 0; kk < 4; ++kk) {
            simdgroup_half8x8 a, b;
            b.thread_elements()[0] = half(x0[c * 32 + 2 * kk]), b.thread_elements()[1] = half(x1[c * 32 + 2 * kk]);
            a.thread_elements()[0] = half(v[2 * kk]), a.thread_elements()[1] = half(v[2 * kk + 1]);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
    }
    simdgroup_store(acc, red[sg], 8);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = sg * 32 + lane; i < 64; i += kMmaSg * 32) {
        float s = 0.f;
        for (int g = 0; g < kMmaSg; ++g) s += red[g][i];
        if (i % 8 < B) y[(i % 8) * P.rows + r0 + i / 8] = s;
    }
}

// clang-format off
#define MVM(T) \
    template [[host_name("mvm_" #T)]] kernel void mvm<T>(device const uchar*, device const float*, device float*, \
        device const int*, constant MvParams&, uint, uint, uint);
MVM(F16)
MVM(BF16)
MVM(Q80)
MVM(Q4K)
MVM(Q5K)
MVM(Q6K)
MVM(Q6KT)
MVM(IQ4XS)
MVM(IQ4NL)
// clang-format on
