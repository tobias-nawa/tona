// Small ops of the pass, as engine/cpu/ref_qwen4exp.cpp computes them (sums in float instead of double). One
// threadgroup of 256 threads per row where a row needs a reduction.

static inline float sigmoidf(float x) { return 1.0f / (1.0f + exp(-x)); }
static inline float siluf(float x) { return x / (1.0f + exp(-x)); }
static inline float softplusf(float x) { return x > 20.0f ? x : log(1.0f + exp(x)); }

// Sum over the 256 threads of a threadgroup; every thread gets the result
static inline float tg_sum(float v, threadgroup float* tmp, uint tid, uint sg, uint lane) {
    v = simd_sum(v);
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 8; ++i) s += tmp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// y_r = x_r / sqrt(mean(x_r^2) + eps) * w_r * post (w optional), rows of n at strides in_stride / out_stride
kernel void rms_rows(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],
                     device const float* w [[buffer(2)]], constant NormParams& P [[buffer(3)]],
                     uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    device const float* xr = x + row * P.in_stride;
    device float* yr = y + row * P.out_stride;
    float ss = 0.f;
    for (uint i = tid; i < P.n; i += 256) ss += xr[i] * xr[i];
    ss = tg_sum(ss, tmp, tid, sg, lane);
    const float scale = 1.0f / sqrt(ss / float(P.n) + P.eps);
    for (uint i = tid; i < P.n; i += 256)
        yr[i] = xr[i] * scale * (P.use_w ? w[(row % P.w_rows) * P.w_stride + i] : 1.0f) * P.post;
}

// x <- silu(x * s)
kernel void silu_scale(device float* x [[buffer(0)]], constant ScaleParams& P [[buffer(1)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < P.n) x[i] = siluf(x[i] * P.s);
}

// g <- silu(g) * u
kernel void swiglu(device float* g [[buffer(0)]], device const float* u [[buffer(1)]], constant uint& n [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    if (i < n) g[i] = siluf(g[i]) * u[i];
}

// Gated residual read per token b: mixed[b][i] = mean over streams c of xn[b][c][i] * sigmoid(gate[b][c][i])
kernel void hc_mixed(device const float* xn [[buffer(0)]], device const float* gate [[buffer(1)]],
                     device float* mixed [[buffer(2)]], constant HcParams& P [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    if (i >= P.d * P.nb) return;
    const uint b = i / P.d, j = i % P.d, base = b * P.hc * P.d;
    float m = 0.f;
    for (uint c = 0; c < P.hc; ++c) m += xn[base + c * P.d + j] * sigmoidf(gate[base + c * P.d + j]);
    mixed[i] = m / float(P.hc);
}

// Gated residual write per token b: R[b][c][i] += y[b][i] * 2 sigmoid(inject[b][c] / hc)
kernel void hc_combine(device float* R [[buffer(0)]], device const float* y [[buffer(1)]],
                       device const float* inject [[buffer(2)]], constant HcParams& P [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= P.d * P.hc * P.nb) return;
    const uint b = i / (P.d * P.hc), c = i / P.d % P.hc;
    R[i] += y[b * P.d + i % P.d] * 2.0f * sigmoidf(inject[b * P.hc + c] / float(P.hc));
}

// Per head: RMS norm with a weight shared by the heads, then NeoX RoPE on the first rope_dims dimensions with the
// cos/sin of the position (host table, double precision, YaRN applied). Heads of dim at head_stride.
kernel void norm_rope(device float* x [[buffer(0)]], device const float* w [[buffer(1)]],
                      device const float* cs [[buffer(2)]], constant NormRopeParams& P [[buffer(3)]],
                      uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    const uint t = head / P.heads;
    device float* xh = x + t * P.token_stride + (head % P.heads) * P.head_stride;
    cs += t * P.rope_dims;
    float ss = 0.f;
    for (uint i = tid; i < P.dim; i += 256) ss += xh[i] * xh[i];
    ss = tg_sum(ss, tmp, tid, sg, lane);
    const float scale = 1.0f / sqrt(ss / float(P.dim) + P.eps);
    for (uint i = tid; i < P.dim; i += 256) xh[i] = xh[i] * scale * w[i];
    threadgroup_barrier(mem_flags::mem_device);
    const uint half_dims = P.rope_dims / 2;
    if (tid < half_dims) {
        const float c = cs[tid], s = cs[half_dims + tid];
        const float x0 = xh[tid], x1 = xh[tid + half_dims];
        xh[tid] = x0 * c - x1 * s;
        xh[tid + half_dims] = x0 * s + x1 * c;
    }
}
