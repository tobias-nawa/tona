// Gated DeltaNet of nb consecutive tokens, as engine/cpu/ref_qwen4exp.cpp Qwen4ExpOracle::deltanet once per token, in
// order. Needs ops.metal (sigmoidf, siluf, softplusf, tg_sum). Head dim 128 (checked on the host).

// Causal depthwise conv over [kconv - 1 old inputs; current] with SiLU; then the window moves by one; token by token.
// Weight (k, c) at c * kconv + k; window [kconv - 1][ch], oldest first; qkv and out [nb][ch].
kernel void dn_conv(device const float* qkv [[buffer(0)]], device float* win [[buffer(1)]],
                    device const float* w [[buffer(2)]], device float* out [[buffer(3)]],
                    constant DnParams& P [[buffer(4)]], uint c [[thread_position_in_grid]]) {
    if (c >= P.ch) return;
    const uint K = P.kconv;
    for (uint t = 0; t < P.nb; ++t) {
        const float in = qkv[t * P.ch + c];
        float acc = 0.f;
        for (uint k = 0; k + 1 < K; ++k) acc += w[c * K + k] * win[k * P.ch + c];
        acc += w[c * K + K - 1] * in;
        out[t * P.ch + c] = siluf(acc);
        for (uint k = 0; k + 2 < K; ++k) win[k * P.ch + c] = win[(k + 1) * P.ch + c];
        win[(K - 2) * P.ch + c] = in;
    }
}

// L2 norm per head of q and k (the first 2 nk heads of each token's conv output): rms_norm(eps / 128) / sqrt(128)
kernel void dn_l2(device float* x [[buffer(0)]], constant DnParams& P [[buffer(1)]],
                  uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                  uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    device float* xh = x + (head / (2 * P.nk)) * P.ch + (head % (2 * P.nk)) * 128;
    const float v = tid < 128 ? xh[tid] : 0.f;
    const float ss = tg_sum(v * v, tmp, tid, sg, lane);
    const float scale = 1.0f / sqrt(ss / 128.0f + P.eps / 128.0f);
    if (tid < 128) xh[tid] = v * scale * (1.0f / sqrt(128.0f));
}

// Per V head j and state row vi, for each token t in order: s *= alpha; d = (v - s.k) * beta; s += k d;
// o = (s.q) / sqrt(128). One simdgroup per 8 rows (4 values per lane each), the rows in registers across the tokens.
// The 8 rows go through each token together: they share k, q, alpha and beta, and their independent sums hide each
// other's latency; every row's operations keep their order, so the result is the same as row by row.
// alpha = exp(softplus(a + dt) * A), beta = sigmoid(b); a, b [nb][nv], o [nb][nv][128].
kernel void dn_step(device float* state [[buffer(0)]], device const float* conv [[buffer(1)]],
                    device const float* a [[buffer(2)]], device const float* b [[buffer(3)]],
                    device const float* dt [[buffer(4)]], device const float* A [[buffer(5)]],
                    device float* o [[buffer(6)]], constant DnParams& P [[buffer(7)]],
                    uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint j = tg.y, kh = j % P.nk;
    const uint v0 = (tg.x * nsg + sg) * 8;
    device float4* rows = (device float4*)(state + (ulong(j) * 128 + v0) * 128) + lane;
    float4 s[8];
    for (uint r = 0; r < 8; ++r) s[r] = rows[r * 32];
    for (uint t = 0; t < P.nb; ++t) {
        device const float* ct = conv + t * P.ch;
        const float alpha = exp(softplusf(a[t * P.nv + j] + dt[j]) * A[j]);
        const float beta = sigmoidf(b[t * P.nv + j]);
        const float4 kk = ((device const float4*)(ct + P.nk * 128 + kh * 128))[lane];
        const float4 qq = ((device const float4*)(ct + kh * 128))[lane];
        device const float* v = ct + 2 * P.nk * 128 + j * 128 + v0;
        float ov[8];
        for (uint r = 0; r < 8; ++r) {
            s[r] *= alpha;
            const float sk = simd_sum(dot(s[r], kk));
            s[r] += kk * ((v[r] - sk) * beta);
            ov[r] = simd_sum(dot(s[r], qq));
        }
        if (lane < 8) o[(t * P.nv + j) * 128 + v0 + lane] = ov[lane] * (1.0f / sqrt(128.0f));
    }
    for (uint r = 0; r < 8; ++r) rows[r * 32] = s[r];
}

// Per V head of every token (o and z [nb][nv][128]): RMS norm with the weight as it is (not 1 + w), then the sigmoid
// gate z
kernel void dn_out(device float* o [[buffer(0)]], device const float* z [[buffer(1)]],
                   device const float* w [[buffer(2)]], constant DnParams& P [[buffer(3)]],
                   uint head [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                   uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    device float* oh = o + head * 128;
    const float v = tid < 128 ? oh[tid] : 0.f;
    const float ss = tg_sum(v * v, tmp, tid, sg, lane);
    const float scale = 1.0f / sqrt(ss / 128.0f + P.eps);
    if (tid < 128) oh[tid] = v * scale * w[tid] * sigmoidf(z[head * 128 + tid]);
}
