// n-gram embedding (PLE) of nb consecutive tokens after the lookup and the key/value mat-vecs, as
// engine/cpu/ref_qwen4exp.cpp Qwen4ExpOracle::ple_gated and Qwen4ExpOracle::ple. Needs ops.metal.

// Per stream c: gate = sigmoid(sgn(s) sqrt(max(|s|, 1e-6))) with s = norm(key_c) . norm(R_c) / sqrt(d);
// gated = val * gate; normed = norm(gated) * w_conv; causal depthwise conv with dilation over the ring;
// R_c += gated + silu(conv); normed replaces the oldest ring entry. Token by token (R, key [nb][hc][d], val [nb][d]):
// token t reads the ring entries the tokens before it wrote. One threadgroup of 256 per stream; a thread keeps its
// channels, so it reads only ring values it wrote itself or that an earlier pass wrote.
kernel void ple_mix(device float* R [[buffer(0)]], device const float* key [[buffer(1)]],
                    device const float* val [[buffer(2)]], device const float* w_key [[buffer(3)]],
                    device const float* w_query [[buffer(4)]], device const float* w_conv [[buffer(5)]],
                    device const float* conv [[buffer(6)]], device float* ring [[buffer(7)]],
                    constant PleParams& P [[buffer(8)]], uint c [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                    uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    const uint d = P.d, hc_dim = P.d * P.hc, K = P.kconv;
    for (uint t = 0; t < P.nb; ++t, R += hc_dim, key += hc_dim, val += d) {
        const uint start = (P.start + t) % P.hist;
        float sk = 0.f, sq = 0.f, kq = 0.f, vv = 0.f;
        for (uint i = tid; i < d; i += 256) {
            const uint ch = c * d + i;
            sk += key[ch] * key[ch];
            sq += R[ch] * R[ch];
            kq += key[ch] * w_key[ch] * R[ch] * w_query[ch];
            vv += val[i] * val[i];
        }
        sk = tg_sum(sk, tmp, tid, sg, lane);
        sq = tg_sum(sq, tmp, tid, sg, lane);
        kq = tg_sum(kq, tmp, tid, sg, lane);
        vv = tg_sum(vv, tmp, tid, sg, lane);
        const float s = kq / sqrt(sk / float(d) + P.eps) / sqrt(sq / float(d) + P.eps) / sqrt(float(d));
        const float gate = sigmoidf(float((s > 0) - (s < 0)) * sqrt(max(fabs(s), 1e-6f)));
        const float nscale = 1.0f / sqrt(vv * gate * gate / float(d) + P.eps);
        for (uint i = tid; i < d; i += 256) {
            const uint ch = c * d + i;
            const float gated = val[i] * gate;
            const float normed = gated * nscale * w_conv[ch];
            float acc = 0.f;
            for (uint k = 0; k < K; ++k) {
                const uint back = (K - 1 - k) * P.dil;
                const float x = back == 0 ? normed : ring[((start + P.hist - back) % P.hist) * hc_dim + ch];
                acc += conv[ch * K + k] * x;
            }
            R[ch] += gated + siluf(acc);
            ring[start * hc_dim + ch] = normed;
        }
    }
}
