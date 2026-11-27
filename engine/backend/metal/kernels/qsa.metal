// QSA layer of one token, as engine/cpu/ref_qwen4exp.cpp Qwen4ExpOracle::qsa: the indexer pools raw keys per block,
// scores the blocks, and attention runs over the top blocks plus the block in progress. Needs ops.metal.

static inline float tg_max(float v, threadgroup float* tmp, uint sg, uint lane) {
    v = simd_max(v);
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = tmp[0];
    for (uint i = 1; i < 8; ++i) m = max(m, tmp[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

// Block b0 / blk is complete: mean of its raw keys, RMS norm with the indexer's k_norm, RoPE at the block's first
// position (cs). One threadgroup of 256.
kernel void idx_pool(device const float* raw [[buffer(0)]], device float* pool [[buffer(1)]],
                     device const float* w [[buffer(2)]], device const float* cs [[buffer(3)]],
                     constant PoolParams& P [[buffer(4)]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    threadgroup float v[512];
    for (uint i = tid; i < P.idim; i += 256) {
        float s = 0.f;
        for (uint m = 0; m < P.blk; ++m) s += raw[(P.b0 + m) * P.idim + i];
        v[i] = s * (1.0f / float(P.blk));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float ss = 0.f;
    for (uint i = tid; i < P.idim; i += 256) ss += v[i] * v[i];
    ss = tg_sum(ss, tmp, tid, sg, lane);
    const float scale = 1.0f / sqrt(ss / float(P.idim) + P.eps);
    for (uint i = tid; i < P.idim; i += 256) v[i] = v[i] * scale * w[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint h = P.rope_dims / 2;
    if (tid < h) {
        const float c = cs[tid], s = cs[h + tid], x0 = v[tid], x1 = v[tid + h];
        v[tid] = x0 * c - x1 * s;
        v[tid + h] = x0 * s + x1 * c;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device float* out = pool + (P.b0 / P.blk) * P.idim;
    for (uint i = tid; i < P.idim; i += 256) out[i] = v[i];
}

// score[b] = sum over heads of ReLU(q_h . k_b) / sqrt(idim); NaN becomes -FLT_MAX (backend/nan_guard.h). One
// simdgroup per block.
kernel void idx_score(device const float* q [[buffer(0)]], device const float* pool [[buffer(1)]],
                      device float* score [[buffer(2)]], constant ScoreParams& P [[buffer(3)]],
                      uint tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint nsg [[simdgroups_per_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    const uint b = tg * nsg + sg;
    if (b >= P.n_pool) return;
    float s = 0.f;
    for (uint h = 0; h < P.heads; ++h) {
        float d = 0.f;
        for (uint i = lane; i < P.idim; i += 32) d += q[h * P.idim + i] * pool[b * P.idim + i];
        s += max(simd_sum(d), 0.0f);
    }
    s /= sqrt(float(P.idim));
    if (lane == 0) score[b] = isnan(s) ? -FLT_MAX : clamp(s, -FLT_MAX, FLT_MAX);
}

// Decode attention, split over the cells (flash-decoding): threadgroup (part p, Q head h) takes cells
// p * ATTN_PART ... of the token's list and writes its unnormalized output, its maximum and its exp sum to
// part[h][p] = [hd | max | sum]; attend_join combines them. The split depends on the cell count alone (the position),
// never on the width of the pass. q holds per head [q hd | gate hd]; K and V caches [pos][nkv][hd]; hd a multiple of
// 128. 256 threads: simdgroup g scores the cells g, g + 8, ... of the part, thread i accumulates V column i.
kernel void attend_part(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
                        device const float* vc [[buffer(2)]], device const int* cells [[buffer(3)]],
                        device float* part [[buffer(4)]], constant AttnParams& P [[buffer(5)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tmp[8];
    threadgroup float sc[ATTN_PART];
    const uint h = tg.y, c0 = tg.x * ATTN_PART, n = min(ATTN_PART, P.ncells - c0);
    const uint kvh = h / (P.nh / P.nkv);
    device const float4* qh = (device const float4*)(q + h * 2 * P.hd);
    for (uint c = sg; c < n; c += 8) {
        device const float4* k = (device const float4*)(kc + (ulong(cells[c0 + c]) * P.nkv + kvh) * P.hd);
        float d = 0.f;
        for (uint i = lane; i < P.hd / 4; i += 32) d += dot(qh[i], k[i]);
        d = simd_sum(d);
        if (lane == 0) sc[c] = d * P.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float mx = tg_max(tid < n ? sc[tid] : -INFINITY, tmp, sg, lane);
    const float e = tid < n ? exp(sc[tid] - mx) : 0.f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < n) sc[tid] = e;
    const float sum = tg_sum(e, tmp, tid, sg, lane);  // its barriers also publish sc
    device float* out = part + (ulong(h) * ATTN_PARTS + tg.x) * (P.hd + 2);
    for (uint i = tid; i < P.hd; i += 256) {
        float acc = 0.f;
        for (uint c = 0; c < n; ++c) acc += sc[c] * vc[(ulong(cells[c0 + c]) * P.nkv + kvh) * P.hd + i];
        out[i] = acc;
    }
    if (tid == 0) out[P.hd] = mx, out[P.hd + 1] = sum;
}

// The parts of Q head h, in part order, scaled to the common maximum, times sigmoid(gate). One threadgroup of 256 per
// head.
kernel void attend_join(device const float* q [[buffer(0)]], device const float* part [[buffer(1)]],
                        device float* o [[buffer(2)]], constant AttnParams& P [[buffer(3)]],
                        uint h [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
    const uint parts = (P.ncells + ATTN_PART - 1) / ATTN_PART;
    device const float* ph = part + ulong(h) * ATTN_PARTS * (P.hd + 2);
    float mx = -INFINITY;
    for (uint p = 0; p < parts; ++p) mx = max(mx, ph[p * (P.hd + 2) + P.hd]);
    float sum = 0.f;
    for (uint p = 0; p < parts; ++p) sum += exp(ph[p * (P.hd + 2) + P.hd] - mx) * ph[p * (P.hd + 2) + P.hd + 1];
    device const float* gate = q + h * 2 * P.hd + P.hd;
    for (uint i = tid; i < P.hd; i += 256) {
        float acc = 0.f;
        for (uint p = 0; p < parts; ++p) acc += exp(ph[p * (P.hd + 2) + P.hd] - mx) * ph[p * (P.hd + 2) + i];
        o[h * P.hd + i] = acc / sum * sigmoidf(gate[i]);
    }
}
