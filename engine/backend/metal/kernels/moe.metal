// Router and expert sum of one token, as engine/cpu/ref_qwen4exp.cpp Qwen4ExpOracle::moe. Needs ops.metal.

// Reductions over a threadgroup of 1024 (32 simdgroups); every thread gets the result
static inline float tg1024_max(float v, threadgroup float* tmp, uint sg, uint lane) {
    v = simd_max(v);
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = tmp[0];
    for (uint i = 1; i < 32; ++i) m = max(m, tmp[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return m;
}

static inline float tg1024_sum(float v, threadgroup float* tmp, uint sg, uint lane) {
    v = simd_sum(v);
    if (lane == 0) tmp[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float s = 0.f;
    for (uint i = 0; i < 32; ++i) s += tmp[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return s;
}

// Router logit of expert e; NaN becomes -FLT_MAX and is counted, +-inf are clamped (backend/nan_guard.h)
static inline float clean_logit(device const float* logits, uint e, uint n, device atomic_uint* nan_count) {
    if (e >= n) return -INFINITY;
    const float l = logits[e];
    if (isnan(l)) atomic_fetch_add_explicit(nan_count, 1u, memory_order_relaxed);
    return isnan(l) ? -FLT_MAX : clamp(l, -FLT_MAX, FLT_MAX);
}

// Softmax over the router logits, top n_used by rank (probability descending, expert ascending on ties), weights
// renormalized over the chosen. One threadgroup of 1024 per token t (logits [nb][n_expert], ids and weights
// [nb][n_used]), n_expert <= 1024.
kernel void route(device const float* logits [[buffer(0)]], device int* ids [[buffer(1)]],
                  device float* weights [[buffer(2)]], device atomic_uint* nan_count [[buffer(3)]],
                  constant RouteParams& P [[buffer(4)]], uint t [[threadgroup_position_in_grid]],
                  uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                  uint lane [[thread_index_in_simdgroup]]) {
    logits += t * P.n_expert, ids += t * P.n_used, weights += t * P.n_used;
    threadgroup float tmp[32];
    threadgroup float p[1024];
    threadgroup float chosen[1024];
    const float l = clean_logit(logits, tid, P.n_expert, nan_count);
    const float m = tg1024_max(l, tmp, sg, lane);
    const float e = tid < P.n_expert ? exp(l - m) : 0.f;
    const float s = tg1024_sum(e, tmp, sg, lane);
    if (tid < P.n_expert) p[tid] = e / s;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P.n_expert) {
        const float pe = p[tid];
        uint rank = 0;
        for (uint j = 0; j < P.n_expert; ++j) rank += p[j] > pe || (p[j] == pe && j < tid);
        if (rank < P.n_used) ids[rank] = int(tid), chosen[rank] = pe;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < P.n_used) {
        float ws = 0.f;
        for (uint r = 0; r < P.n_used; ++r) ws += chosen[r];
        weights[tid] = chosen[tid] / ws;
    }
}

// The slots of a pass grouped by expert: off[e] ... off[e + 1] - 1 index perm, which lists expert e's slots in slot
// order; glist lists the experts with slots in expert order, then -1 up to P.slots. With P.split, an expert with c
// slots has ceil(c / split) entries e + (chunk << 16), one mvg threadgroup row each. One thread per expert counts and
// lists its own slots, so the result never depends on timing; ~2 reads of the id list per thread (5120 ids for a
// 512-token chunk with 10 experts per token).
kernel void moe_group(device const int* ids [[buffer(0)]], device int* off [[buffer(1)]],
                      device int* perm [[buffer(2)]], device int* glist [[buffer(3)]],
                      constant GroupParams& P [[buffer(4)]], uint e [[thread_index_in_threadgroup]]) {
    threadgroup int count[1024];
    int n = 0;
    if (e < P.n_expert)
        for (uint s = 0; s < P.slots; ++s) n += ids[s] == int(e);
    count[e] = n;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e == 0) {
        int at = 0;
        uint g = 0;
        for (uint j = 0; j < P.n_expert; ++j) {
            const int c = count[j];
            count[j] = at, off[j] = at;
            const int chunks = P.split ? (c + int(P.split) - 1) / int(P.split) : min(c, 1);
            for (int k = 0; k < chunks; ++k) glist[g++] = int(j) + (k << 16);
            at += c;
        }
        off[P.n_expert] = at;
        for (; g < P.slots; ++g) glist[g] = -1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= P.n_expert) return;
    int at = count[e];
    for (uint s = 0; s < P.slots; ++s)
        if (ids[s] == int(e)) perm[at++] = int(s);
}

// y[t] = sum over the chosen of weight * expert output, plus the shared expert times sigmoid of its gate
kernel void moe_sum(device const float* out [[buffer(0)]], device const float* weights [[buffer(1)]],
                    device const float* shared_out [[buffer(2)]], device const float* shared_gate [[buffer(3)]],
                    device float* y [[buffer(4)]], constant SumParams& P [[buffer(5)]],
                    uint i [[thread_position_in_grid]]) {
    if (i >= P.d * P.nb) return;
    const uint t = i / P.d, j = i % P.d;
    float acc = 0.f;
    for (uint s = 0; s < P.n_used; ++s) acc += weights[t * P.n_used + s] * out[(t * P.n_used + s) * P.d + j];
    y[i] = acc + shared_out[i] * sigmoidf(shared_gate[t]);
}
