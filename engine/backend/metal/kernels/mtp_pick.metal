// The pick of the MTP head's truncated LM head on the GPU, as the reference (engine/gpu/mega_op_mtp.h, op_mtp_pick):
// each threadgroup reduces its slice of the rows to {row, max, sum of exp(l - max), NaN count}; the threadgroup that
// finishes last merges all slices in index order (one dispatch instead of two: 33 against 41 us per pick, micro probe
// in research/measurements/2026-11-14-mac-mtp-gpu-pick.md), so q's bits never depend on which slice finished first.
// Logits are clamped (NaN = -FLT_MAX, +-inf = +-FLT_MAX), so the first valid row wins at the least; on a tie the
// smaller row wins (as the host's first-greater scan). Free rows (token < 0) are skipped. The host reads 4 floats and
// takes q = 1 / sum.

static inline float pick_clamp(float v) { return isnan(v) ? -FLT_MAX : clamp(v, -FLT_MAX, FLT_MAX); }

static inline void pick_better(thread float& bv, thread uint& br, float v, uint r) {
    if (v > bv || (v == bv && r < br)) bv = v, br = r;
}

// Best (value, row) of the 256 threads; every thread gets it
static inline void pick_reduce(thread float& bv, thread uint& br, threadgroup float* tv, threadgroup uint* tr, uint sg,
                               uint lane) {
    for (ushort off = 16; off > 0; off /= 2) {
        const float v = simd_shuffle_down(bv, off);
        const uint r = simd_shuffle_down(br, off);
        pick_better(bv, br, v, r);
    }
    if (lane == 0) tv[sg] = bv, tr[sg] = br;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bv = tv[0], br = tr[0];
    for (uint i = 1; i < 8; i++) pick_better(bv, br, tv[i], tr[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Row (as bits), max, sum of exp(l - max) over the valid rows of slice g, NaN logits of the slice; every thread gets
// them
struct PickTg {
    threadgroup float* tv;
    threadgroup float* tmp;
    threadgroup uint* tr;
};

static inline float4 pick_slice(device const float* lg, device const int* map, constant PickParams& P, uint g, PickTg t,
                                uint3 ids) {
    const uint tid = ids.x, sg = ids.y, lane = ids.z;
    threadgroup float* tv = t.tv;
    threadgroup float* tmp = t.tmp;
    threadgroup uint* tr = t.tr;
    const uint r0 = g * P.per, r1 = min(P.rows, r0 + P.per);
    float bv = -INFINITY, nan_n = 0.f;
    uint br = P.rows;
    for (uint r = r0 + tid; r < r1; r += 256) {
        if (map[r] < 0) continue;
        const float v = lg[r];
        nan_n += isnan(v) ? 1.f : 0.f;
        if (pick_clamp(v) > bv) bv = pick_clamp(v), br = r;
    }
    pick_reduce(bv, br, tv, tr, sg, lane);
    float s = 0.f;
    if (bv > -INFINITY)
        for (uint r = r0 + tid; r < r1; r += 256)
            if (map[r] >= 0) s += fast::exp(pick_clamp(lg[r]) - bv);
    s = tg_sum(s, tmp, tid, sg, lane);
    nan_n = tg_sum(nan_n, tmp, tid, sg, lane);
    return float4(as_type<float>(br), bv, s, nan_n);
}

// The slices merged by thread index (a fixed tree whatever the order they finished in)
static inline float4 pick_merge(coherent(device) device const float* part, constant PickParams& P, PickTg t,
                                uint3 ids) {
    const uint tid = ids.x, sg = ids.y, lane = ids.z;
    threadgroup float* tv = t.tv;
    threadgroup float* tmp = t.tmp;
    threadgroup uint* tr = t.tr;
    float bv = -INFINITY, pv = -INFINITY, ps = 0.f, nan_n = 0.f;
    uint br = P.rows;
    if (tid < P.groups) {
        pv = part[4 * tid + 1], ps = part[4 * tid + 2], nan_n = part[4 * tid + 3];
        if (pv > -INFINITY) bv = pv, br = as_type<uint>(part[4 * tid]);
    }
    pick_reduce(bv, br, tv, tr, sg, lane);
    float s = pv > -INFINITY ? ps * fast::exp(pv - bv) : 0.f;
    s = tg_sum(s, tmp, tid, sg, lane);
    nan_n = tg_sum(nan_n, tmp, tid, sg, lane);
    return float4(as_type<float>(br), bv, s, nan_n);
}

// out: row (as bits; rows when no row is valid), max, sum, NaN logits. done counts the finished slices; the last
// threadgroup sets it back to 0 for the next pass. Fences: a slice's part is visible on the device before its count
// (MSL 3.2 orders fences only as seq_cst or relaxed). part is coherent(device): without it the last threadgroup read
// stale slices in the engine (42 of 200 picks wrong, a different hash each run), though not in an isolated probe.
kernel void mtp_pick(device const float* lg [[buffer(0)]], device const int* map [[buffer(1)]],
                     coherent(device) device float* part [[buffer(2)]], device float* out [[buffer(3)]],
                     device atomic_uint* done [[buffer(4)]], constant PickParams& P [[buffer(5)]],
                     uint g [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float tv[8], tmp[8];
    threadgroup uint tr[8];
    // Shared by the threadgroup: thread 0 writes it, every thread reads it after the barrier. The compiler sees a path
    // where a thread other than 0 reads it unwritten; an initialisation by every thread would race with thread 0's
    // store unless another barrier followed
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wsometimes-uninitialized"
    threadgroup bool last;
    const PickTg t{tv, tmp, tr};
    const uint3 ids(tid, sg, lane);
    const float4 mine = pick_slice(lg, map, P, g, t, ids);
    if (tid == 0) {
        part[4 * g] = mine.x, part[4 * g + 1] = mine.y, part[4 * g + 2] = mine.z, part[4 * g + 3] = mine.w;
        atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
        last = atomic_fetch_add_explicit(done, 1u, memory_order_relaxed) == P.groups - 1;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (!last) return;
#pragma clang diagnostic pop
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const float4 all = pick_merge(part, P, t, ids);
    if (tid == 0) {
        out[0] = all.x, out[1] = all.y, out[2] = all.z, out[3] = all.w;
        atomic_store_explicit(done, 0u, memory_order_relaxed);
    }
}
