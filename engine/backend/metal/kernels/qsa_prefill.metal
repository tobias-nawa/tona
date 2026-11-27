// QSA of a prefill chunk: the indexer scores of all tokens as matrix products, a top-k select per token in O(n) into
// a block bitmap, and attention per tile of 8 tokens. Each token's result depends on its own selection only, never on
// which tokens share its tile or where the chunk starts (docs/use/prompt-cache.md): matrix products reduce every output
// element alone, and attention walks windows of 8 cells aligned to absolute positions, in ascending order; a window a
// token does not use leaves its running sums unchanged bit for bit. Needs ops.metal and qsa.metal.

// score[t][b] = sum over heads of ReLU(q_th . pool_b) / sqrt(idim) for b < n_pool of token t; NaN becomes -FLT_MAX.
// Threadgroup: 8 tokens × 32 blocks, simdgroup s takes blocks 8s ... 8s + 7.
kernel void idx_score_t(device const float* q [[buffer(0)]], device const float* pool [[buffer(1)]],
                        device float* score [[buffer(2)]], constant ScoreTParams& P [[buffer(3)]],
                        uint2 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],
                        uint lane [[thread_index_in_simdgroup]]) {
    threadgroup float st[4][64];
    const uint t0 = tg.y * 8, b0 = tg.x * 32 + sg * 8;
    float sum[2] = {0.f, 0.f};
    for (uint h = 0; h < P.heads; ++h) {
        simdgroup_float8x8 acc = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        for (uint k = 0; k < P.idim; k += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, q + ulong(t0) * P.q_row + h * P.idim + k, P.q_row);
            simdgroup_load(b, pool + ulong(b0) * P.idim + k, P.idim, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(acc, a, b, acc);
        }
        simdgroup_store(acc, st[sg], 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint e = 0; e < 2; ++e) sum[e] += max(st[sg][lane + 32 * e], 0.0f);
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint e = 0; e < 2; ++e) {
        const uint i = lane + 32 * e, t = t0 + i / 8, b = b0 + i % 8;
        if (t >= P.nb || b >= (P.pos0 + t + 1) / P.blk) continue;
        const float s = sum[e] / sqrt(float(P.idim));
        score[ulong(t) * P.score_row + b] = isnan(s) ? -FLT_MAX : clamp(s, -FLT_MAX, FLT_MAX);
    }
}

// Orderable key of a score: larger score, larger key
static inline uint score_key(float s) {
    const uint u = as_type<uint>(s);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Radix step over the keys that match `prefix` in the bits above `shift + 8`: the digit d at `shift` such that the
// keys with a larger digit number fewer than `need` and those with digit >= d at least `need`
struct Scores {
    device const float* s;
    uint n;
};

static inline uint select_digit(Scores sc, uint prefix, uint shift, thread uint& need, threadgroup atomic_uint* hist,
                                uint tid) {
    device const float* s = sc.s;
    const uint n = sc.n;
    for (uint i = tid; i < 256; i += 1024) atomic_store_explicit(&hist[i], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint hi = shift + 8;
    for (uint b = tid; b < n; b += 1024) {
        const uint k = score_key(s[b]);
        if (hi >= 32 || (k >> hi) == (prefix >> hi))
            atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint d = 255, above = 0;
    for (;; --d) {
        const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
        if (above + c >= need || d == 0) break;
        above += c;
    }
    need -= above;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return d;
}

// Bitmap of the n_top best blocks of token t (score descending, block ascending on ties, as idx_top), over the
// n_pool blocks complete at its position; all of them if n_pool <= n_top. One threadgroup of 1024 per token; thread i
// owns the bitmap words i, i + 1024, ... in a contiguous range so the ties go to the smaller blocks.
kernel void idx_select(device const float* score [[buffer(0)]], device uint* bits [[buffer(1)]],
                       constant SelectParams& P [[buffer(2)]], uint t [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint eq[1024];
    const uint n = (P.pos0 + t + 1) / P.blk, words = (n + 31) / 32;
    device const float* s = score + ulong(t) * P.score_row;
    device uint* bm = bits + ulong(t) * P.bm_row;
    const uint per = (words + 1023) / 1024, w0 = min(tid * per, words), w1 = min(w0 + per, words);
    if (n <= P.n_top) {
        for (uint w = w0; w < w1; ++w) bm[w] = (w + 1) * 32 <= n ? ~0u : (1u << (n % 32)) - 1u;
        return;
    }
    uint need = P.n_top, key = 0;
    for (int shift = 24; shift >= 0; shift -= 8)
        key |= select_digit(Scores{s, n}, key, uint(shift), need, hist, tid) << shift;
    uint mine = 0;  // keys equal to the threshold in this thread's range
    for (uint b = w0 * 32; b < min(w1 * 32, n); ++b) mine += score_key(s[b]) == key;
    eq[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    for (uint i = 0; i < tid; ++i) before += eq[i];
    for (uint w = w0; w < w1; ++w) {
        uint word = 0;
        for (uint j = 0; j < 32 && w * 32 + j < n; ++j) {
            const uint k = score_key(s[w * 32 + j]);
            const bool take = k > key || (k == key && before < need);
            before += k == key;
            word |= uint(take) << j;
        }
        bm[w] = word;
    }
}

// Decode: the top n_top of the n_pool block scores of one token, by the radix select of idx_select (ties to the
// smaller block), their cells written in block order. One threadgroup of 1024; O(n_pool) instead of the rank by
// counting (O(n_pool^2)).
kernel void idx_top_r(device const float* score [[buffer(0)]], device int* cells [[buffer(1)]],
                      constant TopParams& P [[buffer(2)]], uint tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint above_of[1024], eq_of[1024];
    const uint n = P.n_pool, per = (n + 1023) / 1024, b0 = min(tid * per, n), b1 = min(b0 + per, n);
    uint need = P.n_top, key = 0;
    const bool all = n <= P.n_top;
    for (int shift = 24; shift >= 0 && !all; shift -= 8)
        key |= select_digit(Scores{score, n}, key, uint(shift), need, hist, tid) << shift;
    uint above = 0, eq = 0;
    for (uint b = b0; b < b1; ++b) {
        const uint k = score_key(score[b]);
        above += all || k > key;
        eq += !all && k == key;
    }
    above_of[tid] = above, eq_of[tid] = eq;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = 0, eq_before = 0;  // blocks taken by the threads before this one; equal keys before this thread's range
    for (uint i = 0; i < tid; ++i) {
        at += above_of[i] + min(eq_of[i], need > eq_before ? need - eq_before : 0u);
        eq_before += eq_of[i];
    }
    for (uint b = b0; b < b1; ++b) {
        const uint k = score_key(score[b]);
        const bool take = all || k > key || (k == key && eq_before < need);
        eq_before += !all && k == key;
        if (!take) continue;
        for (uint m = 0; m < P.blk; ++m) cells[at * P.blk + m] = int(b * P.blk + m);
        ++at;
    }
}

// Is cell c in the selection of tile token tt: its own block in progress up to its position, else a selected block
static inline bool cell_used(constant AttnTParams& P, device const uint* bits, uint t, uint c) {
    if (t >= P.nb) return false;  // rows past the chunk stay zero
    const uint pos = P.pos0 + t, tail0 = (pos + 1) / P.blk * P.blk;
    if (c > pos) return false;
    if (c >= tail0) return true;
    const uint b = c / P.blk;
    return (bits[ulong(t) * P.bm_row + b / 32] >> (b % 32)) & 1u;
}

struct AttnIo {
    device const float* q;
    device const float* kc;
    device const float* vc;
    device const uint* bits;
};

struct AttnAt {
    uint tid, sg, h0, kvh, t0;
};

struct AttnTg {
    threadgroup float* s;   // [H][64] scores, then probabilities
    threadgroup float* dg;  // [H][64] diagonal rescale matrices
    threadgroup float* m;   // [8H] running max per row (head, token)
    threadgroup float* l;   // [8H] running sum
    threadgroup uchar* used;
};

// Online softmax update of window w for the rows of this threadgroup (rows: 3 heads × 8 tokens)
template <uint H>
static inline void attn_window(constant AttnTParams& P, AttnIo io, AttnTg g, uint w, AttnAt at,
                               thread simdgroup_float8x8 (&acc)[H][8]) {
    const uint tid = at.tid, sg = at.sg, h0 = at.h0, kvh = at.kvh;
    const ulong kv_row = ulong(P.nkv) * P.hd, c0 = ulong(w) * 8;
    if (tid < 64) g.used[tid] = cell_used(P, io.bits, at.t0 + tid / 8, uint(c0) + tid % 8);
    if (sg < H) {
        simdgroup_float8x8 s = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
        device const float* qh = io.q + ulong(at.t0) * P.q_row + (h0 + sg) * 2 * P.hd;
        device const float* kh = io.kc + c0 * kv_row + kvh * P.hd;
        for (uint k = 0; k < P.hd; k += 8) {
            simdgroup_float8x8 a, b;
            simdgroup_load(a, qh + k, P.q_row);
            simdgroup_load(b, kh + k, kv_row, ulong2(0, 0), true);
            simdgroup_multiply_accumulate(s, a, b, s);
        }
        simdgroup_store(s, g.s + sg * 64, 8);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid < 8 * H) {
        const uint h = tid / 8, tt = tid % 8;
        threadgroup float* row = g.s + h * 64 + tt * 8;
        const float m_old = g.m[tid];
        float m_new = m_old;
        for (uint c = 0; c < 8; ++c)
            if (g.used[tt * 8 + c]) m_new = max(m_new, row[c] * P.scale);
        const float f = m_old == -INFINITY ? 0.0f : exp(m_old - m_new);
        float sum = 0.f;
        for (uint c = 0; c < 8; ++c) {
            const float p = g.used[tt * 8 + c] ? exp(row[c] * P.scale - m_new) : 0.0f;
            row[c] = p;
            sum += p;
        }
        g.m[tid] = m_new;
        g.l[tid] = g.l[tid] * f + sum;
        g.dg[h * 64 + tt * 9] = f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device const float* vh = io.vc + c0 * kv_row + kvh * P.hd + sg * 64;
    for (uint h = 0; h < H; ++h) {
        simdgroup_float8x8 p, d;
        simdgroup_load(p, g.s + h * 64, 8);
        simdgroup_load(d, g.dg + h * 64, 8);
        for (uint j = 0; j < 8; ++j) {
            simdgroup_float8x8 v;
            simdgroup_load(v, vh + j * 8, kv_row);
            simdgroup_multiply(acc[h][j], d, acc[h][j]);
            simdgroup_multiply_accumulate(acc[h][j], p, v, acc[h][j]);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

// The windows of the blocks below b_tail that any token of the tile selected, ascending; each once
template <uint H>
static inline void attn_selected(constant AttnTParams& P, AttnIo io, AttnTg g, AttnAt at, uint b_tail,
                                 threadgroup uint* uni, thread simdgroup_float8x8 (&acc)[H][8]) {
    for (uint wd = 0; wd * 32 < b_tail; ++wd) {
        if (at.tid == 0) {
            uint u = 0;
            for (uint tt = 0; tt < 8 && at.t0 + tt < P.nb; ++tt) u |= io.bits[ulong(at.t0 + tt) * P.bm_row + wd];
            uni[0] = u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint u = uni[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < 32; ++j) {
            const uint b = wd * 32 + j, w = b * P.blk / 8;
            if (b >= b_tail || !((u >> j) & 1u)) continue;
            if (j > 0 && ((u >> (j - 1)) & 1u) && (b - 1) * P.blk / 8 == w) continue;  // window already done
            attn_window<H>(P, io, g, w, at, acc);
        }
    }
}

// The position of one thread: threadgroup in the grid, thread and simdgroup in the threadgroup
struct AttnThread {
    uint3 tg;
    uint tid, sg;
};

// Attention of 8 tokens (tg.x) × H query heads (tg.z) of KV head tg.y, over the cells each token uses: the selected
// windows below the tile's first block in progress (from the union of the 8 bitmaps), then the windows up to the last
// token's position (blocks of at most 8 cells, 8 a multiple of the block). Output o[t][head] = softmax-weighted V times
// sigmoid(gate). 4 simdgroups; simdgroup s owns output columns 64s ... 64s + 63 (head_dim 256). Every head and token
// is computed alone (a cell a token does not use leaves its row unchanged, factor 1 and weight 0), so H changes
// only the number of threadgroups, never a bit. H = 1 is faster than H = 3 (which shares the K and V loads of a window
// over 3 heads): -32 % on a 35-token chunk at 7900 tokens of context (3 times the threadgroups, 40 on 32 cores with
// H = 3), -2 to -5 % on 2048-token chunks (research/measurements/2026-11-15-mac-attn-h1.md).
template <uint H>
static inline void attend_tile(AttnIo io, device float* o, constant AttnTParams& P, AttnTg g, threadgroup uint* uni,
                               AttnThread th) {
    const uint3 tg = th.tg;
    const uint tid = th.tid, sg = th.sg;
    device const float* q = io.q;
    threadgroup float *dg = g.dg, *m = g.m, *l = g.l;
    const uint t0 = P.t_base + tg.x * 8, kvh = tg.y, h0 = kvh * (P.nh / P.nkv) + tg.z * H;
    for (uint i = tid; i < H * 64; i += 128) dg[i] = 0.f;
    if (tid < 8 * H) m[tid] = -INFINITY, l[tid] = 0.f;
    simdgroup_float8x8 acc[H][8];
    for (uint h = 0; h < H; ++h)
        for (uint j = 0; j < 8; ++j) acc[h][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.f);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const AttnAt at{tid, sg, h0, kvh, t0};
    const uint first = P.pos0 + t0, last = min(first + 7, P.pos0 + P.nb - 1);
    const uint w_tail = (first + 1) / P.blk * P.blk / 8;  // first window that may hold a block in progress
    attn_selected<H>(P, io, g, at, w_tail * 8 / P.blk, uni, acc);
    for (uint w = w_tail; w * 8 <= last; ++w) attn_window<H>(P, io, g, w, at, acc);
    for (uint h = 0; h < H; ++h)
        for (uint j = 0; j < 8; ++j)
            simdgroup_store(acc[h][j], o + ulong(t0) * P.o_row + (h0 + h) * P.hd + sg * 64 + j * 8, P.o_row);
    threadgroup_barrier(mem_flags::mem_device);
    for (uint i = tid; i < H * 8 * P.hd; i += 128) {
        const uint h = i / (8 * P.hd), tt = i / P.hd % 8, c = i % P.hd;
        if (t0 + tt >= P.nb) continue;
        device float* y = o + ulong(t0 + tt) * P.o_row + (h0 + h) * P.hd + c;
        const float gate = q[ulong(t0 + tt) * P.q_row + (h0 + h) * 2 * P.hd + P.hd + c];
        *y = *y / l[h * 8 + tt] * sigmoidf(gate);
    }
}

kernel void attend_t(device const float* q [[buffer(0)]], device const float* kc [[buffer(1)]],
                     device const float* vc [[buffer(2)]], device const uint* bits [[buffer(3)]],
                     device float* o [[buffer(4)]], constant AttnTParams& P [[buffer(5)]],
                     uint3 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                     uint sg [[simdgroup_index_in_threadgroup]]) {
    threadgroup float s[64], dg[64], m[8], l[8];
    threadgroup uchar used[64];
    threadgroup uint uni[1];
    attend_tile<1>(AttnIo{q, kc, vc, bits}, o, P, AttnTg{s, dg, m, l, used}, uni, AttnThread{tg, tid, sg});
}
