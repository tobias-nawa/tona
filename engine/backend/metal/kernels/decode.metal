// Group decoders of the GGUF weight types, shared by the Metal backend and the benchmarks in bench/mac. Each dec()
// decodes group g (8 consecutive weights) of a row starting at p, following engine/model/quant.cpp. The IQ tables
// come first: engine/model/quant_tables.h converted to MSL (shaders.mm).

static inline float h2f(device const uchar* p) { return float(*(device const half*)p); }
static inline uint rd32(device const uchar* p) {
    return uint(p[0]) | uint(p[1]) << 8 | uint(p[2]) << 16 | uint(p[3]) << 24;
}
static inline float sgn(uchar s, int k) { return (s & kmask_iq2xs[k]) ? -1.f : 1.f; }

// group g (of 8 weights) of a row starting at p
struct IQ2S {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 82;
        uint j = g % 32, ib = j / 4, l = j % 4;
        float d = h2f(b);
        uchar sc = b[2 + 64 + 8 + ib];
        float dl = d * (0.5f + ((sc >> (4 * (l / 2))) & 0xf)) * 0.25f;
        uint idx = b[2 + ib * 4 + l] | ((uint(b[2 + 64 + ib]) << (8 - 2 * l)) & 0x300);
        ulong gr = iq2s_grid[idx];
        uchar s = b[2 + 32 + ib * 4 + l];
        for (int k = 0; k < 8; ++k) w[k] = dl * float((gr >> (8 * k)) & 0xff) * sgn(s, k);
    }
};

struct IQ3XXS {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 98;
        uint j = g % 32, ib = j / 4, l = j % 4;
        float d = h2f(b);
        uint aux = rd32(b + 2 + 64 + 4 * ib);
        float db = d * (0.5f + (aux >> 28)) * 0.5f;
        uchar s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        uint g1 = iq3xxs_grid[b[2 + ib * 8 + 2 * l]], g2 = iq3xxs_grid[b[2 + ib * 8 + 2 * l + 1]];
        for (int k = 0; k < 4; ++k) {
            w[k] = db * float((g1 >> (8 * k)) & 0xff) * sgn(s, k);
            w[k + 4] = db * float((g2 >> (8 * k)) & 0xff) * sgn(s, k + 4);
        }
    }
};

// TONA_IQ3_XXS_R (docs/use/tona-types.md): 196-byte super-blocks of 512 weights, index bytes at 0 (group g of the
// super-block at 2g), sign and scale words at 128 (one per 32 weights), block scales at 192; aligned loads instead of
// IQ3_XXS's byte picks, the same arithmetic per weight (research/measurements/2026-11-12-mac-expert-decode-probes.md)
struct IQ3XXSR {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 64) * 196;
        uint j = g % 64, l = j % 4;
        float d = float(*(device const half*)(b + 192 + 2 * (j / 32)));
        uint aux = *(device const uint*)(b + 128 + 4 * (j / 4));
        float db = d * (0.5f + (aux >> 28)) * 0.5f;
        uchar s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        ushort ix = *(device const ushort*)(b + 2 * j);
        uint g1 = iq3xxs_grid[ix & 0xff], g2 = iq3xxs_grid[ix >> 8];
        for (int k = 0; k < 4; ++k) {
            w[k] = db * float((g1 >> (8 * k)) & 0xff) * sgn(s, k);
            w[k + 4] = db * float((g2 >> (8 * k)) & 0xff) * sgn(s, k + 4);
        }
    }
};

struct IQ3S {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 110;
        uint j = g % 32, ib = j / 4, l = j % 4;
        float d = h2f(b);
        uchar sc = b[2 + 64 + 8 + 32 + ib / 2];
        float db = d * (1 + 2 * ((ib % 2) ? (sc >> 4) : (sc & 0xf)));
        uint qh = b[2 + 64 + ib];
        uint i1 = b[2 + 8 * ib + 2 * l] | ((qh << (8 - 2 * l)) & 256);
        uint i2 = b[2 + 8 * ib + 2 * l + 1] | ((qh << (7 - 2 * l)) & 256);
        uchar s = b[2 + 64 + 8 + 4 * ib + l];
        uint g1 = iq3s_grid[i1], g2 = iq3s_grid[i2];
        for (int k = 0; k < 4; ++k) {
            w[k] = db * float((g1 >> (8 * k)) & 0xff) * sgn(s, k);
            w[k + 4] = db * float((g2 >> (8 * k)) & 0xff) * sgn(s, k + 4);
        }
    }
};

// 8 bytes from a 2-byte aligned address (Q6_K, IQ4_NL and Q8_0 blocks have even, not 4-aligned sizes)
static inline uint2 load8_a2(device const uchar* p) {
    packed_ushort4 v = *(device const packed_ushort4*)p;
    return uint2(uint(v[0]) | uint(v[1]) << 16, uint(v[2]) | uint(v[3]) << 16);
}

static inline void put8(thread float* w, uchar4 a, uchar4 b, float scale, float shift) {
    float4 fa = float4(a) * scale + shift, fb = float4(b) * scale + shift;
    w[0] = fa.x, w[1] = fa.y, w[2] = fa.z, w[3] = fa.w, w[4] = fb.x, w[5] = fb.y, w[6] = fb.z, w[7] = fb.w;
}

struct IQ4XS {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 136;
        uint j = g % 32, ib = j / 4, sub = j % 4;
        float d = h2f(b);
        uint sh = uint(b[2]) | uint(b[3]) << 8;
        int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf) | (((sh >> (2 * ib)) & 3) << 4);
        float dl = d * float(ls - 32);
        uint2 q = *(device const uint2*)(b + 8 + 16 * ib + 8 * (sub % 2));
        uint s4 = sub < 2 ? 0 : 4;
        uchar4 a = as_type<uchar4>((q.x >> s4) & 0x0F0F0F0Fu), c = as_type<uchar4>((q.y >> s4) & 0x0F0F0F0Fu);
        for (int k = 0; k < 4; ++k) w[k] = dl * float(kvalues_iq4nl[a[k]]), w[4 + k] = dl * float(kvalues_iq4nl[c[k]]);
    }
};

struct IQ4NL {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 4) * 18;
        uint sub = g % 4;
        float d = h2f(b);
        uint2 q = load8_a2(b + 2 + 8 * (sub % 2));
        uint s4 = sub < 2 ? 0 : 4;
        uchar4 a = as_type<uchar4>((q.x >> s4) & 0x0F0F0F0Fu), c = as_type<uchar4>((q.y >> s4) & 0x0F0F0F0Fu);
        for (int k = 0; k < 4; ++k) w[k] = d * float(kvalues_iq4nl[a[k]]), w[4 + k] = d * float(kvalues_iq4nl[c[k]]);
    }
};

struct Q20 {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 8) * 18;
        uint s = g % 8;
        float d = h2f(b);
        uint q = uint(b[2 + 2 * s]) | uint(b[3 + 2 * s]) << 8;
        for (int k = 0; k < 8; ++k) w[k] = (float((q >> (2 * k)) & 3) - 1.f) * d;
    }
};

struct IQ2XXS {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 66;
        uint j = g % 32, ib = j / 4, l = j % 4;
        float d = h2f(b);
        uint aux = rd32(b + 2 + 8 * ib + 4);
        float db = d * (0.5f + (aux >> 28)) * 0.25f;
        ulong gr = iq2xxs_grid[b[2 + 8 * ib + l]];
        uchar s = ksigns_iq2xs[(aux >> (7 * l)) & 127];
        for (int k = 0; k < 8; ++k) w[k] = db * float((gr >> (8 * k)) & 0xff) * sgn(s, k);
    }
};

struct Q80 {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 4) * 34;
        float d = h2f(b);
        uint2 q = load8_a2(b + 2 + 8 * (g % 4));
        float4 fa = float4(as_type<char4>(q.x)) * d, fb = float4(as_type<char4>(q.y)) * d;
        w[0] = fa.x, w[1] = fa.y, w[2] = fa.z, w[3] = fa.w, w[4] = fb.x, w[5] = fb.y, w[6] = fb.z, w[7] = fb.w;
    }
};

static inline void scale_min_k4(uint j, device const uchar* q, thread uint& d, thread uint& m) {
    if (j < 4) {
        d = q[j] & 63, m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4), m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct Q4K {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 144;
        uint j = g % 32, c = j / 8, h = (j % 8) / 4, sub = j % 4, sc, m;
        scale_min_k4(2 * c + h, b + 4, sc, m);
        float dl = h2f(b) * float(sc), ml = h2f(b + 2) * float(m);
        uint2 q = *(device const uint2*)(b + 16 + 32 * c + 8 * sub);
        uint s4 = 4 * h;
        put8(w, as_type<uchar4>((q.x >> s4) & 0x0F0F0F0Fu), as_type<uchar4>((q.y >> s4) & 0x0F0F0F0Fu), dl, -ml);
    }
};

struct Q5K {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 176;
        uint j = g % 32, c = j / 8, h = (j % 8) / 4, sub = j % 4, sc, m;
        scale_min_k4(2 * c + h, b + 4, sc, m);
        float dl = h2f(b) * float(sc), ml = h2f(b + 2) * float(m);
        uint2 q = *(device const uint2*)(b + 48 + 32 * c + 8 * sub);
        uint2 qh = *(device const uint2*)(b + 16 + 8 * sub);
        uint s4 = 4 * h, sb = 2 * c + h;
        uint lo = ((q.x >> s4) & 0x0F0F0F0Fu) | (((qh.x >> sb) & 0x01010101u) << 4);
        uint hi = ((q.y >> s4) & 0x0F0F0F0Fu) | (((qh.y >> sb) & 0x01010101u) << 4);
        put8(w, as_type<uchar4>(lo), as_type<uchar4>(hi), dl, -ml);
    }
};

struct Q6K {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const uchar* b = p + (g / 32) * 210;
        uint j = g % 32, n = j / 16, v0 = 8 * (j % 16), qq = v0 / 32, l0 = v0 % 32;
        uint2 ql = load8_a2(b + 64 * n + (qq & 1) * 32 + l0);
        uint2 qh = load8_a2(b + 128 + 32 * n + l0);
        float d = h2f(b + 208) * float(((device const char*)(b + 192 + 8 * n))[l0 / 16 + 2 * qq]);
        uint s4 = qq < 2 ? 0 : 4, s2 = 2 * qq;
        uint lo = ((ql.x >> s4) & 0x0F0F0F0Fu) | (((qh.x >> s2) & 0x03030303u) << 4);
        uint hi = ((ql.y >> s4) & 0x0F0F0F0Fu) | (((qh.y >> s2) & 0x03030303u) << 4);
        put8(w, as_type<uchar4>(lo), as_type<uchar4>(hi), d, -32.f * d);
    }
};

struct BF16 {
    static void dec(device const uchar* p, uint g, thread float* w) {
        uint4 v = *(device const uint4*)(p + 16 * g);
        for (int k = 0; k < 4; ++k)
            w[2 * k] = as_type<float>(v[k] << 16), w[2 * k + 1] = as_type<float>(v[k] & 0xFFFF0000u);
    }
};

struct F32 {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const float4* v = (device const float4*)(p + 32 * g);
        float4 a = v[0], b = v[1];
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w, w[4] = b.x, w[5] = b.y, w[6] = b.z, w[7] = b.w;
    }
};

struct F16 {
    static void dec(device const uchar* p, uint g, thread float* w) {
        device const half4* v = (device const half4*)(p + 16 * g);
        float4 a = float4(v[0]), b = float4(v[1]);
        w[0] = a.x, w[1] = a.y, w[2] = a.z, w[3] = a.w, w[4] = b.x, w[5] = b.y, w[6] = b.z, w[7] = b.w;
    }
};

// TONA_Q6_K_T (docs/use/tona-types.md): tiles of 8 rows in MMA fragment order; entry e = 32 c + lane holds group q of
// row r8 in a 32-bit low-bit word, a 16-bit high-bit word and a scale byte. The arithmetic of Q6K, so the same bits.
// bits 4j and 4j + 1 of a 16-bit word (j = 0..3) to bits 8j and 8j + 1
static inline uint spread_pairs(uint h) {
    h &= 0x3333u;
    h = (h | h << 8) & 0x00FF00FFu;
    return (h | h << 4) & 0x0F0F0F0Fu;
}

struct Q6KT {
    static void dec_tile(device const uchar* t, uint r8, uint k, uint g, thread float* w) {
        const uint c = g / 4, q = g % 4;
        const uint e = 32 * c + (((r8 & 4) << 2) | ((r8 & 3) << 1) | ((q & 2) << 2) | (q & 1));
        const uint ql = *(device const uint*)(t + 4 * e), qh = *(device const ushort*)(t + 4 * k + 2 * e);
        const float d =
            h2f(t + 6 * k + k / 2 + 2 * ((c / 8) * 8 + r8)) * float(((device const char*)(t + 6 * k))[e / 2]);
        put(ql, qh, d, w);
    }
    // the even weights (0, 2, 4, 6) and the odd ones as bytes: one mask for the nibbles, the bit pairs spread
    static void put(uint ql, uint qh, float d, thread float* w) {
        const uint ev = (ql & 0x0F0F0F0Fu) | spread_pairs(qh) << 4;
        const uint od = ((ql >> 4) & 0x0F0F0F0Fu) | spread_pairs(qh >> 2) << 4;
        const float m = -32.f * d;
        const float4 a = float4(as_type<uchar4>(ev)) * d + m, b = float4(as_type<uchar4>(od)) * d + m;
        w[0] = a.x, w[1] = b.x, w[2] = a.y, w[3] = b.y, w[4] = a.z, w[5] = b.z, w[6] = a.w, w[7] = b.w;
    }
};

// Group g of row `row` of a matrix at w with rs bytes per row. Rows of the plain types are contiguous; a tile type
// finds its tile and the row inside it (only dense layer matrices have one: model/gguf.cpp check_tile_type)
template <typename T>
struct RowAt {
    static void dec(device const uchar* w, uint row, uint rs, uint g, thread float* v) {
        T::dec(w + ulong(row) * rs, g, v);
    }
};
template <>
struct RowAt<Q6KT> {
    static void dec(device const uchar* w, uint row, uint rs, uint g, thread float* v) {
        Q6KT::dec_tile(w + ulong(row / 8) * 8 * rs, row % 8, rs / 210 * 256, g, v);
    }
};

// Group g of rows r0 .. r0 + R of a matrix: load(g) reads what the R rows need, dec decodes row r of it. Plain types
// decode row by row.
template <typename T, int R>
struct Rows {
    device const uchar* w;
    uint r0, rs, k;
    uint load(uint g) const { return g; }
    void dec(uint g, int r, thread float* v) const { RowAt<T>::dec(w, r0 + r, rs, g, v); }
};
// Q6KT, 4 rows (r0 % 4 == 0): the lane's entries of the 4 rows are e0 + o + 2r, neighbours in the tile, so each field
// comes in one aligned load instead of four. Same values, same order as dec_tile. With four loads per field the
// 2-token kernel ran 3.5 ms per pass slower than Q6K (research/measurements/2026-11-12-mac-q6k-tile.md, run 6).
struct Q6KT4 {
    uint ql[4], qh[4];
    float d[4];
};
template <>
struct Rows<Q6KT, 4> {
    device const uchar* w;
    uint r0, rs, k;
    Q6KT4 load(uint g) const {
        device const uchar* t = w + ulong(r0 / 8) * 8 * rs;
        const uint c = g / 4, q = g % 4, e0 = 32 * c + ((r0 & 4) << 2) + ((q & 2) << 2);
        const uint4 l0 = *(device const uint4*)(t + 4 * e0), l1 = *(device const uint4*)(t + 4 * e0 + 16);
        const uint4 h = *(device const uint4*)(t + 4 * k + 2 * e0);
        const char4 sc = as_type<char4>(*(device const uint*)(t + 6 * k + e0 / 2));
        const float4 d = float4(*(device const half4*)(t + 6 * k + k / 2 + 2 * ((c / 8) * 8 + (r0 & 4))));
        const bool o = q & 1;
        Q6KT4 f;
        f.ql[0] = o ? l0.y : l0.x, f.ql[1] = o ? l0.w : l0.z, f.ql[2] = o ? l1.y : l1.x, f.ql[3] = o ? l1.w : l1.z;
        for (int r = 0; r < 4; ++r) f.qh[r] = o ? h[r] >> 16 : h[r] & 0xFFFFu, f.d[r] = d[r] * float(sc[r]);
        return f;
    }
    void dec(thread const Q6KT4& f, int r, thread float* v) const { Q6KT::put(f.ql[r], f.qh[r], f.d[r], v); }
};

// dot products of rows r0 .. r0 + R (row stride rs bytes) with B inputs; every input load serves all R rows
template <typename T, int B, int R>
static inline void rows_dot(device const uchar* w, uint r0, uint rs, device const float* x, uint k, uint lane,
                            thread float (&acc)[R][B]) {
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = 0.f;
    const Rows<T, R> rows{w, r0, rs, k};
    for (uint g = lane; g < k / 8; g += 32) {
        float4 xa[B], xc[B];
        for (int b = 0; b < B; ++b) {
            device const float4* xv = (device const float4*)(x + b * k + g * 8);
            xa[b] = xv[0], xc[b] = xv[1];
        }
        const auto f = rows.load(g);
        for (int r = 0; r < R; ++r) {
            float v[8];
            rows.dec(f, r, v);
            float4 wa = float4(v[0], v[1], v[2], v[3]), wc = float4(v[4], v[5], v[6], v[7]);
            for (int b = 0; b < B; ++b) acc[r][b] += dot(wa, xa[b]) + dot(wc, xc[b]);
        }
    }
    for (int r = 0; r < R; ++r)
        for (int b = 0; b < B; ++b) acc[r][b] = simd_sum(acc[r][b]);
}

// The same from a row pointer (experts, benchmarks): plain types only
template <typename T, int B, int R>
static inline void rows_dot(device const uchar* row, uint rs, device const float* x, uint k, uint lane,
                            thread float (&acc)[R][B]) {
    rows_dot<T, B, R>(row, 0u, rs, x, k, lane, acc);
}
