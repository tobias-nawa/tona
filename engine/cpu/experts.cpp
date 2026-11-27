#include "cpu/experts.h"

#include <immintrin.h>
#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <thread>

#include "backend/env.h"
#include "backend/runtime_env.h"
#include "cpu/detmath.h"
#include "cpu/experts_act.h"
#include "cpu/experts_decode.h"

namespace tona::cpu {

namespace {

using namespace tona::cpu::detail;

// Summation order as row_q8 on the GPU with L = 8 lanes per row (expert groups in the megakernel,
// TONA_MEGA_LANES_EXP): lane j sums the sub-blocks j, j + 8, ... in turn, s = fma(d_x, r, s) with
// r = fma(s_hi, hi, s_lo * lo) (Q2_0 adds fma(-s, sum x_lo, r), fma(-s, sum x_hi, r)), then the xor tree over 4, 2, 1.
// Integer partial sums are exact, so the result is bit-identical to the GPU. Per 8 sub-blocks (4 zmm): vpdpbusd gives
// 16 sums over 4 values per zmm; unpack/add folds four zmm into the sums per 16-value half (lane 4g + k: group g of
// zmm k).
[[gnu::always_inline]] inline __m512i fold4(__m512i a, __m512i b, __m512i c, __m512i d) {
    const __m512i t1 = _mm512_add_epi32(_mm512_unpacklo_epi32(a, b), _mm512_unpackhi_epi32(a, b));
    const __m512i t2 = _mm512_add_epi32(_mm512_unpacklo_epi32(c, d), _mm512_unpackhi_epi32(c, d));
    return _mm512_add_epi32(_mm512_unpacklo_epi64(t1, t2), _mm512_unpackhi_epi64(t1, t2));
}

// Metadata of x per sub-block as float fields (padded to 8), once per call instead of per row; writing them scalar
// and reading them as a vector right away stalled store forwarding
constexpr int kMaxSubs = 128;
struct XMeta {
    alignas(32) float d[kMaxSubs], s0[kMaxSubs], s1[kMaxSubs];
};

// What the row kernels need besides the weights: x (int8 + metadata, stride xs values), the vectors in mask, and
// the output y (stride ys). subs = row length in 32-value sub-blocks.
struct RowJob {
    const uint8_t* w;
    int64_t row_bytes, subs;
    const int8_t* xq;
    const XMeta* xm;
    int64_t xs;
    uint32_t mask;
    float* y;
    int64_t ys;
};

// A matvec over the rows [r0, r1) of nb vectors, with x metadata already expanded to XMeta
struct MatvecJob {
    uint32_t type;
    const uint8_t* w;
    int64_t row_bytes, in, r0, r1;
    const int8_t* xq;
    const XMeta* xm;
    int64_t xs;
    int nb;
    uint32_t mask;
    float* y;
    int64_t ys;
};

// One group of 2 * NCH sub-blocks from s0 (NCH = 4 except at the end of a row) for R rows and NB tokens; NCH is
// fixed so the unpacked weights stay in registers
template <uint32_t T, int R, int NB, int NCH>
[[gnu::always_inline]] inline void group_step(const RowJob& j, int64_t s0, int64_t r, __m256 (&acc)[R][NB]) {
    const __m512i zero = _mm512_setzero_si512();
    // Lane s of the group: lo of sub-block s (even: zmm s/2 group 0, odd: zmm (s-1)/2 group 2), then hi
    const __m512i perm = _mm512_setr_epi32(0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15);
    constexpr __mmask8 lanes = (__mmask8)((1u << (2 * NCH)) - 1);
    __m256 xd[NB], xs0[NB], xs1[NB];
#pragma GCC unroll 4
    for (int b = 0; b < NB; b++) {
        xd[b] = _mm256_loadu_ps(j.xm[b].d + s0);
        if constexpr (T == 42 || kOffsetFormat<T>)
            xs0[b] = _mm256_loadu_ps(j.xm[b].s0 + s0), xs1[b] = _mm256_loadu_ps(j.xm[b].s1 + s0);
    }
#pragma GCC unroll 4
    for (int i = 0; i < R; i++) {
        const uint8_t* row = j.w + (r + i) * j.row_bytes;
        Dec o[NCH];
#pragma GCC unroll 4
        for (int64_t c = 0; c < NCH; c++) decode<T>(row, s0 + 2 * c, o[c]);
        __m256 SL, SH, BL, BH;
        if constexpr (kOffsetFormat<T>)
            scales_bias8<T>(row, s0, SL, SH, BL, BH);
        else
            scales8<T>(row, s0, 2 * NCH, SL, SH);
#pragma GCC unroll 4
        for (int b = 0; b < NB; b++) {
            __m512i dot[4] = {zero, zero, zero, zero};
#pragma GCC unroll 4
            for (int64_t c = 0; c < NCH; c++) {
                const __m512i x = _mm512_loadu_si512(j.xq + b * j.xs + (s0 + 2 * c) * 32);
                dot[c] = _mm512_dpbusd_epi32(zero, o[c].u, _mm512_mask_sub_epi8(x, o[c].neg, zero, x));
            }
            const __m512i lh = _mm512_permutexvar_epi32(perm, fold4(dot[0], dot[1], dot[2], dot[3]));
            const __m256 lo = _mm256_cvtepi32_ps(_mm512_castsi512_si256(lh));
            const __m256 hi = _mm256_cvtepi32_ps(_mm512_extracti32x8_epi32(lh, 1));
            __m256 rr = _mm256_fmadd_ps(SH, hi, _mm256_mul_ps(SL, lo));
            if constexpr (T == 42) {
                const __m256 nb_ = _mm256_sub_ps(_mm256_setzero_ps(), SL);  // b_lo = b_hi = -s
                rr = _mm256_fmadd_ps(nb_, xs0[b], rr);
                rr = _mm256_fmadd_ps(nb_, xs1[b], rr);
            } else if constexpr (kOffsetFormat<T>) {
                rr = _mm256_fmadd_ps(BL, xs0[b], rr);
                rr = _mm256_fmadd_ps(BH, xs1[b], rr);
            }
            if constexpr (NCH == 4)
                acc[i][b] = _mm256_fmadd_ps(xd[b], rr, acc[i][b]);
            else
                acc[i][b] = _mm256_mask3_fmadd_ps(xd[b], rr, acc[i][b], lanes);
        }
    }
}

// R rows at once (independent chains), x used per 8 sub-blocks for all R rows
template <uint32_t T, int R, int NB>
inline void rows_r(const RowJob& j, int64_t r) {
    __m256 acc[R][NB];
    for (int i = 0; i < R; i++)
        for (int b = 0; b < NB; b++) acc[i][b] = _mm256_setzero_ps();
    int64_t s0 = 0;
    for (; s0 + 8 <= j.subs; s0 += 8) group_step<T, R, NB, 4>(j, s0, r, acc);
    switch ((j.subs - s0) / 2) {  // subs is even (row length is a multiple of 64)
        case 1:
            group_step<T, R, NB, 1>(j, s0, r, acc);
            break;
        case 2:
            group_step<T, R, NB, 2>(j, s0, r, acc);
            break;
        case 3:
            group_step<T, R, NB, 3>(j, s0, r, acc);
            break;
        default:
            break;
    }
    for (int i = 0; i < R; i++)
        for (int b = 0; b < NB; b++) {
            if (!(j.mask >> b & 1)) continue;
            // xor tree like __shfl_xor with 4, 2, 1: lane j + lane j ^ off
            const __m128 h = _mm_add_ps(_mm256_castps256_ps128(acc[i][b]), _mm256_extractf128_ps(acc[i][b], 1));
            const __m128 q = _mm_add_ps(h, _mm_movehl_ps(h, h));
            j.y[b * j.ys + r + i] = _mm_cvtss_f32(_mm_add_ss(q, _mm_shuffle_ps(q, q, 1)));
        }
}

// nb <= 4 vectors: the set bits of mask are collected at the front so only used tokens compute
template <uint32_t T, int NB>
void rows_nb(const RowJob& j, int64_t r0, int64_t r1) {
    int64_t r = r0;
    for (; r + 4 <= r1; r += 4) rows_r<T, 4, NB>(j, r);
    for (; r < r1; r++) rows_r<T, 1, NB>(j, r);
}

void build_xmeta(const Q8Meta* m, int64_t subs, XMeta& o) {
    for (int64_t j = 0; j < (subs + 7) / 8 * 8; j++) {
        const bool in_row = j < subs;
        o.d[j] = in_row ? m[j].d : 0.0f;
        o.s0[j] = in_row ? (float)m[j].s0 : 0.0f;
        o.s1[j] = in_row ? (float)m[j].s1 : 0.0f;
    }
}

// Tokens in groups of up to 4 per pass over the weights; the tokens of mask must be consecutive
template <uint32_t T>
void rows_contiguous(const RowJob& j0, const MatvecJob& a, int n, int first) {
    // Each token computes the same steps whichever group it lands in
    for (int k0 = 0; k0 < n; k0 += 4) {
        RowJob j = j0;
        j.mask = (1u << std::min(4, n - k0)) - 1;
        j.xq = a.xq + first * a.xs + k0 * a.xs;
        j.xm = a.xm + first + k0;
        j.y = a.y + first * a.ys + k0 * a.ys;
        switch (std::min(4, n - k0)) {
            case 1:
                rows_nb<T, 1>(j, a.r0, a.r1);
                break;
            case 2:
                rows_nb<T, 2>(j, a.r0, a.r1);
                break;
            case 3:
                rows_nb<T, 3>(j, a.r0, a.r1);
                break;
            case 4:
                rows_nb<T, 4>(j, a.r0, a.r1);
                break;
            default:
                break;
        }
    }
}

// xm: metadata per token b (build_xmeta); only those with a bit in mask are read
template <uint32_t T>
void rows(const MatvecJob& a) {
    RowJob j{a.w, a.row_bytes, a.in / 32, a.xq, a.xm, a.xs, 1, a.y, a.ys};
    // Used tokens one by one: x and y per token through pointers, so gaps in mask cost nothing
    int used[8], n = 0;
    for (int b = 0; b < a.nb; b++)
        if (a.mask >> b & 1) used[n++] = b;
    if (n == 0) return;
    // Consecutive tokens (the common case, 0..n-1) in one pass, otherwise one token at a time
    bool contiguous = true;
    for (int k = 0; k < n; k++) contiguous &= used[k] == used[0] + k;
    if (contiguous) return rows_contiguous<T>(j, a, n, used[0]);
    for (int k = 0; k < n; k++) {
        j.xq = a.xq + used[k] * a.xs, j.xm = a.xm + used[k], j.y = a.y + used[k] * a.ys;
        rows_nb<T, 1>(j, a.r0, a.r1);
    }
}

// K-quant rows hold whole blocks of 256 values; a row that does not would read past its end
int64_t k_quant_row_bytes(int64_t n, int64_t block_bytes) {
    if (n % 256) throw std::runtime_error("cpu::experts: a K-quant row is not a multiple of 256 values");
    return n / 256 * block_bytes;
}

int64_t row_bytes_of(uint32_t type, int64_t n) {
    switch (type) {
        case 16:
            return k_quant_row_bytes(n, 66);
        case 22:
            return n / 256 * 82;
        case 18:
            return n / 256 * 98;
        case 21:
            return n / 256 * 110;
        case 23:
            return n / 256 * 136;
        case 20:
            return n / 32 * 18;
        case 42:
            return n / 64 * 18;
        case 12:
            return k_quant_row_bytes(n, 144);
        case 14:
            return k_quant_row_bytes(n, 210);
        default:
            break;
    }
    throw std::runtime_error("cpu::experts: format not supported");
}

void matvec_x(const MatvecJob& a) {
    switch (a.type) {
        case 16:
            return rows<16>(a);
        case 22:
            return rows<22>(a);
        case 18:
            return rows<18>(a);
        case 21:
            return rows<21>(a);
        case 23:
            return rows<23>(a);
        case 20:
            return rows<20>(a);
        case 42:
            return rows<42>(a);
        case 12:
            return rows<12>(a);
        case 14:
            return rows<14>(a);
        default:
            break;
    }
    throw std::runtime_error("cpu::matvec_q8: format not supported");
}

}  // namespace

int64_t row_bytes(uint32_t type, int64_t n) { return row_bytes_of(type, n); }

bool expert_type_supported(uint32_t type) {
    return type == 16 || type == 22 || type == 18 || type == 21 || type == 23 || type == 20 || type == 42 ||
           type == 12 || type == 14;
}

void quantize_q8(const float* x, int64_t n, int8_t* q, Q8Meta* m) {
    for (int64_t b = 0; b < n / 32; b++) {
        float amax = 0;
        for (int k = 0; k < 32; k++) amax = std::max(amax, std::fabs(x[b * 32 + k]));
        const float d = amax / 127.0f;
        int s0 = 0, s1 = 0;
        for (int k = 0; k < 32; k++) {
            const int qi = d > 0 ? (int)std::rint(x[b * 32 + k] / d) : 0;
            q[b * 32 + k] = (int8_t)qi;
            (k < 16 ? s0 : s1) += qi;
        }
        m[b] = {d, (int16_t)s0, (int16_t)s1};
    }
}

void matvec_q8(const MatvecArgs& a) {
    if (a.nb > 8) throw std::runtime_error("cpu::matvec_q8: at most 8 vectors");
    if (a.in % 64) throw std::runtime_error("cpu::matvec_q8: row length is not a multiple of 64");
    if (a.in / 32 > kMaxSubs) throw std::runtime_error("cpu::matvec_q8: row too long");
    XMeta xmv[8];
    for (int b = 0; b < a.nb; b++)
        if (a.mask >> b & 1) build_xmeta(a.xm + b * a.xs / 32, a.in / 32, xmv[b]);
    matvec_x({a.type, a.w, a.row_bytes, a.in, a.r0, a.r1, a.xq, xmv, a.xs, a.nb, a.mask, a.y, a.ys});
}

// Threads of the experts: OMP_NUM_THREADS, which main() sets to the physical cores unless a launcher or the user chose
// (backend/runtime_env.h). Without it (tests, tools that do not restart) also the physical cores: with SMT, 12 instead
// of 6 threads on the 9600X made the experts slower. The planner's 60 us per CPU expert (backend/cpu_cost.h) was
// measured at 6 threads on the 9600X (6 physical cores); on the 9950X the engine runs 16 and the constant has not been
// re-fitted (research/measurements/2026-11-17-omp-threads.md measured decode +1.3 to +4.3 % with 16 against 6 threads).
int expert_threads() {
    static const int n = [] {
        if (env::text("OMP_NUM_THREADS")) return omp_get_max_threads();
        return runtime_env::detect_physical_cores();
    }();
    return n;
}

namespace {

constexpr int64_t kChunk = 16;  // rows per work chunk

// State shared by the three parallel stages of experts_ffn_fix
struct FfnCtx {
    FfnCtx(const FfnArgs& args, ExpertScratch& scratch, int64_t* out)
        : a(args),
          s(scratch),
          xa(reinterpret_cast<XMeta*>(scratch.xa.data())),
          y(out),
          mask(args.E.size(), 0),
          ne((int)args.E.size()) {}

    XMeta xx[8];  // metadata of x, once per layer (left uninitialized, filled by build_xmeta)
    const FfnArgs& a;
    ExpertScratch& s;
    XMeta* xa;  // metadata of the activations per (expert, token), after rounding
    int64_t* y;
    std::vector<uint32_t> mask;  // per expert: tokens that use it
    int ne;
};

// Throws if matvec_q8 could throw: a throw inside the OpenMP region ends in std::terminate
void check_ffn_args(const FfnArgs& a) {
    if (a.nb < 1 || a.nb > 8 || a.d % 64 || a.ff % 64 || a.d / 32 > kMaxSubs || a.ff / 32 > kMaxSubs)
        throw std::runtime_error("cpu::experts_ffn: shapes do not fit");
    for (const ExpertW& X : a.E)
        (void)(X.gate ? row_bytes_of(X.t_gate, a.d) : 0), (void)row_bytes_of(X.t_up, a.d),
            (void)row_bytes_of(X.t_down, a.ff);
}

// gate and up (an ungated expert has up only): chunk (expert, 16 rows), y per expert [nb][ff]
void stage_gate_up(const FfnCtx& c) {
    const FfnArgs& a = c.a;
    const int64_t gchunks = (a.ff + kChunk - 1) / kChunk;
#pragma omp for schedule(dynamic, 2)
    for (int64_t k = 0; k < c.ne * gchunks; k++) {
        const int e = (int)(k / gchunks);
        const int64_t r0 = (k % gchunks) * kChunk, r1 = std::min(a.ff, r0 + kChunk);
        const ExpertW& X = a.E[e];
        if (X.gate)
            matvec_x({X.t_gate, X.gate, row_bytes_of(X.t_gate, a.d), a.d, r0, r1, a.xq, c.xx, a.d, a.nb, c.mask[e],
                      c.s.g.data() + (size_t)e * a.nb * a.ff, a.ff});
        matvec_x({X.t_up, X.up, row_bytes_of(X.t_up, a.d), a.d, r0, r1, a.xq, c.xx, a.d, a.nb, c.mask[e],
                  c.s.u.data() + (size_t)e * a.nb * a.ff, a.ff});
    }
}

// silu(g) * u (ungated: relu(u)^2) -> int8 per (expert, token)
void stage_activate(const FfnCtx& c) {
    const FfnArgs& a = c.a;
    const int64_t ff = a.ff;
#pragma omp for schedule(static)
    for (int64_t k = 0; k < (int64_t)c.ne * a.nb; k++) {
        const int e = (int)(k / a.nb), b = (int)(k % a.nb);
        if (!(c.mask[e] >> b & 1)) continue;
        float* g = a.E[e].gate ? c.s.g.data() + ((size_t)e * a.nb + b) * ff : nullptr;
        const float* u = c.s.u.data() + ((size_t)e * a.nb + b) * ff;
        int8_t* aq = c.s.aq.data() + ((size_t)e * a.nb + b) * ff;
        Q8Meta* am = c.s.am.data() + ((size_t)e * a.nb + b) * ff / 32;
        if (a.E[e].gate)
            act_quant(g, u, ff, aq, am);
        else
            relu2_quant(u, ff, aq, am);
        build_xmeta(c.s.am.data() + ((size_t)e * a.nb + b) * ff / 32, ff / 32, c.xa[(size_t)e * a.nb + b]);
    }
}

// y[0..n) += to_fix(wgt * t[0..n)); the vector path (n = 16) does the same steps as det::to_fix
void add_weighted_fix(int64_t* y, const float* t, float wgt, int64_t n) {
    if (n != kChunk) {
        for (int64_t r = 0; r < n; r++) y[r] += det::to_fix(wgt * t[r]);
        return;
    }
    const __m512 v = _mm512_mul_ps(_mm512_mul_ps(_mm512_set1_ps(wgt), _mm512_loadu_ps(t)), _mm512_set1_ps(det::kFix));
    const __m512 c = _mm512_roundscale_ps(
        _mm512_min_ps(_mm512_max_ps(v, _mm512_set1_ps(-0x1p62f)), _mm512_set1_ps(0x1p62f)), _MM_FROUND_TO_NEAREST_INT);
    _mm512_storeu_si512(y, _mm512_add_epi64(_mm512_loadu_si512(y), _mm512_cvtps_epi64(_mm512_castps512_ps256(c))));
    _mm512_storeu_si512(y + 8,
                        _mm512_add_epi64(_mm512_loadu_si512(y + 8), _mm512_cvtps_epi64(_mm512_extractf32x8_ps(c, 1))));
}

// down: chunk of 16 rows of y, summed over all experts (no race)
void stage_down(const FfnCtx& c) {
    const FfnArgs& a = c.a;
#pragma omp for schedule(dynamic, 4)
    for (int64_t k = 0; k < (a.d + kChunk - 1) / kChunk; k++) {
        const int64_t r0 = k * kChunk, r1 = std::min(a.d, r0 + kChunk);
        float t[8 * kChunk];  // up to 8 tokens per pass
        for (int e = 0; e < c.ne; e++) {
            const ExpertW& X = a.E[e];
            matvec_x({X.t_down, X.down, row_bytes_of(X.t_down, a.ff), a.ff, r0, r1,
                      c.s.aq.data() + (size_t)e * a.nb * a.ff, c.xa + (size_t)e * a.nb, a.ff, a.nb, c.mask[e], t - r0,
                      kChunk});
            for (int b = 0; b < a.nb; b++)
                if (c.mask[e] >> b & 1)
                    add_weighted_fix(c.y + b * a.d + r0, t + b * kChunk, a.w[b * c.ne + e], r1 - r0);
        }
    }
}

}  // namespace

void experts_ffn(const FfnArgs& a, float* y, ExpertScratch& s) {
    s.yfix.resize((size_t)a.nb * a.d);
    experts_ffn_fix(a, s.yfix.data(), s);
    for (int64_t i = 0; i < a.nb * a.d; i++) y[i] = det::from_fix(s.yfix[i]);
}

void experts_ffn_fix(const FfnArgs& a, int64_t* y, ExpertScratch& s) {
    const int ne = (int)a.E.size();
    std::fill(y, y + a.nb * a.d, 0);
    if (ne == 0) return;
    check_ffn_args(a);
    s.g.resize(a.E[0].gate ? (size_t)ne * a.nb * a.ff : 0), s.u.resize((size_t)ne * a.nb * a.ff);
    s.aq.resize((size_t)ne * a.nb * a.ff), s.am.resize((size_t)ne * a.nb * a.ff / 32);
    s.xa.resize(sizeof(XMeta) * ne * a.nb);
    FfnCtx c(a, s, y);
    for (int b = 0; b < a.nb; b++) build_xmeta(a.xm + b * a.d / 32, a.d / 32, c.xx[b]);
    for (int e = 0; e < ne; e++)
        for (int b = 0; b < a.nb; b++)
            if (a.w[b * ne + e] != 0.0f) c.mask[e] |= 1u << b;
#pragma omp parallel num_threads(expert_threads())
    {
        stage_gate_up(c);
        stage_activate(c);
        stage_down(c);
    }
}

}  // namespace tona::cpu
