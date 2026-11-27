// Prefill attention over the whole prefix on WMMA (prefill_fattn.h): the tile of the decode step (fattn_tile.h) with
// rows = tokens x heads of one KV head and a causal mask per row.
#include "backend/env.h"
#include "gpu/fattn_tile.h"
#include "gpu/fattn_wave.h"
#include "gpu/prefill_fattn.h"

namespace tona::gpu {

namespace {

// Block per (MT tiles of 16 / G tokens each, KV head). Row r of tile m: token (m * TT + r / G), head r % G of the
// group. The MT tiles share every staged chunk of K and V (the staging, its global loads and the transposed LDS writes,
// was about 15 % of the attention cost at two tokens per tile, research/measurements/2026-11-19-ornith-opt2.md); each
// tile runs the same scores, softmax and accumulate as the single tile did, over the same chunks in the same order, so
// the result is bit-identical to MT = 1. A tile whose rows all lie before the chunk skips it: it would have added exact
// zeros (and the single tile never reached those chunks).
template <int HD, int G, int MT>
__global__ void __launch_bounds__(fa::THREADS) k_fattn_prefill(FattnPrefillArgs a) {
    constexpr int TT = 16 / G;
    using Geo = fa::Geo<HD>;
    // K, V^T, then per tile the running softmax state, then the scores and probabilities of the tile in turn
    constexpr int SHARED = Geo::KS_BYTES + Geo::VT_BYTES + fa::PT_BYTES + Geo::PS_BYTES + 16 * 4;
    __shared__ __attribute__((aligned(16))) float sm[(SHARED + MT * 32 * 4) / 4];
    const int tile = blockIdx.x / a.nkv, kvh = blockIdx.x % a.nkv, tb = tile * TT * MT;
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    fa::Lds<HD> l[MT];
    fa::half8 qa[MT][Geo::STEPS] = {};
    fa::float8 acc[MT] = {};
    float* const state = sm + SHARED / 4;
#pragma unroll
    for (int m = 0; m < MT; m++) {
        l[m].Ks = reinterpret_cast<_Float16*>(sm);
        l[m].Vt = l[m].Ks + fa::CC * Geo::LQ;
        l[m].pt = reinterpret_cast<float*>(l[m].Vt + HD * fa::CC);
        l[m].Ps = reinterpret_cast<_Float16*>(l[m].pt + fa::PT_BYTES / 4);
        l[m].arow = reinterpret_cast<float*>(l[m].Ps + 16 * fa::CC);
        l[m].mrow = state + m * 32, l[m].lrow = l[m].mrow + 16;
        const int tok = tb + m * TT + r16 / G;
        const float* qr = tok < a.n_tokens ? a.q + ((int64_t)tok * a.nh + kvh * G + r16 % G) * HD : nullptr;
        fa::load_q<HD>(qr, a.scale, qa[m], lane);
        fa::reset_rows(l[m].mrow, l[m].lrow);
    }
    const int64_t n_keys = a.pos0 + std::min(tb + TT * MT, a.n_tokens);
    const fa::KvHead kv{a.kv, a.row, a.nkv, kvh, a.kv_cold, a.kv_hot};
    for (int64_t c0 = 0; c0 < n_keys; c0 += fa::CC) {
        const int cn = (int)std::min<int64_t>(fa::CC, n_keys - c0);
        lds_sync();
        fa::stage<HD>(kv, l[0], c0, cn);
        lds_sync();
#pragma unroll
        for (int m = 0; m < MT; m++) {
            const int t0 = tb + m * TT;
            if (t0 >= a.n_tokens || c0 > a.pos0 + t0 + TT - 1) continue;  // block-uniform: no row of the tile sees it
            const int64_t row_pos = a.pos0 + t0 + w / G;
            const int valid =
                t0 + w / G < a.n_tokens ? (int)std::min<int64_t>(std::max<int64_t>(row_pos - c0 + 1, 0), cn) : 0;
            fa::scores<HD>(qa[m], l[m], lane);
            lds_sync();
            fa::softmax<HD>(l[m], valid, lane);
            lds_sync();
            fa::accumulate<HD>(acc[m], l[m], lane);
        }
    }
    if (w >= Geo::OUT_WAVES) return;
#pragma unroll
    for (int m = 0; m < MT; m++) {
        const int t0 = tb + m * TT;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int r = kh + e, token = t0 + r / G;
            if (token >= a.n_tokens) continue;
            const int64_t i = ((int64_t)token * a.nh + kvh * G + r % G) * HD + w * 16 + r16;
            const float ov = acc[m][e] / l[m].lrow[r];
            a.o[i] = (_Float16)(a.gate ? ov * sigm(a.gate[i]) : ov);
        }
    }
}

// Block per (W waves of one tile each = W * 16 / G tokens, KV head), the heaviest tokens first: the causal work grows
// with the token, and the last blocks then are the short ones. A chunk of CCB cells is staged for all waves; each wave
// runs the 16-cell steps of its own tile in registers (fattn_wave.h) and skips the steps no row of it sees (it would
// add exact zeros). Bit-identical to k_fattn_prefill.
template <int HD, int G, int W, int CCB>
__global__ void __launch_bounds__(W * 32) k_fattn_prefill_wave(FattnPrefillArgs a) {
    constexpr int TT = 16 / G, THREADS = W * 32;
    using Geo = fw::Geo<HD, CCB>;
    __shared__ __attribute__((aligned(16))) _Float16 sm[Geo::BYTES / 2];
    _Float16* const Ks = sm;
    _Float16* const Vt = sm + CCB * Geo::LQ;
    const int ntile = (a.n_tokens + TT * W - 1) / (TT * W);
    const int tile = ntile - 1 - blockIdx.x / a.nkv, kvh = blockIdx.x % a.nkv, tb = tile * TT * W;
    const int lane = threadIdx.x % 32, w = threadIdx.x / 32, r16 = lane & 15, kh = (lane >> 4) * 8;
    const int t0 = tb + w * TT, tok = t0 + r16 / G;
    const bool row_ok = tok < a.n_tokens;
    fw::State<HD> s;
    fw::load_q<HD>(s, row_ok ? a.q + ((int64_t)tok * a.nh + kvh * G + r16 % G) * HD : nullptr, a.scale, lane);
    const int64_t n_keys = a.pos0 + std::min(tb + TT * W, a.n_tokens);
    const int64_t wave_last = t0 < a.n_tokens ? a.pos0 + std::min(t0 + TT, a.n_tokens) - 1 : -1;
    const int64_t row_pos = a.pos0 + tok;
    const fa::KvHead kv{a.kv, a.row, a.nkv, kvh, a.kv_cold, a.kv_hot};
    for (int64_t c0 = 0; c0 < n_keys; c0 += CCB) {
        const int cn = (int)std::min<int64_t>(CCB, n_keys - c0);
        lds_sync();
        fw::stage<HD, CCB, THREADS>(kv, Ks, Vt, c0, cn);
        lds_sync();
#pragma unroll
        for (int sub = 0; sub < CCB / 16; sub++) {
            const int64_t cb = c0 + sub * 16;
            if (cb >= c0 + cn || cb > wave_last) continue;  // wave-uniform: no row of the tile sees the step
            const int valid = row_ok ? (int)std::min<int64_t>(std::max<int64_t>(row_pos - cb + 1, 0), 16) : 0;
            float m = s.m, l = s.l, al;
            const fa::half8 pb = fw::softmax(fw::scores<HD, CCB>(s, Ks, sub, lane), valid, m, l, al, lane);
            s.m = m, s.l = l;
            fw::accumulate<HD, CCB>(s, Vt, sub, pb, al, lane);
        }
    }
    if (!row_ok) return;
    const int64_t base = ((int64_t)tok * a.nh + kvh * G + r16 % G) * HD + kh;
#pragma unroll
    for (int d = 0; d < HD / 16; d++)
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const int64_t i = base + 16 * d + e;
            const float ov = s.acc[d][e] / s.l;
            a.o[i] = (_Float16)(a.gate ? ov * sigm(a.gate[i]) : ov);
        }
}

// The wave kernel: W tiles per block, CCB cells per staged chunk
template <int HD, int G>
void launch_wave(const FattnPrefillArgs& a, hipStream_t st) {
    constexpr int TT = 16 / G, W = 16, CCB = HD == 128 ? 64 : 32;
    k_fattn_prefill_wave<HD, G, W, CCB><<<(a.n_tokens + TT * W - 1) / (TT * W) * a.nkv, W * 32, 0, st>>>(a);
}

// Tokens per block: 8 at most (MT tiles), the point where the staging stops being the cost (TONA_FATTN_PF_MT1=1 and
// FattnPrefillKernel::single_tile restore the single tile for a comparison)
template <int HD, int G>
void launch_tiles(const FattnPrefillArgs& a, hipStream_t st, bool single) {
    constexpr int TT = 16 / G, MT = TT >= 8 ? 1 : 8 / TT;
    const auto blocks = [&](int mt) { return (a.n_tokens + TT * mt - 1) / (TT * mt) * a.nkv; };
    if (single)
        k_fattn_prefill<HD, G, 1><<<blocks(1), fa::THREADS, 0, st>>>(a);
    else
        k_fattn_prefill<HD, G, MT><<<blocks(MT), fa::THREADS, 0, st>>>(a);
}

template <int HD, int G>
void launch(const FattnPrefillArgs& a, hipStream_t st, FattnPrefillKernel k) {
    static const bool env_single = env::flag("TONA_FATTN_PF_MT1"), env_old = env::flag("TONA_FATTN_PF_OLD");
    if (k == FattnPrefillKernel::automatic)
        k = env_single ? FattnPrefillKernel::single_tile
            : env_old  ? FattnPrefillKernel::multi_tile
                       : FattnPrefillKernel::wave;
    if (k == FattnPrefillKernel::wave)
        launch_wave<HD, G>(a, st);
    else
        launch_tiles<HD, G>(a, st, k == FattnPrefillKernel::single_tile);
}

template <int HD>
bool launch_group(const FattnPrefillArgs& a, hipStream_t st, FattnPrefillKernel k) {
    switch (a.nh / a.nkv) {
        case 1:
            launch<HD, 1>(a, st, k);
            return true;
        case 2:
            launch<HD, 2>(a, st, k);
            return true;
        case 4:
            launch<HD, 4>(a, st, k);
            return true;
        case 8:
            launch<HD, 8>(a, st, k);
            return true;
        case 16:
            launch<HD, 16>(a, st, k);
            return true;
        default:
            return false;
    }
}

}  // namespace

bool launch_fattn_prefill(const FattnPrefillArgs& a, hipStream_t st, FattnPrefillKernel k) {
    if (a.nkv <= 0 || a.nh % a.nkv != 0) return false;
    return a.hd == 128 ? launch_group<128>(a, st, k) : a.hd == 256 ? launch_group<256>(a, st, k) : false;
}

}  // namespace tona::gpu
