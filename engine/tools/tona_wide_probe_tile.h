// tona-wide-probe --tile: Q6_K matrices in GGUF order against TONA_Q6_K_T (tile order, docs/use/tona-types.md) on the
// two dense paths of a pass, the int8 row kernel (L lanes per row, as row_q8) and the WMMA kernel (k_wide). Same
// weights, same x, outputs compared bit for bit. Included once into tona_wide_probe.hip after
// tona_wide_probe_kernels.h.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

#include "gpu/matvec_dev.h"
#include "model/gguf.h"
#include "tona_hip_check.h"

namespace {

constexpr int kTileRows = 8;

// Lane of the 8x8 fragment that holds group q (8 weights) of row r8 in a chunk of 32 weights
// (engine/model/quant_tile.cpp)
__host__ __device__ inline int tile_lane(int r8, int q) {
    return ((r8 & 4) << 2) | ((r8 & 3) << 1) | ((q & 2) << 2) | (q & 1);
}

// One 6-bit code of weight w (0 .. k - 1) of row r8 into the tile: low 4 bits and high 2 bits at the entry of its group
void put_code(uint8_t* tile, int64_t k, int r8, int64_t w, int code) {
    const int64_t e = 32 * (w / 32) + tile_lane(r8, (int)(w % 32) / 8);
    const int i = (int)(w % 8);
    uint32_t ql_word;
    uint16_t qh_word;
    std::memcpy(&ql_word, tile + 4 * e, 4);
    std::memcpy(&qh_word, tile + 4 * k + 2 * e, 2);
    ql_word |= (uint32_t)(code & 0xF) << (4 * i);
    qh_word |= (uint16_t)((code >> 4) << (2 * i));
    std::memcpy(tile + 4 * e, &ql_word, 4);
    std::memcpy(tile + 4 * k + 2 * e, &qh_word, 2);
}

// Super-block s (210 bytes, GGUF order) of row r8 into the tile
void pack_superblock(uint8_t* tile, int64_t k, int r8, int64_t s, const uint8_t* b) {
    std::memcpy(tile + 6 * k + k / 2 + 2 * (8 * s + r8), b + 208, 2);
    for (int j = 0; j < 256; j++) {
        const int n = j / 128, low = j % 32, quarter = (j % 128) / 32;
        const int ql = b[64 * n + low + 32 * (quarter & 1)] >> (4 * (quarter >= 2)) & 0xF;
        put_code(tile, k, r8, s * 256 + j, ql | (((b[128 + 32 * n + low] >> (2 * quarter)) & 3) << 4));
    }
    for (int h = 0; h < 16; h++) {  // scale of the weights 16 * (16 s + h) .. + 15
        const int64_t c = 8 * s + h / 2;
        const int q = 2 * (h % 2);
        const int in_chunk = 8 * (r8 >> 2) + 4 * (q >> 1) + (r8 & 3);
        tile[6 * k + 16 * c + in_chunk] = b[192 + h];
    }
}

// Q6_K rows (GGUF order) of k weights as tiles of 8 rows; the same bytes as tools/coder_requant/repack_tile.py
std::vector<uint8_t> pack_q6k_tiles(const uint8_t* src, int64_t rows, int64_t k) {
    const int64_t sb = k / 256, rb = sb * 210;
    std::vector<uint8_t> out((size_t)(rows * rb), 0);
    for (int64_t t = 0; t < rows / kTileRows; t++)
        for (int r8 = 0; r8 < kTileRows; r8++)
            for (int64_t s = 0; s < sb; s++)
                pack_superblock(out.data() + t * kTileRows * rb, k, r8, s, src + (t * kTileRows + r8) * rb + s * 210);
    return out;
}

using namespace tona::gpu;

struct TileArgs {
    const uint8_t* w;
    int64_t rows, row_bytes;
    int in;
    const int8_t* xq;
    const Q8Meta* xm;
    const _Float16* x16;
    int nb;
    float* y;
};

template <bool TILE>
__device__ __forceinline__ void decode_any(const TileArgs& a, int64_t row, int sub, Dec& o) {
    constexpr uint32_t kType = TILE ? kQ6KTile : 14;
    decode_row<kType>(row_at<kType>(a.w, row, a.row_bytes), a.in, sub, Tabs{}, o);
}

// What a lane of the row kernel reads besides the weights: the task, x in LDS and the lanes per row
struct RowCtx {
    const TileArgs& a;
    const int8_t* lx;
    const Q8Meta* lm;
    int L;
};

// One vector b of a round: the unpacked sub-blocks o of this lane times x of the vector, added to s
__device__ __forceinline__ void row_vector(const RowCtx& c, const Dec (&o)[2], int sub0, float& s, int b) {
    const int subs = c.a.in / 32;
    uint4 xa[2], xc[2];
    Q8Meta mm[2];
#pragma unroll
    for (int g = 0; g < 2; g++) {
        if (sub0 + g * c.L >= subs) continue;
        const uint4* xp = reinterpret_cast<const uint4*>(c.lx + (int64_t)b * c.a.in + (int64_t)(sub0 + g * c.L) * 32);
        xa[g] = xp[0], xc[g] = xp[1], mm[g] = c.lm[b * subs + sub0 + g * c.L];
    }
#pragma unroll
    for (int g = 0; g < 2; g++) {
        if (sub0 + g * c.L >= subs) continue;
        const int xi[8] = {(int)xa[g].x, (int)xa[g].y, (int)xa[g].z, (int)xa[g].w,
                           (int)xc[g].x, (int)xc[g].y, (int)xc[g].z, (int)xc[g].w};
        s = apply_q8<14>(o[g], xi, mm[g], s);
    }
}

// The partial sums of one lane j over its sub-blocks (G = 2 per round as row_q8)
template <bool TILE, int NB>
__device__ __forceinline__ void row_partial(const RowCtx& c, int64_t row, int j, float (&s)[NB]) {
    const int subs = c.a.in / 32;
    for (int sub0 = j; sub0 < subs; sub0 += 2 * c.L) {
        Dec o[2];
#pragma unroll
        for (int g = 0; g < 2; g++)
            if (sub0 + g * c.L < subs) decode_any<TILE>(c.a, row, sub0 + g * c.L, o[g]);
#pragma unroll
        for (int b = 0; b < NB; b++)
            if (b < c.a.nb) row_vector(c, o, sub0, s[b], b);
    }
}

// Int8 matvec as row_q8 (L lanes per row, 32 / L rows per wave), x in LDS, 16 waves per block as k_mega
template <bool TILE, int NB>
__global__ void __launch_bounds__(PT) k_row(TileArgs a) {
    extern __shared__ __attribute__((aligned(16))) int8_t lx[];  // 8 * in + 8 * in / 32 * 8 bytes, as k_probe
    const int subs = a.in / 32;
    Q8Meta* lm = reinterpret_cast<Q8Meta*>(lx + (int64_t)a.nb * a.in);
    for (int i = threadIdx.x; i < a.nb * a.in / 16; i += PT)
        reinterpret_cast<uint4*>(lx)[i] = reinterpret_cast<const uint4*>(a.xq)[i];
    for (int i = threadIdx.x; i < a.nb * subs; i += PT) lm[i] = a.xm[i];
    __syncthreads();
    const int lane = threadIdx.x % WAVE;
    const int L = (int)min<int64_t>(WAVE, (subs & -subs)), rpw = WAVE / L, j = lane % L;
    const int64_t wave = (int64_t)blockIdx.x * PW + threadIdx.x / WAVE, W = (int64_t)gridDim.x * PW;
    for (int64_t u = wave; u * rpw < a.rows; u += W) {
        const int64_t row = u * rpw + lane / L;
        float s[NB] = {};
        row_partial<TILE, NB>(RowCtx{a, lx, lm, L}, row, j, s);
#pragma unroll
        for (int b = 0; b < NB; b++)
            for (int off = L / 2; off > 0; off /= 2) s[b] += __shfl_xor(s[b], off, WAVE);
        if (j == 0)
#pragma unroll
            for (int b = 0; b < NB; b++)
                if (b < a.nb) a.y[(int64_t)b * a.rows + row] = s[b];
    }
}

// k_wide of the probe (a wave computes 16 rows with f16 WMMA) with either weight order
template <bool TILE>
__global__ void __launch_bounds__(PT) k_wide_t(TileArgs a) {
    const int lane = threadIdx.x % WAVE, r = lane & 15, g = lane >> 4;
    const int64_t W = (int64_t)gridDim.x * PW;
    const _Float16* xr = a.x16 + (int64_t)min(r, a.nb - 1) * a.in + 8 * (int64_t)g;
    for (int64_t grp = (int64_t)blockIdx.x * PW + threadIdx.x / WAVE; grp * 16 < a.rows; grp += W) {
        const int64_t row = grp * 16 + r, crow = min<int64_t>(row, a.rows - 1);
        float8 acc = {};
        for (int sub = 0; sub < a.in / 32; sub += 2) {
            Dec o;
            decode_any<TILE>(a, crow, sub + g, o);
            uint32_t own[4], snd[4], got[4];
            split_words(o, g, own, snd);
            const float sc4[4] = {o.s_lo, o.s_hi, o.b_lo, o.b_hi};
            float gs[4];
            exchange_lanes(snd, sc4, got, gs);
#pragma unroll
            for (int q = 0; q < 2; q++) {
                const bool mine = (q == 0) == (g == 0);
                acc = mma_subblock<true>(acc, mine ? own : got, mine ? sc4 : gs, xr + (int64_t)(sub + q) * 32);
            }
        }
        if (row < a.rows)
#pragma unroll
            for (int e = 0; e < 8; e++)
                if (8 * g + e < a.nb) a.y[(int64_t)(8 * g + e) * a.rows + row] = acc[e];
    }
}

template <class Go>
double time_us(Go go, int runs) {
    go();
    CHECK(hipDeviceSynchronize());
    std::vector<double> us;
    for (int r = 0; r < runs; r++) {
        const auto t0 = std::chrono::steady_clock::now();
        go();
        CHECK(hipDeviceSynchronize());
        us.push_back(std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - t0).count());
    }
    std::sort(us.begin(), us.end());
    return us[us.size() / 2];
}

void launch_row(bool tile, const TileArgs& a, int blocks) {
    const int nb = a.nb;
    const size_t lds = (size_t)8 * a.in + (size_t)8 * a.in / 32 * sizeof(Q8Meta);
    if (tile) {
        if (nb == 1)
            hipLaunchKernelGGL((k_row<true, 1>), dim3(blocks), dim3(PT), lds, 0, a);
        else if (nb <= 4)
            hipLaunchKernelGGL((k_row<true, 4>), dim3(blocks), dim3(PT), lds, 0, a);
        else
            hipLaunchKernelGGL((k_row<true, 8>), dim3(blocks), dim3(PT), lds, 0, a);
    } else {
        if (nb == 1)
            hipLaunchKernelGGL((k_row<false, 1>), dim3(blocks), dim3(PT), lds, 0, a);
        else if (nb <= 4)
            hipLaunchKernelGGL((k_row<false, 4>), dim3(blocks), dim3(PT), lds, 0, a);
        else
            hipLaunchKernelGGL((k_row<false, 8>), dim3(blocks), dim3(PT), lds, 0, a);
    }
}

void launch_wide_t(bool tile, const TileArgs& a, int blocks) {
    if (tile)
        hipLaunchKernelGGL(k_wide_t<true>, dim3(blocks), dim3(PT), 0, 0, a);
    else
        hipLaunchKernelGGL(k_wide_t<false>, dim3(blocks), dim3(PT), 0, 0, a);
}

// x of 8 tokens: float, f16 and Q8 of its 32-blocks
struct TileX {
    explicit TileX(int64_t in) : q(8 * in), meta(8 * in / 32 * sizeof(Q8Meta)), x16(8 * in * 2), xf(8 * in * 4) {
        std::mt19937 rng(11);
        std::normal_distribution<float> nd(0, 1);
        std::vector<float> x(8 * in), xp(8 * in);
        std::vector<_Float16> h(8 * in);
        for (auto& v : x) v = nd(rng);
        for (int64_t i = 0; i < 8 * in; i++) xp[i] = x[i];
        for (int64_t i = 0; i < 8 * in; i++) h[i] = (_Float16)xp[i];
        CHECK(hipMemcpy(xf.get(), xp.data(), xp.size() * 4, hipMemcpyHostToDevice));
        CHECK(hipMemcpy(x16.get(), h.data(), h.size() * 2, hipMemcpyHostToDevice));
        for (int b = 0; b < 8; b++)
            quantize_q8(xf.get<float>() + b * in, in,
                        Q8Vec{q.get<int8_t>() + b * in, meta.get<Q8Meta>() + b * in / 32});
    }
    tona::tools::DeviceBuffer q, meta, x16, xf;
};

struct TileSet {
    int64_t in, rows, rb;
    std::vector<uint8_t> gguf, tile;
};

// The Q6_K blk.* matrices with row length `in`, joined up to cap bytes, rows a multiple of 8
TileSet collect_q6k(const tona::Gguf& g, int64_t in, int64_t cap) {
    TileSet s{in, 0, in / 256 * 210, {}, {}};
    for (const auto& t : g.tensors()) {
        if (t.type != 14 || t.dims.size() != 2 || (int64_t)t.dims[0] != in || t.name.rfind("blk.", 0)) continue;
        const int64_t take = std::min<int64_t>((int64_t)t.bytes, cap - (int64_t)s.gguf.size()) / s.rb / 8 * 8 * s.rb;
        if (take <= 0) break;
        s.gguf.insert(s.gguf.end(), t.data, t.data + take);
    }
    s.rows = (int64_t)s.gguf.size() / s.rb;
    s.tile = pack_q6k_tiles(s.gguf.data(), s.rows, in);
    return s;
}

struct TileRun {
    const TileSet& s;
    int blocks, runs;
};

void tile_one_width(const TileRun& r, const tona::tools::DeviceBuffer& dg, const tona::tools::DeviceBuffer& dt,
                    const TileX& xg, const TileX& xt, int nb) {
    using tona::tools::DeviceBuffer;
    DeviceBuffer yg(8 * r.s.rows * 4), yt(8 * r.s.rows * 4);
    const TileArgs ag{
        dg.get<uint8_t>(),      r.s.rows, r.s.rb,         (int)r.s.in, xg.q.get<int8_t>(), xg.meta.get<Q8Meta>(),
        xg.x16.get<_Float16>(), nb,       yg.get<float>()};
    TileArgs at = ag;
    at.w = dt.get<uint8_t>(), at.xq = xt.q.get<int8_t>(), at.xm = xt.meta.get<Q8Meta>();
    at.x16 = xt.x16.get<_Float16>(), at.y = yt.get<float>();
    const double mb = (double)r.s.gguf.size() / 1e6;
    for (int path = 0; path < 2; path++) {
        const bool wide = path == 1;
        auto go = [&](bool tile) {
            wide ? launch_wide_t(tile, tile ? at : ag, r.blocks) : launch_row(tile, tile ? at : ag, r.blocks);
        };
        const double ug = time_us([&] { go(false); }, r.runs), ut = time_us([&] { go(true); }, r.runs);
        std::vector<float> a(nb * r.s.rows), b(nb * r.s.rows);
        CHECK(hipMemcpy(a.data(), yg.get(), a.size() * 4, hipMemcpyDeviceToHost));
        CHECK(hipMemcpy(b.data(), yt.get(), b.size() * 4, hipMemcpyDeviceToHost));
        const bool same = std::memcmp(a.data(), b.data(), a.size() * 4) == 0;
        double md = 0, mr = 0;
        for (size_t i = 0; i < a.size(); i++)
            md = std::max(md, (double)std::fabs(a[i] - b[i])), mr = std::max(mr, (double)std::fabs(a[i]));
        printf("| %lld | %lld | %.0f | %s | %d | %.0f | %.0f | %.0f | %.0f | %.2f | %s (%.1e) |\n", (long long)r.s.in,
               (long long)r.s.rows, mb, wide ? "wide" : "row", nb, ug, mb / ug * 1e3, ut, mb / ut * 1e3, ug / ut,
               same ? "bit-identical" : "differs", md / mr);
    }
}

int run_tile(const tona::Gguf& g, int64_t cap, int runs, int cus) {
    using tona::tools::DeviceBuffer;
    printf(
        "| k | Rows | MB | Path | nb | GGUF us | GB/s | tile us | GB/s | speed-up | output "
        "|\n|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|\n");
    for (int64_t in : {2560, 6144}) {
        const TileSet s = collect_q6k(g, in, cap);
        if (!s.rows) continue;
        {  // the packer must agree with the engine's reader of the type (dequantize_rows)
            std::vector<float> a(4 * in), b(4 * in);
            for (int64_t row : {int64_t{0}, int64_t{9}, s.rows - 1}) {
                tona::dequantize_rows(14, s.gguf.data(), row, 1, in, a.data());
                tona::dequantize_rows(1019, s.tile.data(), row, 1, in, b.data());
                if (std::memcmp(a.data(), b.data(), (size_t)in * 4) != 0)
                    throw std::runtime_error("tile packer differs from dequantize_rows");
            }
        }
        DeviceBuffer dg(s.gguf.size() + kMatvecReadPad), dt(s.tile.size() + kMatvecReadPad);
        CHECK(hipMemcpy(dg.get(), s.gguf.data(), s.gguf.size(), hipMemcpyHostToDevice));
        CHECK(hipMemcpy(dt.get(), s.tile.data(), s.tile.size(), hipMemcpyHostToDevice));
        const TileX xg(in), xt(in);
        for (int nb = 1; nb <= 8; nb++) tile_one_width({s, cus, runs}, dg, dt, xg, xt, nb);
    }
    return 0;
}

}  // namespace
