// Rows of Tona's tile types (docs/use/tona-types.md) and dequantize_rows, the row access that works for every type.
#include <cstring>
#include <format>
#include <stdexcept>

#include "model/gguf.h"
#include "model/quant.h"

namespace tona {

namespace {

constexpr uint32_t kQ6KTile = 1019;  // TONA_Q6_K_T
constexpr int64_t kTileRows = 8;

// The lane of the 8x8 MMA fragment that holds group q (8 weights) of row r8 in a 32-weight chunk: the inverse of
// mma_fm / mma_fn in the Metal kernels (lane bit 4 is row bit 2, lane bits 1-2 are row bits 0-1, lane bit 3 is group
// bit 1, lane bit 0 is group bit 0)
int tile_lane(int r8, int q) { return ((r8 & 4) << 2) | ((r8 & 3) << 1) | ((q & 2) << 2) | (q & 1); }

// Row r8 of a TONA_Q6_K_T tile of 8 rows of k weights: entry e = 32 c + lane holds 8 weights; low nibbles as one
// 32-bit word at 4e, high bit pairs as one 16-bit word at 4k + 2e, the scale of its 16 weights at 6k + e / 2, and d
// of row r8's super-block s at 6k + k / 2 + 2 (8 s + r8). The same arithmetic as deq_q6_k, so the same bits.
void deq_q6_k_t_row(const uint8_t* tile, int64_t k, int r8, float* y) {
    const uint8_t* qh_at = tile + 4 * k;
    const auto* sc = reinterpret_cast<const int8_t*>(tile + 6 * k);
    const uint8_t* d_at = tile + 6 * k + k / 2;
    for (int64_t c = 0; c < k / 32; c++) {
        uint16_t dh;
        std::memcpy(&dh, d_at + 2 * ((c / 8) * kTileRows + r8), 2);
        const float d = fp16_to_fp32(dh);
        for (int q = 0; q < 4; q++) {
            const int64_t e = 32 * c + tile_lane(r8, q);
            uint32_t ql;
            uint16_t qh;
            std::memcpy(&ql, tile + 4 * e, 4);
            std::memcpy(&qh, qh_at + 2 * e, 2);
            const int64_t si = e / 2;
            const float ds = d * sc[si];
            for (int i = 0; i < 8; i++) {
                const int8_t v = (int8_t)(((ql >> (4 * i)) & 0xF) | (((qh >> (2 * i)) & 3) << 4)) - 32;
                y[32 * c + 8 * (int64_t)q + i] = ds * v;
            }
        }
    }
}

// The 256 six-bit codes of a Q6_K block (ggml's ql/qh layout), one byte each: decoding once per block instead of once
// per weight is what keeps the tiling at load time short (research/measurements/2026-11-13-pc-q6k-tile-upload.md)
void q6_codes(const uint8_t* b, uint8_t* codes) {
    for (int n = 0; n < 2; n++)
        for (int l = 0; l < 32; l++) {
            const uint8_t lo0 = b[64 * n + l], lo1 = b[64 * n + 32 + l], hi = b[128 + 32 * n + l];
            uint8_t* c = codes + 128 * (int64_t)n + l;
            c[0] = (uint8_t)((lo0 & 0xF) | ((hi & 3) << 4));
            c[32] = (uint8_t)((lo1 & 0xF) | (((hi >> 2) & 3) << 4));
            c[64] = (uint8_t)((lo0 >> 4) | (((hi >> 4) & 3) << 4));
            c[96] = (uint8_t)((lo1 >> 4) | ((hi >> 6) << 4));
        }
}

// One tile: the inverse of deq_q6_k_t_row, entry by entry
void tile_one(const uint8_t* rows, int64_t rb, int64_t k, uint8_t* tile) {
    uint8_t codes[256];
    for (int r8 = 0; r8 < kTileRows; r8++)
        for (int64_t sb = 0; sb < k / 256; sb++) {
            const uint8_t* b = rows + r8 * rb + sb * 210;
            q6_codes(b, codes);
            std::memcpy(tile + 6 * k + k / 2 + 2 * (sb * kTileRows + r8), b + 208, 2);
            for (int cc = 0; cc < 8; cc++)
                for (int q = 0; q < 4; q++) {
                    const int64_t c = sb * 8 + cc, e = 32 * c + tile_lane(r8, q);
                    const uint8_t* w = codes + 32 * (int64_t)cc + 8 * (int64_t)q;
                    uint32_t ql = 0;
                    uint16_t qh = 0;
                    for (int i = 0; i < 8; i++) {
                        ql |= (uint32_t)(w[i] & 0xF) << (4 * i);
                        qh |= (uint16_t)((w[i] >> 4) << (2 * i));
                    }
                    std::memcpy(tile + 4 * e, &ql, 4);
                    std::memcpy(tile + 4 * k + 2 * e, &qh, 2);
                    if (!(q & 1)) tile[6 * k + e / 2] = b[192 + 2 * cc + (q >> 1)];  // one scale per pair of entries
                }
        }
}

}  // namespace

void tile_q6_k(const uint8_t* src, int64_t k, int64_t t0, int64_t t1, uint8_t* dst) {
    const int64_t rb = k / 256 * 210, tb = kTileRows * rb;
    for (int64_t t = t0; t < t1; t++) tile_one(src + t * tb, rb, k, dst + t * tb);
}

bool row_contiguous(uint32_t type) { return type != kQ6KTile; }

void dequantize_rows(uint32_t type, const void* base, int64_t row0, int64_t nrows, int64_t k, float* dst) {
    const GgmlTypeInfo* ti = ggml_type_info(type);
    if (!ti || k % ti->block_elems) throw std::runtime_error(std::format("dequantize_rows: type {}, k {}", type, k));
    const int64_t rb = k / ti->block_elems * ti->block_bytes;
    const auto* b = static_cast<const uint8_t*>(base);
    for (int64_t r = 0; r < nrows; r++) {
        const int64_t row = row0 + r;
        if (row_contiguous(type))
            dequantize_row(type, b + row * rb, dst + r * k, k);
        else
            deq_q6_k_t_row(b + (row / kTileRows) * kTileRows * rb, k, (int)(row % kTileRows), dst + r * k);
    }
}

}  // namespace tona
