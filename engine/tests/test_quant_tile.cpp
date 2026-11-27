// dequantize_rows against dequantize_row for every type, and TONA_Q6_K_T (docs/use/tona-types.md) against the Q6_K
// rows it was packed from, bit for bit.
#include <bit>
#include <cstring>
#include <random>
#include <vector>

#include "model/gguf.h"
#include "model/quant.h"
#include "tests/test_support.h"

namespace {

constexpr uint16_t kFp16Half = 0x3800;

bool same_floats(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); i++)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}

// rows x k weights of type in random bytes; the fp16 scales of the K-quants made finite so the floats compare
std::vector<uint8_t> random_rows(uint32_t type, int64_t rows, int64_t k, uint32_t seed) {
    const tona::GgmlTypeInfo* ti = tona::ggml_type_info(type);
    std::vector<uint8_t> b(rows * k / ti->block_elems * ti->block_bytes);
    std::mt19937 rng(seed);
    for (auto& v : b) v = (uint8_t)rng();
    if (type == 14)
        for (size_t at = 208; at < b.size(); at += 210) std::memcpy(&b[at], &kFp16Half, 2);
    return b;
}

// One Q6_K weight: its 6-bit code, the scale of its 16 weights and d of its block (ggml's layout)
struct Q6Weight {
    uint8_t q;
    int8_t sc;
    uint16_t d;
};

Q6Weight q6_weight(const uint8_t* row, int64_t w) {
    const uint8_t* b = row + (w / 256) * 210;
    const int64_t j = w % 256, n = j / 128, l = j % 32, quarter = (j % 128) / 32;
    const uint8_t ql = b[64 * n + l + 32 * (quarter & 1)], qh = b[128 + 32 * n + l];
    Q6Weight r{};
    r.q = (uint8_t)((quarter < 2 ? ql & 0xF : ql >> 4) | (((qh >> (2 * quarter)) & 3) << 4));
    r.sc = (int8_t)b[192 + j / 16];
    std::memcpy(&r.d, b + 208, 2);
    return r;
}

int tile_lane(int r8, int q) { return ((r8 & 4) << 2) | ((r8 & 3) << 1) | ((q & 2) << 2) | (q & 1); }

// The layout of docs/use/tona-types.md, written independently of the decoder: entry e = 32 c + lane
std::vector<uint8_t> pack_tiles(const std::vector<uint8_t>& q6, int64_t rows, int64_t k) {
    const int64_t rb = k / 256 * 210;
    std::vector<uint8_t> out(q6.size(), 0);
    for (int64_t t = 0; t < rows / 8; t++) {
        uint8_t* tile = out.data() + t * 8 * rb;
        for (int r8 = 0; r8 < 8; r8++)
            for (int64_t c = 0; c < k / 32; c++)
                for (int q = 0; q < 4; q++) {
                    const int64_t e = 32 * c + tile_lane(r8, q);
                    uint32_t ql = 0;
                    uint16_t qh = 0;
                    Q6Weight wt{};
                    for (int i = 0; i < 8; i++) {
                        wt = q6_weight(q6.data() + (8 * t + r8) * rb, 32 * c + 8 * (int64_t)q + i);
                        ql |= (uint32_t)(wt.q & 0xF) << (4 * i);
                        qh |= (uint16_t)((wt.q >> 4) << (2 * i));
                    }
                    std::memcpy(tile + 4 * e, &ql, 4);
                    std::memcpy(tile + 4 * k + 2 * e, &qh, 2);
                    tile[6 * k + e / 2] = (uint8_t)wt.sc;
                    std::memcpy(tile + 6 * k + k / 2 + 2 * ((c / 8) * 8 + r8), &wt.d, 2);
                }
    }
    return out;
}

void test_tile_type_matches_q6_k() {
    const int64_t rows = 16, k = 512;
    const auto q6 = random_rows(14, rows, k, 7);
    const auto tiles = pack_tiles(q6, rows, k);
    std::vector<float> want(rows * k), got(rows * k), part(3 * k);
    tona::dequantize_rows(14, q6.data(), 0, rows, k, want.data());
    tona::dequantize_rows(1019, tiles.data(), 0, rows, k, got.data());
    EXPECT(same_floats(want, got));
    tona::dequantize_rows(1019, tiles.data(), 6, 3, k, part.data());  // across the tile border
    EXPECT(same_floats(part, std::vector<float>(want.begin() + 6 * k, want.begin() + 9 * k)));
    EXPECT(!tona::row_contiguous(1019) && tona::row_contiguous(14) && tona::dequant_supported(1019));
    EXPECT(tona::test::throws([&] { tona::dequantize_row(1019, tiles.data(), got.data(), k); }));
    EXPECT(tona::ggml_type_info(1019)->block_elems == 256 && tona::ggml_type_info(1019)->block_bytes == 210);
}

// The production tiler writes the same bytes as the independent packer, in two ranges as threads would
void test_tiler_matches_the_packer() {
    const int64_t rows = 24, k = 768;
    const auto q6 = random_rows(14, rows, k, 11);
    const auto want = pack_tiles(q6, rows, k);
    std::vector<uint8_t> got(q6.size(), 0xAB);
    tona::tile_q6_k(q6.data(), k, 0, 1, got.data());
    tona::tile_q6_k(q6.data(), k, 1, rows / 8, got.data());
    EXPECT(got == want);
}

void test_rows_equal_the_row_loop_for_every_type() {
    for (uint32_t type : {0u, 1u, 8u, 12u, 13u, 14u, 16u, 18u, 20u, 21u, 22u, 23u, 30u, 42u, 1018u}) {
        const tona::GgmlTypeInfo* ti = tona::ggml_type_info(type);
        const int64_t rows = 5, k = 2 * std::max<int64_t>(ti->block_elems, 256);
        const auto b = random_rows(type, rows, k, type);
        const int64_t rb = k / ti->block_elems * ti->block_bytes;
        std::vector<float> want(3 * k), got(3 * k);
        for (int64_t r = 0; r < 3; r++) tona::dequantize_row(type, b.data() + (1 + r) * rb, want.data() + r * k, k);
        tona::dequantize_rows(type, b.data(), 1, 3, k, got.data());
        EXPECT(same_floats(want, got));
    }
}

}  // namespace

int main() {
    return tona::test::run("test_quant_tile", [] {
        test_tile_type_matches_q6_k();
        test_tiler_matches_the_packer();
        test_rows_equal_the_row_loop_for_every_type();
    });
}
