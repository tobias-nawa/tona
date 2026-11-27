// Dequantization against fixed vectors. A block is filled with the byte pattern (i * 37 + 11) & 0xFF and the scales
// are set to fixed fp16 values (d = 0.5, dmin = 0.0625). The expected hash, sum and values are the output of
// model/quant.cpp on 2026-10-28 (IQ2_XXS: 2026-11-11), which matched ggml (llama.cpp b11822, ggml_get_type_traits
// to_float) bit for bit for these blocks; tests/test_dequant_ggml.cpp repeats that comparison on random blocks when
// a llama.cpp build is available. The hand-computed vectors at the end need no reference at all.
#include <bit>
#include <cmath>
#include <cstring>
#include <vector>

#include "model/gguf.h"
#include "model/quant.h"
#include "tests/test_support.h"

namespace {

constexpr uint16_t FP16_HALF = 0x3800, FP16_SIXTEENTH = 0x2C00;

struct Vector {
    uint64_t hash;  // FNV-1a over the little-endian bytes of all floats
    double sum;
    uint32_t type;  // after the 8-byte members: no padding
    float first[4], last;
};

constexpr Vector kVectors[] = {
    {0x7975e9a986d3bd44ull, -88, 8, {42.5f, 61, -48.5f, -30}, -24},
    {0x58611718d5176e15ull, 15, 42, {0, 0, 0, 0}, 0.5f},
    {0x0dd4f045e9cb5108ull, 28804, 12, {167.3125f, -3.1875f, 74.3125f, 151.8125f}, 30.8125f},
    {0x5210f130448214e0ull, 59692, 13, {415.3125f, -3.1875f, 322.3125f, 151.8125f}, 60.8125f},
    {0xdc22ba59d66b0185ull, 2306, 14, {-715.5f, 848, 291.5f, -265}, -30},
    {0x62691c5e3f745950ull, 381.875, 16, {-29.5625f, 29.5625f, 17.1875f, -17.1875f}, -40.3125f},
    {0xfbea90d9f1da2ef4ull, -342, 18, {-6.5f, 4.5f, -3.5f, 1.5f}, -7.5f},
    {0x512b503f3e8b5379ull, -128.5, 20, {-17.5f, 12.5f, 56.5f, -24.5f}, 0.5f},
    {0x50b72b3bbc22f4b4ull, 728, 21, {-40.5f, 121.5f, -175.5f, -94.5f}, 112.5f},
    {0xe7c5b6fcc1f1c28bull, -400.3125, 22, {-72.5625f, 42.1875f, -13.5f, 13.5f}, 39.0625f},
    {0x67d093d31cb47e24ull, -5502.5, 23, {32.5f, -0.5f, -34.5f, 41.5f}, -8},
};

void put_fp16(std::vector<uint8_t>& block, size_t offset, uint16_t h) { std::memcpy(&block[offset], &h, 2); }

std::vector<uint8_t> pattern_block(const tona::GgmlTypeInfo& ti, uint32_t type) {
    std::vector<uint8_t> block(ti.block_bytes);
    for (size_t i = 0; i < block.size(); i++) block[i] = (uint8_t)((i * 37 + 11) & 0xFF);
    if (type == 14) {  // Q6_K keeps d at the end of the block
        put_fp16(block, 208, FP16_HALF);
    } else {
        put_fp16(block, 0, FP16_HALF);
        if (type == 12 || type == 13) put_fp16(block, 2, FP16_SIXTEENTH);
    }
    return block;
}

uint64_t fnv1a(const std::vector<float>& v) {
    uint64_t h = 0xcbf29ce484222325ull;
    for (float f : v) {
        const uint32_t u = std::bit_cast<uint32_t>(f);
        for (int k = 0; k < 4; k++) h = (h ^ ((u >> (8 * k)) & 0xFF)) * 0x100000001b3ull;
    }
    return h;
}

bool same_floats(const float* a, const float* b, size_t n) {
    for (size_t i = 0; i < n; i++)
        if (std::bit_cast<uint32_t>(a[i]) != std::bit_cast<uint32_t>(b[i])) return false;
    return true;
}

void check_vector(const Vector& v) {
    const tona::GgmlTypeInfo* ti = tona::ggml_type_info(v.type);
    EXPECT(ti != nullptr && tona::dequant_supported(v.type));
    if (!ti) return;
    const auto block = pattern_block(*ti, v.type);
    std::vector<float> out(ti->block_elems);
    tona::dequantize_row(v.type, block.data(), out.data(), ti->block_elems);
    double sum = 0;
    for (float f : out) sum += f;
    EXPECT(fnv1a(out) == v.hash && sum == v.sum);
    EXPECT(same_floats(out.data(), v.first, 4) && out.back() == v.last);
}

void test_fixed_blocks() {
    for (const Vector& v : kVectors) check_vector(v);
}

// Several blocks in a row are independent
void test_rows_of_several_blocks() {
    const tona::GgmlTypeInfo* ti = tona::ggml_type_info(8);
    std::vector<uint8_t> row;
    for (int b = 0; b < 3; b++) {
        auto block = pattern_block(*ti, 8);
        put_fp16(block, 0, b == 1 ? FP16_SIXTEENTH : FP16_HALF);
        row.insert(row.end(), block.begin(), block.end());
    }
    std::vector<float> out(96), single(32);
    tona::dequantize_row(8, row.data(), out.data(), 96);
    tona::dequantize_row(8, row.data() + ti->block_bytes, single.data(), 32);
    EXPECT(same_floats(out.data() + 32, single.data(), 32));
    EXPECT(out[32] == out[0] / 8);  // the middle block has d = 1/16 instead of 1/2
}

void test_q8_0_by_hand() {
    std::vector<uint8_t> block(34, 0);
    put_fp16(block, 0, 0x4000);  // d = 2
    for (int j = 0; j < 32; j++) block[2 + j] = (uint8_t)(int8_t)(j - 16);
    float out[32];
    tona::dequantize_row(8, block.data(), out, 32);
    for (int j = 0; j < 32; j++) EXPECT(out[j] == 2.0f * (j - 16));
}

void test_q2_0_by_hand() {
    std::vector<uint8_t> block(18, 0);
    put_fp16(block, 0, 0x3C00);  // d = 1
    block[2] = 0b11100100;       // codes 0, 1, 2, 3 (lowest two bits first) = -1, 0, +1, +2
    float out[64];
    tona::dequantize_row(42, block.data(), out, 64);
    EXPECT(out[0] == -1 && out[1] == 0 && out[2] == 1 && out[3] == 2);
    EXPECT(out[4] == -1);  // the remaining bytes are zero codes
}

// NVFP4 (type 40) with scale bytes and codes chosen so every value is known without running the decoder: the scale byte
// 0x38 is 1.0 halved (0.5), 0x30 is 0.25, 0x00 and 0x7F (NaN) are 0, 0x7E is 448 halved (224); code 7 is 6 (12
// doubled), code 0xF is -6, code 5 is 3, code 2 is 1, code 1 is 0.5
void test_nvfp4_by_hand() {
    std::vector<uint8_t> block(36, 0);
    block[0] = 0x38, block[1] = 0x30, block[2] = 0x00, block[3] = 0x7E;
    block[4 + 0] = 0xF7;   // sub-block 0: weight 0 = code 7, weight 8 = code F
    block[4 + 1] = 0x52;   // weight 1 = code 2, weight 9 = code 5
    block[4 + 8] = 0x7F;   // sub-block 1: weight 16 = code F, weight 24 = code 7
    block[4 + 16] = 0x77;  // sub-block 2 has scale 0: every weight 0
    block[4 + 24] = 0x21;  // sub-block 3: weight 48 = code 1, weight 56 = code 2
    float out[64];
    tona::dequantize_row(40, block.data(), out, 64);
    EXPECT(out[0] == 6 && out[8] == -6 && out[1] == 1 && out[9] == 3);
    EXPECT(out[16] == -3 && out[24] == 3 && out[17] == 0);
    EXPECT(out[32] == 0 && out[40] == 0);
    EXPECT(out[48] == 224.0f && out[56] == 448.0f && out[49] == 0);
    block[3] = 0x7F;  // NaN scale byte reads as 0, not as a NaN
    tona::dequantize_row(40, block.data(), out, 64);
    EXPECT(out[48] == 0 && out[56] == 0);
    block[0] = 0x01;  // subnormal scale: 2^-9 halved
    tona::dequantize_row(40, block.data(), out, 64);
    EXPECT(out[0] == 12 * std::ldexp(1.0f, -10));
    // Two blocks in a row are independent
    std::vector<uint8_t> two(block);
    two.insert(two.end(), block.begin(), block.end());
    float both[128];
    tona::dequantize_row(40, two.data(), both, 128);
    EXPECT(same_floats(both, out, 64) && same_floats(both + 64, out, 64));
}

// TONA_IQ3_XXS_R: two IQ3_XXS blocks repacked into one super-block (index bytes, sign and scale words, scales)
// give the same floats bit for bit
void test_iq3_xxs_repacked() {
    const tona::GgmlTypeInfo* ti = tona::ggml_type_info(18);
    std::vector<uint8_t> two, sb(196);
    for (size_t b = 0; b < 2; b++) {  // size_t: the offsets below are size_t products
        auto block = pattern_block(*ti, 18);
        for (auto& v : block) v = (uint8_t)(v + 91 * b);
        put_fp16(block, 0, b ? FP16_SIXTEENTH : FP16_HALF);
        two.insert(two.end(), block.begin(), block.end());
        std::memcpy(&sb[64 * b], &block[2], 64);
        std::memcpy(&sb[128 + 32 * b], &block[66], 32);
        std::memcpy(&sb[192 + 2 * b], block.data(), 2);
    }
    std::vector<float> want(512), got(512);
    tona::dequantize_row(18, two.data(), want.data(), 512);
    tona::dequantize_row(1018, sb.data(), got.data(), 512);
    EXPECT(tona::dequant_supported(1018) && same_floats(want.data(), got.data(), 512));
    EXPECT(tona::ggml_type_info(1018)->block_elems == 512 && tona::ggml_type_info(1018)->block_bytes == 196);
}

void test_float_types() {
    const float f32[3] = {1.5f, -0.0f, 3e10f};
    float out[3];
    tona::dequantize_row(0, f32, out, 3);
    EXPECT(out[0] == 1.5f && std::signbit(out[1]) && out[2] == 3e10f);

    const uint16_t f16[6] = {0x3C00, 0xC000, 0x0000, 0x7C00, 0x0001, 0x7E00};
    float h[6];
    tona::dequantize_row(1, f16, h, 6);
    EXPECT(h[0] == 1 && h[1] == -2 && h[2] == 0 && std::isinf(h[3]) && h[4] == std::ldexp(1.0f, -24) &&
           std::isnan(h[5]));

    const uint16_t bf[4] = {0x3F80, 0x4000, 0xBF80, 0x7F80};
    float b[4];
    tona::dequantize_row(30, bf, b, 4);
    EXPECT(b[0] == 1 && b[1] == 2 && b[2] == -1 && std::isinf(b[3]));
    EXPECT(tona::fp16_to_fp32(0x3555) == 0.333251953125f && tona::bf16_to_fp32(0x3EAB) == 0.333984375f);
}

void test_unsupported_types() {
    float out[32];
    const uint8_t src[64] = {};
    for (uint32_t t : {2u, 3u, 9u, 15u, 100u}) {
        EXPECT(!tona::dequant_supported(t));
        EXPECT(tona::test::throws([&] { tona::dequantize_row(t, src, out, 32); }));
    }
    EXPECT(tona::dequant_supported(12) && tona::dequant_supported(30));
    EXPECT(tona::ggml_type_info(9999) == nullptr);
}

}  // namespace

int main() {
    return tona::test::run("test_quant", [] {
        test_fixed_blocks();
        test_rows_of_several_blocks();
        test_q8_0_by_hand();
        test_q2_0_by_hand();
        test_nvfp4_by_hand();
        test_iq3_xxs_repacked();
        test_float_types();
        test_unsupported_types();
    });
}
