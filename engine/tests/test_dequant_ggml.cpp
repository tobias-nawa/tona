// Dequantization against ggml (llama.cpp b11822): random blocks of all types and, with TONA_MODEL=<part 1 of the GGUF>,
// real rows of every tensor. Compared bit by bit.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "backend/env.h"
#include "ggml.h"
#include "model/gguf.h"
#include "model/quant.h"

static int g_fail = 0;

static uint32_t bits(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
}

static bool compare(const char* what, uint32_t type, const void* src, int64_t n) {
    std::vector<float> a(n), b(n);
    tona::dequantize_row(type, src, a.data(), n);
    auto to_float = ggml_get_type_traits((ggml_type)type)->to_float;
    if (to_float)
        to_float(src, b.data(), n);
    else if (type == GGML_TYPE_F32)
        std::memcpy(b.data(), src, n * 4);  // ggml has no to_float for F32
    else {
        fprintf(stderr, "%s: ggml has no to_float for type %u\n", what, type);
        g_fail++;
        return false;
    }
    for (int64_t i = 0; i < n; i++) {
        if (bits(a[i]) != bits(b[i])) {
            fprintf(stderr, "%s (%s): element %lld: ours %.9g, ggml %.9g\n", what, tona::ggml_type_info(type)->name,
                    (long long)i, a[i], b[i]);
            g_fail++;
            return false;
        }
    }
    return true;
}

// Position of the fp16 scales in the block, so random data gives finite values
static std::vector<int> scale_offsets(uint32_t type) {
    switch (type) {
        case 12:
        case 13:
            return {0, 2};  // Q4_K, Q5_K: d, dmin
        case 14:
            return {208};  // Q6_K: d at the end
        case 0:
        case 1:
        case 30:
            return {};
        default:
            return {0};
    }
}

static void random_float_block(uint32_t t, std::vector<uint8_t>& buf, int n, std::mt19937_64& rng) {
    // Floating point: random finite values instead of random bits (no NaN)
    std::uniform_real_distribution<float> u(-4, 4);
    for (int i = 0; i < n; i++) {
        float f = u(rng);
        if (t == 0)
            std::memcpy(&buf[(size_t)i * 4], &f, 4);
        else if (t == 1) {
            _Float16 h = (_Float16)f;
            std::memcpy(&buf[(size_t)i * 2], &h, 2);
        } else {
            uint32_t v;
            std::memcpy(&v, &f, 4);
            uint16_t h = v >> 16;
            std::memcpy(&buf[(size_t)i * 2], &h, 2);
        }
    }
}

static void check_random_blocks(std::mt19937_64& rng) {
    const uint32_t types[] = {0, 1, 8, 12, 13, 14, 16, 18, 20, 21, 22, 23, 30, 40, 42};
    for (uint32_t t : types) {
        const tona::GgmlTypeInfo* ti = tona::ggml_type_info(t);
        const int blocks = 64;
        std::vector<uint8_t> buf((size_t)blocks * ti->block_bytes);
        for (auto& x : buf) x = (uint8_t)rng();
        if (t == 0 || t == 1 || t == 30) random_float_block(t, buf, blocks * (int)ti->block_elems, rng);
        for (int b = 0; b < blocks; b++)
            for (int off : scale_offsets(t)) {
                _Float16 h = (_Float16)std::uniform_real_distribution<float>(-2, 2)(rng);
                std::memcpy(&buf[(size_t)b * ti->block_bytes + off], &h, 2);
            }
        compare("random", t, buf.data(), (int64_t)blocks * ti->block_elems);
    }
}

static void check_model_rows(const char* path, std::mt19937_64& rng) {
    int rows = 0, tensors = 0;
    tona::Gguf g(path);
    for (const auto& t : g.tensors()) {
        if (!tona::dequant_supported(t.type)) {
            fprintf(stderr, "not supported: %s (%u)\n", t.name.c_str(), t.type);
            g_fail++;
            continue;
        }
        const tona::GgmlTypeInfo* ti = tona::ggml_type_info(t.type);
        const uint64_t row_elems = t.dims[0], row_bytes = row_elems / ti->block_elems * ti->block_bytes;
        const uint64_t n_rows = t.elements() / row_elems;
        // First, middle, last and a random row; the n-gram table has 320 million rows
        for (uint64_t r : {uint64_t(0), n_rows / 2, n_rows - 1, (uint64_t)(rng() % n_rows)}) {
            compare(t.name.c_str(), t.type, t.data + r * row_bytes, (int64_t)row_elems);
            rows++;
        }
        tensors++;
    }
    printf("model: %d tensors, %d rows compared\n", tensors, rows);
}

int main() {
    std::mt19937_64 rng(42);
    check_random_blocks(rng);
    if (const char* path = tona::env::text("TONA_MODEL"))
        check_model_rows(path, rng);
    else
        printf("TONA_MODEL not set: random blocks only\n");
    if (g_fail) {
        fprintf(stderr, "%d mismatches\n", g_fail);
        return 1;
    }
    printf("test_dequant_ggml: ok\n");
    return 0;
}
