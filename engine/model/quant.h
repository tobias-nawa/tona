// CPU reference for dequantization: GGUF block to float, bit-exact with ggml (checked in tests/).
// Slow and simple; it is the oracle for the GPU kernels, not part of the hot path.
#pragma once

#include <cstdint>

namespace tona {

// True if this ggml type is supported
bool dequant_supported(uint32_t type);

// n elements (a multiple of the block size) from src to dst; throws for an unsupported type and for a type whose rows
// are not contiguous (row_contiguous)
void dequantize_row(uint32_t type, const void* src, float* dst, int64_t n);

// False for Tona's tile types (docs/use/tona-types.md), which interleave 8 rows: read those only with dequantize_rows
bool row_contiguous(uint32_t type);

// Rows row0 .. row0 + nrows of a matrix with rows of k weights at base, to dst (nrows * k floats); every type
void dequantize_rows(uint32_t type, const void* base, int64_t row0, int64_t nrows, int64_t k, float* dst);

// Packs tiles t0 .. t1 (8 rows each) of a Q6_K matrix with rows of k weights at src into TONA_Q6_K_T at dst, same
// offsets (a tile has as many bytes as its 8 Q6_K rows). Tiles are independent: callers split the range across threads.
void tile_q6_k(const uint8_t* src, int64_t k, int64_t t0, int64_t t1, uint8_t* dst);

float fp16_to_fp32(uint16_t h);
uint16_t fp32_to_fp16(float x);
float bf16_to_fp32(uint16_t h);

}  // namespace tona
