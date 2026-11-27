// A random int8 KV cache in VRAM for the tests of full attention (the layout of mega_op_fattn.h and of the QSA
// attention) and the kernel that fills it. The launch machinery is in mega_rig.h.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "mega_rig.h"

namespace rig {

__device__ inline uint32_t mix(uint32_t x) {
    x ^= x >> 16, x *= 0x7feb352dU, x ^= x >> 15, x *= 0x846ca68bU, x ^= x >> 16;
    return x;
}

// Random int8 payload and scales in [0.002, 0.05) (the range of the real cache); row: bytes per position, the f32
// scales start at `scales` bytes into the row
__global__ void k_fill_kv(uint8_t* kv, int64_t rows, int row, int scales, uint32_t seed) {
    const int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * row) return;
    const int col = (int)(i % row);
    const uint32_t h = mix((uint32_t)i * 2654435761U ^ seed ^ (uint32_t)((i / row) >> 20));
    if (col < scales)
        kv[i] = (uint8_t)(int8_t)((int)(h % 255) - 127);
    else if ((col - scales) % 4 == 0)
        reinterpret_cast<float*>(kv + i)[0] = 0.002f + 0.048f * (float)(h & 0xffff) / 65536.0f;
}

// Cache of `rows` positions in VRAM filled with random values
class RandomCache {
public:
    RandomCache(int64_t rows, int nkv, int hd, uint32_t seed)
        : row_(2 * nkv * hd + nkv * hd / 16 * 4), rows_(rows), buf_((size_t)rows * row_) {
        CHECK(hipMemset(buf_.get(), 0, (size_t)rows * row_));
        const int64_t total = rows * row_;
        k_fill_kv<<<(unsigned)((total + 255) / 256), 256>>>(buf_.get<uint8_t>(), rows, row_, 2 * nkv * hd, seed);
        CHECK(hipDeviceSynchronize());
    }
    uint8_t* get() const { return buf_.get<uint8_t>(); }
    int row() const { return row_; }
    int64_t rows() const { return rows_; }

private:
    int row_;
    int64_t rows_;
    DeviceBuffer buf_;
};

}  // namespace rig
