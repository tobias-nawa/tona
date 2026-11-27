// Kernel launches that must stay in gpu/mega_weight_kernels.hip (there is no -fgpu-rdc: a launch needs the kernel in
// the same translation unit). The host code in the other files calls these wrappers.
#pragma once

#include <cstdint>

namespace tona::gpu {

// Q8_0 -> Q2_0 on the GPU, one thread per block of 64 (two Q8_0 blocks); see k_q8_to_q2 in mega_weight_kernels.hip
struct Q8ToQ2 {
    const uint8_t* src;  // experts e0 ... of the matrix, `rows` rows of `in` values each
    int64_t nblk;        // blocks of 64 values to convert
    int in, rows, e0;
    uint8_t* dst;  // expert e at dst + e * ebytes, row i at i * rb
    int64_t ebytes, rb;
};
void launch_q8_to_q2(const Q8ToQ2& a);

// Rows r0 ... r0 + count of the truncated MTP head, copied from the full LM head in VRAM (token per row from map)
void launch_mtp_rows(const uint8_t* head, uint8_t* dst, const int32_t* map, int64_t r0, int row_bytes, unsigned count);

}  // namespace tona::gpu
