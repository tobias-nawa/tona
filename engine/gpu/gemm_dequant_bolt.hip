// Unpacking of NVFP4 and IQ2_XXS weights to f16 and E4M3 (Bolt); a unit of its own so that the machine code of the
// units of the other architectures does not move (kernels: gpu/gemm_dequant_dev.h). The unpackers of the other formats
// stay unused in this unit
#pragma clang diagnostic ignored "-Wunused-function"

#include "gpu/gemm_dequant_dev.h"
#include "gpu/gemm_q_bolt.h"

namespace tona::gpu {

namespace {

template <>
__device__ void deq_sub<16>(const uint8_t* row, int sub, float* o) {  // IQ2_XXS
    const uint8_t* b = row + (sub / 8) * 66;
    const uint8_t* q = b + 2 + 8 * (sub % 8);
    const uint32_t aux = (uint32_t)u16(q + 4) | ((uint32_t)u16(q + 6) << 16);
    const float d = h2f(b) * (0.5f + (aux >> 28)) * 0.25f;
    for (int l = 0; l < 4; l++) {
        const uint8_t sg = d_ksigns[(aux >> 7 * l) & 127];
        const uint8_t* grid = reinterpret_cast<const uint8_t*>(&d_iq2xxs_grid.v[q[l]]);
        for (int j = 0; j < 8; j++) o[8 * l + j] = d * grid[j] * ((sg >> j) & 1 ? -1.0f : 1.0f);
    }
}
template <>
__device__ void deq_sub<40>(const uint8_t* row, int sub, float* o) {  // NVFP4, 64-value blocks
    constexpr int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    const uint8_t* b = row + (sub / 2) * 36;
    const int h = sub & 1;
    for (int i = 0; i < 2; i++) {
        const float d = ue4m3_half(b[2 * h + i]);
        const uint8_t* q = b + 4 + 16 * h + 8 * i;
        for (int j = 0; j < 8; j++) o[16 * i + j] = kv[q[j] & 0xf] * d, o[16 * i + j + 8] = kv[q[j] >> 4] * d;
    }
}
constexpr DequantFormat FORMATS[] = {
    {16, launch_dequant<16>, launch_dequant_fp8<16>},
    {40, launch_dequant<40>, launch_dequant_fp8<40>},
};

}  // namespace

void dequant_bolt_multi(const DequantShape& d, const void* const* w, _Float16* const* out, int n, hipStream_t stream) {
    upload_tables();  // the tables of this unit, not of gemm_dequant.hip
    if (!dequant_multi_in(FORMATS, d, w, out, n, stream))
        throw std::runtime_error(std::format("dequant_f16: no kernel for type {}", d.type));
}

void dequant_bolt_fp8(const DequantShape& d, const void* w, uint8_t* out, float* scale, hipStream_t stream) {
    upload_tables();
    if (!dequant_fp8_in(FORMATS, d, w, out, scale, stream))
        throw std::runtime_error(std::format("dequant_fp8: no kernel for type {}", d.type));
}

}  // namespace tona::gpu
