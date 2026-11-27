// Unpacking of weights from the GGUF formats to f16, formats of the Coder and Ornith (kernels: gpu/gemm_dequant_dev.h).
#include "gpu/gemm_dequant_dev.h"
#include "gpu/gemm_q_bolt.h"

namespace tona::gpu {

namespace {

constexpr DequantFormat FORMATS[] = {
    {0, launch_dequant<0>, launch_dequant_fp8<0>},    {1, launch_dequant<1>, launch_dequant_fp8<1>},
    {30, launch_dequant<30>, launch_dequant_fp8<30>}, {8, launch_dequant<8>, launch_dequant_fp8<8>},
    {20, launch_dequant<20>, launch_dequant_fp8<20>}, {42, launch_dequant<42>, launch_dequant_fp8<42>},
    {12, launch_dequant<12>, launch_dequant_fp8<12>}, {13, launch_dequant<13>, launch_dequant_fp8<13>},
    {14, launch_dequant<14>, launch_dequant_fp8<14>}, {22, launch_dequant<22>, launch_dequant_fp8<22>},
    {18, launch_dequant<18>, launch_dequant_fp8<18>}, {21, launch_dequant<21>, launch_dequant_fp8<21>},
    {23, launch_dequant<23>, launch_dequant_fp8<23>}, {1019, launch_dequant<1019>, launch_dequant_fp8<1019>},
};

}  // namespace

bool dequant_f16_supported(uint32_t type) {
    if (gemm_q_bolt_type(type)) return true;
    for (const DequantFormat& f : FORMATS)
        if (f.type == type) return true;
    return false;
}

void dequant_f16_multi(const DequantShape& d, const void* const* w, _Float16* const* out, int n, hipStream_t stream) {
    upload_tables();
    if (d.in % 32 || d.ld_out % 8) throw std::runtime_error("dequant_f16: row length is not a multiple of 32");
    if (gemm_q_bolt_type(d.type)) return dequant_bolt_multi(d, w, out, n, stream);
    if (!dequant_multi_in(FORMATS, d, w, out, n, stream))
        throw std::runtime_error(std::format("dequant_f16: no kernel for type {}", d.type));
}

void dequant_fp8(const DequantShape& d, const void* w, uint8_t* out, float* scale, hipStream_t stream) {
    upload_tables();
    if (d.in % 64 || d.in > FP8_MAX_IN || d.ld_out != d.in)
        throw std::runtime_error(std::format("dequant_fp8: row length {} does not fit", d.in));
    if (gemm_q_bolt_type(d.type)) return dequant_bolt_fp8(d, w, out, scale, stream);
    if (!dequant_fp8_in(FORMATS, d, w, out, scale, stream))
        throw std::runtime_error(std::format("dequant_fp8: no kernel for type {}", d.type));
}

void dequant_f16(const DequantShape& d, const void* w, _Float16* out, hipStream_t stream) {
    dequant_f16_multi(d, &w, &out, 1, stream);
}

}  // namespace tona::gpu
