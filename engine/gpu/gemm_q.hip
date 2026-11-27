// GEMM with the weights in GGUF format, formats of the Coder and Ornith (kernels: gpu/gemm_q_dev.h).
#include "gpu/gemm_q_bolt.h"
#include "gpu/gemm_q_dev.h"

namespace tona::gpu {

namespace {

constexpr QFormat FORMATS[] = {
    {0, launch_gemm_q<0>},   {1, launch_gemm_q<1>},       {30, launch_gemm_q<30>}, {8, launch_gemm_q<8>},
    {20, launch_gemm_q<20>}, {42, launch_gemm_q<42>},     {12, launch_gemm_q<12>}, {13, launch_gemm_q<13>},
    {14, launch_gemm_q<14>}, {22, launch_gemm_q<22>},     {18, launch_gemm_q<18>}, {21, launch_gemm_q<21>},
    {23, launch_gemm_q<23>}, {1019, launch_gemm_q<1019>},
};

}  // namespace

void gemm_q(uint32_t type, const GemmQSeg* segs, int nseg, hipStream_t stream) {
    upload_tables();
    if (gemm_q_bolt_type(type)) return gemm_q_bolt(type, segs, nseg, stream);
    if (!run_gemm_q(FORMATS, std::size(FORMATS), type, segs, nseg, stream))
        throw std::runtime_error(std::format("gemm_q: no kernel for type {}", type));
}

}  // namespace tona::gpu
