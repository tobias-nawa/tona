// GEMM with the weights in NVFP4 and IQ2_XXS (Bolt); a unit of its own so that the machine code of the units of the
// other architectures does not move (kernels: gpu/gemm_q_dev.h).
#include "gpu/gemm_q_bolt.h"
// The unpackers of the other formats stay unused in this unit
#pragma clang diagnostic ignored "-Wunused-function"

#include "gpu/gemm_q_dev.h"

namespace tona::gpu {

namespace {

constexpr QFormat FORMATS[] = {{16, launch_gemm_q<16>}, {40, launch_gemm_q<40>}};

}  // namespace

bool gemm_q_bolt_type(uint32_t type) { return type == 16 || type == 40; }

void gemm_q_bolt(uint32_t type, const GemmQSeg* segs, int nseg, hipStream_t stream) {
    upload_tables();
    if (!run_gemm_q(FORMATS, std::size(FORMATS), type, segs, nseg, stream))
        throw std::runtime_error(std::format("gemm_q: no kernel for type {}", type));
}

}  // namespace tona::gpu
