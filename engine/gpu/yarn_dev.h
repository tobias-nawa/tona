// YaRN table of the RoPE of the QSA layers in constant memory (backend/rope_yarn.h). Constant memory is per translation
// unit: every .hip file that rotates (the kernel unit of the architecture, prefill_qsa.hip, ops.hip) uploads the table
// itself. Constant memory, not a kernel argument: the table is 264 bytes and the rotation sits in ~12 call sites of ops
// that share the kernel arguments of the megakernel program.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <stdexcept>
#include <string>

#include "backend/rope_yarn.h"

namespace tona::gpu {
namespace {

// Initialised native (all 1.0): a translation unit that never gets the table (tests and tools calling kernels directly)
// must not rotate by zero
static_assert(rope::YARN_PAIRS == 32, "initialiser below");
__constant__ double d_yarn_mul[rope::YARN_PAIRS] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
                                                    1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
__constant__ float d_yarn_mscale = 1.0f;

// Only at start-up, before the first kernel of the process runs: hipMemcpyToSymbol blocks, and a blocking HIP call
// while k_mega waits on the CPU would hang the GPU (AGENTS.md, GPU safety).
[[maybe_unused]] void upload_yarn(const rope::YarnTable& t) {
    if (hipMemcpyToSymbol(HIP_SYMBOL(d_yarn_mul), t.mul, sizeof t.mul) != hipSuccess ||
        hipMemcpyToSymbol(HIP_SYMBOL(d_yarn_mscale), &t.mscale, sizeof t.mscale) != hipSuccess)
        throw std::runtime_error("upload of the YaRN table failed");
}

// Reads the table of this translation unit back and aborts if it differs from `want`: three copies exist, and one that
// was never uploaded must not silently rotate natively.
[[maybe_unused]] void check_yarn(const rope::YarnTable& want, const char* unit) {
    double mul[rope::YARN_PAIRS];
    float mscale = 0;
    if (hipMemcpyFromSymbol(mul, HIP_SYMBOL(d_yarn_mul), sizeof mul) != hipSuccess ||
        hipMemcpyFromSymbol(&mscale, HIP_SYMBOL(d_yarn_mscale), sizeof mscale) != hipSuccess)
        throw std::runtime_error("read-back of the YaRN table failed");
    bool same = mscale == want.mscale;
    for (int i = 0; i < rope::YARN_PAIRS; i++) same = same && mul[i] == want.mul[i];
    if (!same) throw std::runtime_error(std::string("the YaRN table of ") + unit + " differs from TONA_ROPE_YARN");
}

// cos and sin of pair i of a NeoX RoPE over n_rot dimensions at position pos; angle in double. Native: mul = 1.0 and
// mscale = 1.0f, both exact, so the values equal the ones before the YaRN switch.
__device__ __forceinline__ void rope_cs(int i, int n_rot, int64_t pos, float base, float& c, float& s) {
    const double theta = (double)pos * (pow((double)base, -2.0 * i / n_rot) * d_yarn_mul[i]);
    c = (float)cos(theta) * d_yarn_mscale;
    s = (float)sin(theta) * d_yarn_mscale;
}

}  // namespace
}  // namespace tona::gpu
