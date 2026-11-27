// Helpers shared by the translation units of gpu::Forward (forward.hip, forward_moe.hip).
#pragma once

#include <format>
#include <stdexcept>

#include "backend/env.h"
#include "gpu/matvec.h"

#define HIPCHECK(x)                                                                                           \
    do {                                                                                                      \
        hipError_t err_ = (x);                                                                                \
        if (err_ != hipSuccess) throw std::runtime_error(std::format("{}: {}", #x, hipGetErrorString(err_))); \
    } while (0)

namespace tona::gpu {

// Matvec through the int8 path if x is available as int8 and the format has the path, otherwise through float
inline void run_matvec(MatvecTask t, const Q8Vec* xq) {
    if (xq && matvec_q8_supported(t.type)) {
        t.xq = *xq;
        matvec_q8(t);
    } else {
        matvec(t);
    }
}

// TONA_FAKE_EXPERTS only for measuring: no reloading, slot e mod slots (wrong result!)
inline bool fake_experts() {
    static const bool fake = env::flag("TONA_FAKE_EXPERTS");
    return fake;
}

}  // namespace tona::gpu
