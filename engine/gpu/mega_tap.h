// TONA_TAP_OUT=file (Ornith): the residual stream of the decode pass after every mixer and after every layer, copied by
// plain OP_RES_NORM steps (no norm, no sum: a copy, so the pass's numbers do not change) into buffers that the host
// writes to a file after the pass. tools/tap_compare.py reads it next to the oracle's trace (TONA_REF_TRACE_OUT of
// tona-check). Off by default: without the variable the program has no extra step. Layout per pass: int32 {slots, d,
// nb, pos}, then slots buffers of nb * d floats; slot 2l is after the mixer of layer l, slot 2l + 1 after layer l.
#pragma once

#include <cstdint>
#include <cstdio>
#include <vector>

namespace tona::gpu {

struct MegaTap {
    std::vector<float*> bufs;
    int d = 0;
    std::FILE* fp = nullptr;
    MegaTap() = default;
    MegaTap(const MegaTap&) = delete;
    MegaTap& operator=(const MegaTap&) = delete;
    void dump(int nb, int64_t pos);  // after a pass of nb tokens at pos
    ~MegaTap() {
        if (fp) std::fclose(fp);
    }
};

}  // namespace tona::gpu
