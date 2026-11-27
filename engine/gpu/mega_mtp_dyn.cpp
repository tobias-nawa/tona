// Host side of the dynamic rows of the MTP draft head at run time (see mega_mtp_dyn.h).
#include "gpu/mega_mtp_dyn.h"

#include <algorithm>

#include "gpu/mega_util.h"

namespace tona::gpu {

void MtpDyn::flush() {
    if (!on || pending.empty()) return;
    // 64: a copy of one byte costs ~10 us, the whole flag array (250 KB) about the same as six of them
    if (pending.size() > 64) {
        HIPCHECK(hipMemcpyAsync(excl_dev, excl_pin, excl_n, hipMemcpyHostToDevice, nullptr));
    } else {
        for (int32_t p : pending)
            HIPCHECK(hipMemcpyAsync(excl_dev + p, excl_pin + p, 1, hipMemcpyHostToDevice, nullptr));
    }
    pending.clear();
}

void MtpDyn::queue_tokens(const int32_t* map_dev, bool sampling) const {
    if (sampling && on && tok_pin)
        HIPCHECK(hipMemcpyAsync(tok_pin, map_dev + first, rows * sizeof(int32_t), hipMemcpyDeviceToHost, nullptr));
}

void MtpDyn::adopt_tokens(std::vector<int32_t>& vocab, bool sampling) const {
    if (sampling && on && tok_pin) std::copy(tok_pin, tok_pin + rows, vocab.begin() + first);
}

}  // namespace tona::gpu
