// Host side of the per-layer tap (gpu/mega_tap.h): after a pass, the tapped residuals go to the file.
#include "gpu/mega_tap.h"

#include "gpu/mega_util.h"

namespace tona::gpu {

void MegaTap::dump(int nb, int64_t pos) {
    const int32_t hd[4] = {(int32_t)bufs.size(), d, nb, (int32_t)pos};
    std::fwrite(hd, sizeof hd, 1, fp);
    std::vector<float> h((size_t)nb * (size_t)d);
    for (const float* b : bufs) {
        HIPCHECK(hipMemcpy(h.data(), b, h.size() * sizeof(float), hipMemcpyDeviceToHost));
        std::fwrite(h.data(), sizeof(float), h.size(), fp);
    }
    std::fflush(fp);
}

}  // namespace tona::gpu
