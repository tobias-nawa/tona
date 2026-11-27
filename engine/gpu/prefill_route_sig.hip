// Prefill routing with the sigmoid policy (prefill_route_sig.h).
#include "gpu/prefill_route_sig.h"
#include "gpu/route_sig_dev.h"

namespace tona::gpu {

namespace {

constexpr int TPB = 256;

__global__ void __launch_bounds__(TPB) k_route_sig(RouteSigArgs a) {
    const int64_t t = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (t >= a.n) return;
    SigRow row;
    route_sig_row(row, a.logits + t * a.E, a.bias, a.E);
    const int bad = (int)wave_sum((float)row.bad);
    if (lane == 0 && bad > 0 && a.nan_word >= 0) atomicAdd(&a.cnt[a.nan_word], 1);
    float score[16];
    int expert[16];
    float sum = 0;  // in rank order from 0, as on the CPU
    for (int r = 0; r < a.ne; r++) {
        const SigBest b = route_sig_next(row);
        score[r] = b.score, expert[r] = b.expert;
        sum += b.score;
    }
    if (lane != 0) return;
    for (int r = 0; r < a.ne; r++) {
        a.sel[t * a.ne + r] = expert[r];
        a.sw[t * a.ne + r] = tona::backend::sigmoid_route::weight(score[r], sum, a.scale);
        atomicAdd(&a.cnt[expert[r]], 1);
    }
}

}  // namespace

void launch_route_sig(const RouteSigArgs& a, hipStream_t st) {
    k_route_sig<<<(unsigned)(((int64_t)a.n * WAVE + TPB - 1) / TPB), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
