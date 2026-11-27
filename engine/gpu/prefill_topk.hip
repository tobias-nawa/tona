// Block selection of the QSA prefill: the n_top best pool blocks of a token, cells ascending, then the tail. The result
// is bit-identical to op_topk of the megakernel (threshold = n_top-th largest score as an unsigned integer, ties go to
// the earlier block); the attention reads the cells in this order, so a different order would change its numbers.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"
#include "gpu/topk_dev.h"

namespace tona::gpu {

namespace {

// Radix selection of the n_top best pool blocks like op_topk (ties: the earlier ones), cells ascending, then the tail
__global__ void __launch_bounds__(TK) k_topk(TopkArgs a) {
    const auto& [score_f, cells_all, ncells, t0, pos0, blk, topk, sstride] = a;
    const int tt = blockIdx.x, t = t0 + tt;
    const int64_t pos = pos0 + t;
    if (!qsa_sel(pos, blk, topk)) return;
    int32_t* cells = cells_all + (int64_t)t * MAX_CELLS;
    const int n_pool = (int)((pos + 1) / blk);
    const int n_top = std::min(n_pool, topk / blk);
    __shared__ TopkLds L;
    const int th = threadIdx.x, w = th / WAVE;
    const int stripe = ((n_pool + NW - 1) / NW + WAVE - 1) / WAVE * WAVE;
    const int lo = std::min(w * stripe, n_pool);
    const TopkStripe s{(const uint32_t*)score_f + tt * sstride, lo, std::min(n_pool, lo + stripe)};
    RadixSel sel{0u, n_top};
    for (int pass = 0; pass < 3; pass++) sel = topk_radix_pass(s, L, sel, pass);
    topk_count(s, L, sel.thr);
    __syncthreads();
    topk_emit(s, sel.thr, topk_wave_start(L, sel.k), cells, blk);
    const int tail = (int)(pos + 1 - (int64_t)n_pool * blk);
    for (int k = th; k < tail; k += TK) cells[n_top * blk + k] = n_pool * blk + k;
    if (th == 0) ncells[t] = n_top * blk + tail;
}

}  // namespace

void launch_topk(const TopkArgs& a, int blocks, hipStream_t st) { k_topk<<<blocks, TK, 0, st>>>(a); }

}  // namespace tona::gpu
