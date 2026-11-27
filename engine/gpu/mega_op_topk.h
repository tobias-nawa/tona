// Step function of the QSA block selection (OP_TOPK). A header of its own so that tests/test_gpu_topk.hip can run it
// without the rest of the QSA steps. Part of the kernel translation unit (arch/<name>/kernel.hip, no -fgpu-rdc); do not
// include elsewhere.
#pragma once

#include "gpu/mega_op_common.h"
#include "gpu/topk_dev.h"

namespace tona::gpu {
namespace {

// Block b for token b: the n_top best pool blocks (scores >= 0, so the bit patterns are monotonic), the earlier ones on
// a tie. Cells in ascending order, then the tail from n_pool * blk. score: i[3] values per token; cells: MAX_CELLS per
// token. Three radix passes of 11, 11 and 10 bits with the histogram in sm (topk_dev.h), the same code as the prefill.
__device__ __noinline__ void op_topk(const MegaOp& op, const Ctx& c) {
    static_assert(THREADS == TK, "the selection runs one thread per histogram slice of 4 bins");
    static_assert(sizeof(TopkLds) <= SM_FLOATS * 4, "the histogram must fit in sm, which must not grow");
    const int blk = op.i[0], sstride = op.i[3];
    TopkLds& L = *reinterpret_cast<TopkLds*>(c.sm);
    for (int b = blockIdx.x; b < c.nb; b += gridDim.x) {
        const int64_t pos = c.pos + b;
        if (!qsa_select(op, pos)) continue;
        const uint32_t* sc = (const uint32_t*)op.p[0] + (int64_t)b * sstride;
        int32_t* cells = (int32_t*)op.p[1] + (int64_t)b * MAX_CELLS;
        const int n_pool = (int)((pos + 1) / blk);
        const int n_top = std::min(n_pool, op.i[1] / blk);
        topk_cells(topk_stripe(sc, n_pool), n_top, blk, cells, L);
        const int tail = (int)(pos + 1 - (int64_t)n_pool * blk);
        for (int j = threadIdx.x; j < tail; j += THREADS) cells[n_top * blk + j] = n_pool * blk + j;
        if (threadIdx.x == 0) c.P->n_cells[c.r0 + b] = n_top * blk + tail;
        __syncthreads();
    }
}

}  // namespace
}  // namespace tona::gpu
