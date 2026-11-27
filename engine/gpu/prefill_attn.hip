// Prefill kernels of the attention over the KV cache (int8): a scalar form and a flash-attention form with WMMA.
#include "backend/env.h"
#include "gpu/attn_wmma1.h"
#include "gpu/attn_wmma2.h"
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kernels.h"

namespace tona::gpu {

namespace {

__global__ void __launch_bounds__(TPB) k_attn(AttnArgs a) {
    const auto& [q, kv, row, cells_all, ncells, gate, o, pos0, nh, nkv, hd, blk, topk, scale] = a;
    __shared__ float red[TPB / WAVE], qh[256], sc[MAX_CELLS];
    const int b = blockIdx.x / nh, h = blockIdx.x % nh, lane = threadIdx.x % WAVE, wv = threadIdx.x / WAVE;
    const int64_t pos = pos0 + b;
    const bool sel = qsa_sel(pos, blk, topk);
    const int32_t* cells = sel ? cells_all + (int64_t)b * MAX_CELLS : nullptr;
    const int n_cells = sel ? ncells[b] : (int)(pos + 1);
    const int kvh = h / (nh / nkv);
    for (int i = threadIdx.x; i < hd; i += TPB) qh[i] = q[((int64_t)b * nh + h) * hd + i];
    lds_sync();
    for (int ci = wv; ci < n_cells; ci += TPB / WAVE) {
        const int64_t cell = cells ? cells[ci] : ci;
        const uint8_t* r = kv + cell * row;
        const int8_t* kc = (const int8_t*)r + kvh * 2 * hd;
        const float* ks = (const float*)(r + (int64_t)nkv * 2 * hd) + kvh * 16;
        float dot = 0;
        for (int k = 0; k < hd / WAVE; k++) dot += qh[lane + WAVE * k] * (float)kc[lane + WAVE * k] * ks[k];
        dot = wave_sum(dot);
        if (lane == 0) sc[ci] = dot * scale;
    }
    lds_sync();
    float mx = -INFINITY;
    for (int ci = threadIdx.x; ci < n_cells; ci += TPB) mx = fmaxf(mx, sc[ci]);
    mx = block_max256(mx, red);
    float sum = 0;
    for (int ci = threadIdx.x; ci < n_cells; ci += TPB) {
        const float e = __expf(sc[ci] - mx);
        sc[ci] = e;
        sum += e;
    }
    sum = block_sum256(sum, red);
    lds_sync();
    const int i = threadIdx.x;
    if (i < hd) {
        float acc = 0;
        for (int ci = 0; ci < n_cells; ci++) {
            const int64_t cell = cells ? cells[ci] : ci;
            const uint8_t* r = kv + cell * row;
            acc += sc[ci] * (float)((const int8_t*)r)[kvh * 2 * hd + hd + i] *
                   ((const float*)(r + (int64_t)nkv * 2 * hd))[kvh * 16 + 8 + i / 32];
        }
        o[((int64_t)b * nh + h) * hd + i] = (_Float16)(acc / sum * sigm(gate[((int64_t)b * nh + h) * hd + i]));
    }
}

}  // namespace

void launch_attn(const AttnArgs& a, int n_tokens, hipStream_t st) { k_attn<<<n_tokens * a.nh, TPB, 0, st>>>(a); }
void launch_attn_wmma_gather(const AttnWmmaArgs& a, int n_tokens, hipStream_t st, int mode) {
    const unsigned blocks = n_tokens * a.nkv;
    switch (mode) {
        case 0:
            k_attn_wmma2<0><<<blocks, TPB, 0, st>>>(a);
            break;
        case 1:
            k_attn_wmma2<1><<<blocks, TPB, 0, st>>>(a);
            break;
        case 5:
            k_attn_wmma2<3><<<blocks, TPB, 0, st>>>(a);
            break;
        case 3:
            k_attn_wmma<false><<<blocks, TPB, 0, st>>>(a);
            break;
        case 4:
            k_attn_wmma<true><<<blocks, TPB, 0, st>>>(a);
            break;
        default:
            k_attn_wmma2<2><<<blocks, TPB, 0, st>>>(a);
            break;
    }
}

void launch_attn_wmma(const AttnWmmaArgs& a, int n_tokens, hipStream_t st) {
    static const bool old_kernel = env::flag("TONA_ATTN_WMMA_OLD");
    if (old_kernel) {
        if (a.kv_host)
            k_attn_wmma<false><<<n_tokens * a.nkv, TPB, 0, st>>>(a);
        else
            k_attn_wmma<true><<<n_tokens * a.nkv, TPB, 0, st>>>(a);
        return;
    }
    launch_attn_wmma_gather(a, n_tokens, st, a.kv_host ? 0 : 2);
}

}  // namespace tona::gpu
