// K and V of a chunk into the int8 KV cache without norm or RoPE (Bolt's attention has neither): the counterpart of
// OP_FATTN_PREP's K and V units with all three flags off, with the same rounding, so a prompt cached by the prefill is
// the cache the decode passes would have written.
#include "gpu/prefill_dev.h"
#include "gpu/prefill_kvstore.h"

namespace tona::gpu {

namespace {

constexpr int NV = 4;  // values of a lane: head size 128

// One wave per (token, K or V head)
__global__ void k_kv_store(KvStoreArgs a) {
    const int64_t per = 2 * a.nkv;
    const int64_t u0 = ((int64_t)blockIdx.x * TPB + threadIdx.x) / WAVE;
    const int lane = threadIdx.x % WAVE;
    if (u0 >= a.n * per) return;
    const int64_t t = u0 / per;
    const int u = (int)(u0 % per);
    const bool is_v = u >= a.nkv;
    const int h = u % a.nkv;
    const float* src = (is_v ? a.tv : a.tk) + (t * a.nkv + h) * 128;
    float v[NV];
#pragma unroll
    for (int k = 0; k < NV; k++) v[k] = src[lane + 32 * k];
    uint8_t* r = a.kv + (a.pos0 + t) * a.row;
    int8_t* q = (int8_t*)r + (int64_t)h * 2 * 128 + (is_v ? 128 : 0);
    float* scale = (float*)(r + (int64_t)a.nkv * 2 * 128) + h * 2 * NV + (is_v ? NV : 0);
#pragma unroll
    for (int k = 0; k < NV; k++) {
        const float m = wave_max(fabsf(v[k]));
        const float sc = m > 0 ? m / 127.0f : 1.0f;
        q[lane + 32 * k] = (int8_t)__float2int_rn(v[k] / sc);
        if (lane == k) scale[k] = sc;
    }
}

}  // namespace

void launch_kv_store(const KvStoreArgs& a, hipStream_t st) {
    k_kv_store<<<blocks_for((int64_t)a.n * 2 * a.nkv * WAVE), TPB, 0, st>>>(a);
}

}  // namespace tona::gpu
