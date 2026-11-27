// K and V of n tokens (positions pos0 ...) into the int8 KV cache, head size 128, no norm and no RoPE. tk, tv:
// [token][nkv][128]; the row layout is the decode step's (mega_op_fattn.h): int8 K and V per KV head, then the scales.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tona::gpu {

struct KvStoreArgs {
    const float *tk, *tv;
    uint8_t* kv;
    int64_t row, pos0;
    int n, nkv;
};
void launch_kv_store(const KvStoreArgs& a, hipStream_t st);

}  // namespace tona::gpu
