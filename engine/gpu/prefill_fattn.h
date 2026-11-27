// Prefill attention over the whole prefix (Ornith, Bolt): causal, every cached position, no indexer. Host-visible
// arguments and launcher; the kernel is in prefill_fattn.hip.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace tona::gpu {

// Tokens pos0 .. pos0 + n_tokens - 1 attend to the int8 cache (row bytes per position, layout as the decode step in
// mega_op_fattn.h), which must already hold K and V of all of them. q: [token][nh][hd] after norm and RoPE; gate: same
// shape or null (no gate); o: f16 [token][nh][hd], the input of the output projection. hd: 128 or 256; the group nh /
// nkv: 1, 2, 4, 8 or 16 (the 16 rows of a tile are tokens times heads of one KV head).
struct FattnPrefillArgs {
    const float* q;
    const uint8_t* kv;
    int64_t row;
    const float* gate;
    _Float16* o;
    int64_t pos0;
    int n_tokens, nh, nkv, hd;
    float scale;
    // A cache split by position (kv_place.h): the cells from kv_hot on are in kv_cold (null: kv holds all of them)
    const uint8_t* kv_cold = nullptr;
    int64_t kv_hot = INT64_MAX;
};

// The kernels; all give the same bits. automatic: the wave kernel (fattn_wave.h), or the one TONA_FATTN_PF_OLD=1
// (multi_tile) and TONA_FATTN_PF_MT1=1 (single_tile) name, for a comparison
enum class FattnPrefillKernel { automatic, wave, multi_tile, single_tile };

// False: an hd or a group without a kernel
bool launch_fattn_prefill(const FattnPrefillArgs& a, hipStream_t st,
                          FattnPrefillKernel k = FattnPrefillKernel::automatic);

}  // namespace tona::gpu
