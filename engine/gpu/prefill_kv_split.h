// K and V of a chunk of tokens into a KV cache split by position (kv_place.h). The write kernels address a flat cache
// (`kv + pos * row`); the tokens before the split go to the VRAM part, the others to the pinned part through a base
// pointer chosen so that the same arithmetic lands in it. Two launches of the unchanged kernels, no new device code:
// the Coder's kernel (k_qsa_prep) stays as it is.
#pragma once

#include <hip/hip_runtime.h>

#include <algorithm>
#include <cstdint>

#include "gpu/prefill_kernels.h"
#include "gpu/prefill_kvstore.h"

namespace tona::gpu {

// The pinned part of one layer's cache from the cell hot on; no cold: the cache is flat
struct KvCut {
    uint8_t* cold = nullptr;
    int64_t hot = INT64_MAX;
};

// Tokens of [pos0, pos0 + n) that are before the split
inline int kv_tokens_before(const KvCut& c, int64_t pos0, int n) {
    return c.cold ? (int)std::clamp<int64_t>(c.hot - pos0, 0, n) : n;
}

inline void launch_qsa_prep_cut(QsaPrepArgs a, const KvCut& c, hipStream_t st) {
    const int n1 = kv_tokens_before(c, a.pos0, a.n);
    if (n1 == a.n) return launch_qsa_prep(a, st);
    QsaPrepArgs rest = a;
    rest.kv = c.cold - c.hot * a.row;
    rest.n = a.n - n1, rest.pos0 = a.pos0 + n1;
    rest.qfull += (int64_t)n1 * 2 * a.nh * a.hd, rest.qn += (int64_t)n1 * a.nh * a.hd;
    rest.g += (int64_t)n1 * a.nh * a.hd;
    rest.tk += (int64_t)n1 * a.nkv * a.hd, rest.tv += (int64_t)n1 * a.nkv * a.hd;
    a.n = n1;
    if (n1 > 0) launch_qsa_prep(a, st);
    launch_qsa_prep(rest, st);
}

// Bolt's K and V store (head size 128, tk and tv [token][nkv][128])
inline void launch_kv_store_cut(KvStoreArgs a, const KvCut& c, hipStream_t st) {
    const int n1 = kv_tokens_before(c, a.pos0, a.n);
    if (n1 == a.n) return launch_kv_store(a, st);
    KvStoreArgs rest = a;
    rest.kv = c.cold - c.hot * a.row;
    rest.n = a.n - n1, rest.pos0 = a.pos0 + n1;
    rest.tk += (int64_t)n1 * a.nkv * 128, rest.tv += (int64_t)n1 * a.nkv * 128;
    a.n = n1;
    if (n1 > 0) launch_kv_store(a, st);
    launch_kv_store(rest, st);
}

}  // namespace tona::gpu
