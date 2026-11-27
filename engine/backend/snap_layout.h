// Size arithmetic of a stashed snapshot (HostSnap): what the indexer pool and the KV rows of a position take, the
// total, and the checks a snapshot must pass before it is loaded. Only sizes: the copies stay per platform.
#pragma once

#include <cstddef>
#include <cstdint>

#include "backend/backend.h"

namespace tona::backend {

// Pooled indexer blocks of pos rows: one per full block plus the one in progress
inline size_t snap_pool_rows(int64_t pos, uint32_t idx_block) { return (size_t)(pos / idx_block + 1); }

// Pool bytes of one QSA layer at pos rows (float keys of idx_dim)
inline size_t snap_pool_bytes(int64_t pos, uint32_t idx_block, uint32_t idx_dim) {
    return snap_pool_rows(pos, idx_block) * idx_dim * sizeof(float);
}

// Whole snapshot: the fixed parts (recurrent state, draft ring) plus KV and pool of every QSA layer
inline size_t snap_total_bytes(size_t fixed, size_t kv_total, size_t pool_total) {
    return fixed + kv_total + pool_total;
}

// Position inside the context and one token per position
inline bool snap_pos_ok(const HostSnap& s, int64_t max_ctx) {
    return s.pos >= 1 && s.pos <= max_ctx && s.tokens.size() == (size_t)s.pos;
}

// KV and pool sizes over all QSA layers
inline bool snap_kv_pool_ok(const HostSnap& s, size_t kv_total, size_t pool_total) {
    return s.kv.size() == kv_total && s.pool.size() == pool_total;
}

}  // namespace tona::backend
