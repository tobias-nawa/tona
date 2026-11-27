// TONA_IDX_SHARE=<list of full QSA layers> (IndexCache, arXiv 2603.12201): the listed QSA layers (index among the QSA
// layers, 0 = first) run their own indexer; every other QSA layer reuses the block selection of the nearest full
// layer before it. Unset or empty: every second QSA layer (0, 2, 4, ...); "all": every layer, bit-identical to the
// engine before IndexCache.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "backend/env.h"

namespace tona::env {

// One flag per QSA layer: true = runs its indexer. Layer 0 is always full (nothing before it to borrow from).
// An index out of range or unreadable aborts at start-up: a typo must not silently measure the default.
inline std::vector<bool> idx_full_layers(int n_qsa) {
    std::vector<bool> full(n_qsa, true);
    const char* e = text("TONA_IDX_SHARE");
    if (e && std::strcmp(e, "all") == 0) return full;
    full.assign(n_qsa, false);
    if (!e) {
        // Every second layer: 6 of 12 lost no task accuracy and passed the Δ-NLL rule; decode pass at 250K -4.5 %
        // (research/measurements/2026-10-30-threshold-main.md, 2026-10-30-indexcache-main.md).
        for (int l = 0; l < n_qsa; l += 2) full[l] = true;
        return full;
    }
    full[0] = true;
    for (const char* p = e; *p;) {
        char* end = nullptr;
        const long v = std::strtol(p, &end, 10);
        if (end == p || v < 0 || v >= n_qsa || (*end && *end != ',')) {
            std::fprintf(stderr, "TONA_IDX_SHARE: \"%s\" is not a list of QSA layer indices 0..%d\n", e, n_qsa - 1);
            std::exit(2);
        }
        full[v] = true;
        p = *end ? end + 1 : end;
    }
    return full;
}

}  // namespace tona::env
