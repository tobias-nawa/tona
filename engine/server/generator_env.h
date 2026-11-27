// Settings read once from the environment, shared by the generator's translation units
#pragma once

#include <algorithm>
#include <cstdio>
#include <cstdlib>

#include "backend/budget.h"
#include "backend/env.h"

namespace tona::server {

// Drafts in a small drafter block (4 tokens with last); a large block (8) costs 8.8 instead of 5.3 ms
constexpr int N_SMALL = 3;

// Up to 7 drafts (passes up to 8 tokens, from 6 through k_wide). TONA_DETERMINISTIC=1: up to 3, all passes bit-exact
inline int n_draft_max() {
    static const int n =
        backend::deterministic() ? 3 : (int)std::clamp<int64_t>(tona::env::integer("TONA_DRAFT_N", 7), 1, 7);
    return n;
}

// Widths from 6 round differently (k_wide): the choice of width may depend only on this request, otherwise the same
// request would get another answer after other requests. Calibration, probe counter and block size per request start
// from the same state (TONA_DRAFT_CALIB_KEEP=1: keep learning across requests, not repeatable)
inline bool calib_keep() {
    static const bool keep = tona::env::flag("TONA_DRAFT_CALIB_KEEP");
    return keep;
}

// TONA_DRAFT_LOG=<file>: measurement log of the drafts (per request "R", per pass width, accepted, p1 p2 p3, pass ms)
inline FILE* draft_log() {
    static FILE* f = tona::env::text("TONA_DRAFT_LOG") ? std::fopen(tona::env::text("TONA_DRAFT_LOG"), "a") : nullptr;
    return f;
}

}  // namespace tona::server
