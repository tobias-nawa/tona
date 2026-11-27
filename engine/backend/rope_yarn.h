// YaRN for the RoPE of the QSA layers and the MTP layer (TONA_ROPE_YARN=off|<factor>, TONA_YARN_MSCALE=<value>).
//
// Formula: YaRN (arXiv 2309.00071) as in transformers `_compute_yarn_parameters` (modeling_rope_utils.py) and llama.cpp
// `rope_yarn` / `ggml_rope_yarn_corr_dims` (ggml-cpu/ops.cpp): pair i keeps its native frequency below the correction
// range [low, high], runs at 1/factor above it and blends linearly in between; cos and sin are scaled by mscale.
// beta_fast 32, beta_slow 1 and original_max_position_embeddings 262144 are the values of the Qwen model card.
//
// The result is a multiplier per pair, so that the angle is `pos * (base^(-2i/n_rot) * mul[i])`. With the YaRN off, mul
// is exactly 1.0 and mscale exactly 1.0f: both are exact in floating point and the angles equal the ones of the engine
// before the switch.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numbers>

#include "backend/env.h"

namespace tona::rope {

constexpr int YARN_PAIRS = 32;  // 64 rotated dimensions of the QSA layers

struct YarnTable {
    double mul[YARN_PAIRS];  // frequency of pair i relative to the native one
    float mscale;            // factor of cos and sin
    double factor;           // 1 = off
};

// Number of rotations of pair i inside the original context -> (fractional) index of the pair
inline double yarn_corr_dim(double n_rot, double orig_ctx, double rotations, double base) {
    return n_rot * std::log(orig_ctx / (rotations * 2.0 * std::numbers::pi)) / (2.0 * std::log(base));
}

// mscale = 0.1 ln(factor) + 1: the YaRN paper's fit, transformers `get_mscale`, llama.cpp's `mscale *= 1 + 0.1 ln(1/s)`
inline double yarn_default_mscale(double factor) { return factor <= 1.0 ? 1.0 : 0.1 * std::log(factor) + 1.0; }

constexpr double YARN_ORIG_CTX = 262144, YARN_BETA_FAST = 32, YARN_BETA_SLOW = 1;

inline YarnTable yarn_table(double factor, double mscale, double base, int n_rot = 2 * YARN_PAIRS) {
    YarnTable t{};
    t.factor = factor;
    t.mscale = (float)mscale;
    const int half = n_rot / 2;
    if (factor <= 1.0) {
        for (int i = 0; i < YARN_PAIRS; i++) t.mul[i] = 1.0;
        t.mscale = 1.0f;
        t.factor = 1.0;
        return t;
    }
    // truncate = true (default of transformers): floor and ceil of the correction range, clamped to the pairs
    const double low = std::max(std::floor(yarn_corr_dim(n_rot, YARN_ORIG_CTX, YARN_BETA_FAST, base)), 0.0);
    double high = std::min(std::ceil(yarn_corr_dim(n_rot, YARN_ORIG_CTX, YARN_BETA_SLOW, base)), (double)(n_rot - 1));
    if (low == high) high += 0.001;
    for (int i = 0; i < YARN_PAIRS; i++) {
        const double ramp = std::clamp((i - low) / (high - low), 0.0, 1.0);
        const double extrapolation = 1.0 - ramp;  // share of the native frequency
        t.mul[i] = i < half ? (1.0 / factor) * (1.0 - extrapolation) + extrapolation : 1.0;
    }
    return t;
}

// Unset, empty, "off" or "0": native. Otherwise a factor above 1; anything else aborts at start-up (a typo must not
// silently measure the native model). TONA_YARN_MSCALE overrides the YaRN fit of mscale.
inline YarnTable yarn_from_env(double base) {
    const char* e = env::text("TONA_ROPE_YARN");
    if (!e || std::strcmp(e, "off") == 0 || std::strcmp(e, "0") == 0) return yarn_table(1.0, 1.0, base);
    char* end = nullptr;
    const double f = std::strtod(e, &end);
    if (end == e || *end || !(f > 1.0) || f > 64.0) {
        std::fprintf(stderr, "TONA_ROPE_YARN: \"%s\" is neither off nor a factor in (1, 64]\n", e);
        std::exit(2);
    }
    const double ms = env::number("TONA_YARN_MSCALE", yarn_default_mscale(f));
    if (!(ms > 0.0)) {
        std::fprintf(stderr, "TONA_YARN_MSCALE must be positive\n");
        std::exit(2);
    }
    return yarn_table(f, ms, base);
}

// Read once: all backends of a process use the same table
inline const YarnTable& yarn_active(double base) {
    static const YarnTable t = yarn_from_env(base);
    return t;
}

struct CosSin {
    double c, s;
};

// cos and sin of pair i at position pos, scaled by mscale, in double: theta = pos * (base^(-2i/n_rot) * mul[i]). The
// one formula of the CPU reference and of every host-side RoPE table.
inline CosSin pair_cos_sin(const YarnTable& yarn, int64_t n_rot, int64_t pos, float base, int64_t i) {
    const double theta = (double)pos * (std::pow((double)base, -2.0 * i / n_rot) * yarn.mul[i]);
    return {std::cos(theta) * yarn.mscale, std::sin(theta) * yarn.mscale};
}

}  // namespace tona::rope
