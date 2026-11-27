#!/usr/bin/env python3
"""Reference values of the YaRN frequencies for engine/tests/test_rope_yarn.cpp.

The code follows transformers `_compute_yarn_parameters` (modeling_rope_utils.py) line by line, in float64 and without
torch, and is independent of engine/backend/rope_yarn.h, which writes the same thing as a multiplier per pair.

Usage: python3 tools/gen_yarn_ref.py > engine/tests/data/yarn_ref.txt
Format per factor: `factor attention_factor` and one line with the 32 inverse frequencies.
"""

import math

BASE = 1.0e7  # rope_theta of the model card
DIM = 64  # head_dim 256 * partial_rotary_factor 0.25
ORIGINAL = 262144  # original_max_position_embeddings
BETA_FAST, BETA_SLOW = 32, 1


def find_correction_dim(num_rotations, dim, base, max_position_embeddings):
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (2 * math.log(base))


def find_correction_range(low_rot, high_rot, dim, base, max_position_embeddings, truncate=True):
    low = find_correction_dim(low_rot, dim, base, max_position_embeddings)
    high = find_correction_dim(high_rot, dim, base, max_position_embeddings)
    if truncate:
        low = math.floor(low)
        high = math.ceil(high)
    return max(low, 0), min(high, dim - 1)


def linear_ramp_factor(low, high, n):
    if low == high:
        high += 0.001
    return [min(max((i - low) / (high - low), 0.0), 1.0) for i in range(n)]


def get_mscale(scale, mscale=1):
    return 1.0 if scale <= 1 else 0.1 * mscale * math.log(scale) + 1.0


def yarn_inv_freq(factor):
    pos_freqs = [BASE ** (i / DIM) for i in range(0, DIM, 2)]
    extrapolation = [1.0 / p for p in pos_freqs]
    interpolation = [1.0 / (factor * p) for p in pos_freqs]
    low, high = find_correction_range(BETA_FAST, BETA_SLOW, DIM, BASE, ORIGINAL)
    extrapolation_factor = [1 - r for r in linear_ramp_factor(low, high, DIM // 2)]
    return [
        interpolation[i] * (1 - extrapolation_factor[i]) + extrapolation[i] * extrapolation_factor[i]
        for i in range(DIM // 2)
    ]


def main():
    for factor in (1.5, 2.0, 4.0, 8.0):
        print(f"{factor!r} {get_mscale(factor)!r}")
        print(" ".join(repr(f) for f in yarn_inv_freq(factor)))


if __name__ == "__main__":
    main()
