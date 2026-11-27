#!/usr/bin/env python3
"""Reference values of the two routing policies for engine/tests/test_route_policy.cpp and test_gpu_route_sig.hip.

Written from the definitions of the policies in float64 and independent of engine/backend/route_policy.h:
  softmax:  p = softmax(r); the K largest p; weight = p / sum of the chosen p
  sigmoid:  s = sigmoid(r); the K largest s + bias; weight = s / max(sum of the chosen s, 2^-14) * scale
            (the rule of DeepSeek-V3 and Nemotron-H with `exp_probs_b`, one group; the floor on the sum is llama.cpp's
            clamp of the weight sum)
A tie goes to the smaller expert index. The inputs are not stored: logits and bias come from the generator below, which
the C++ tests implement the same way (xorshift64*, 24 bits per value, scaled by a power of two, so every value is the
same float in both languages).

Usage: python3 tools/gen_route_ref.py > engine/tests/data/route_ref.txt
Format per case: `case name E K gate scale bias levels seed`, then the chosen experts in rank order and their weights.
"""

import itertools
import math

MASK = (1 << 64) - 1
MIN_SUM = 6.103515625e-5  # 2^-14


def stream(seed):
    """xorshift64* seeded with seed + 1; yields the upper 24 bits of each output."""
    x = (seed + 1) * 0x9E3779B97F4A7C15 & MASK
    while True:
        x ^= x >> 12
        x ^= (x << 25) & MASK
        x ^= x >> 27
        yield ((x * 0x2545F4914F6CDD1D) & MASK) >> 40


def logits(seed, n, levels):
    """Uniform in [-4, 4) in steps of 2^-21; with levels > 0 only that many distinct values (exact ties)."""
    g = stream(seed)
    if levels:
        return [(next(g) % levels) * 0.5 - 1.0 for _ in range(n)]
    return [next(g) / 2**21 - 4.0 for _ in range(n)]


def bias_of(seed, n):
    """Uniform in [-0.5, 0.5) in steps of 2^-24."""
    g = stream(seed + 1000)
    return [next(g) / 2**24 - 0.5 for _ in range(n)]


def rank(keys, k):
    """The k best by key, descending; a tie goes to the smaller index."""
    return sorted(range(len(keys)), key=lambda e: (-keys[e], e))[:k]


def clear_of_ties(keys, order):
    """True if the ranking does not depend on rounding: adjacent chosen keys and the first one left out differ by
    more than float error unless they are exactly equal."""
    ranked = [keys[e] for e in order] + [sorted(keys, reverse=True)[len(order)]]
    return all(a == b or a - b > 1e-5 * max(abs(a), 1e-3) for a, b in itertools.pairwise(ranked))


def softmax_case(r, k):
    mx = max(r)
    ex = [math.exp(v - mx) for v in r]
    total = sum(ex)
    p = [v / total for v in ex]
    order = rank(p, k)
    chosen = sum(p[e] for e in order)
    return order, [p[e] / chosen for e in order], clear_of_ties(p, order)


def sigmoid_case(r, bias, k, scale):
    s = [1.0 / (1.0 + math.exp(-v)) for v in r]
    keys = [a + b for a, b in zip(s, bias, strict=True)]
    order = rank(keys, k)
    chosen = max(sum(s[e] for e in order), MIN_SUM)
    return order, [s[e] / chosen * scale for e in order], clear_of_ties(keys, order)


# name, E, K, gate, scale, bias, levels
CASES = [
    ("softmax-64", 64, 8, "softmax", 1.0, 0, 0),
    ("softmax-256", 256, 8, "softmax", 1.0, 0, 0),
    ("softmax-496", 496, 9, "softmax", 1.0, 0, 0),
    ("softmax-512", 512, 16, "softmax", 1.0, 0, 0),
    ("softmax-ties-256", 256, 8, "softmax", 1.0, 0, 3),
    ("softmax-all-equal", 64, 8, "softmax", 1.0, 0, 1),
    ("sigmoid-64", 64, 8, "sigmoid", 5.0, 1, 0),
    ("sigmoid-240", 240, 9, "sigmoid", 5.0, 1, 0),
    ("sigmoid-496", 496, 9, "sigmoid", 5.0, 1, 0),
    ("sigmoid-512", 512, 16, "sigmoid", 5.0, 1, 0),
    ("sigmoid-nobias-256", 256, 8, "sigmoid", 1.0, 0, 0),
    ("sigmoid-scale-2.5", 128, 6, "sigmoid", 2.5, 1, 0),
    ("sigmoid-ties-256", 256, 8, "sigmoid", 5.0, 0, 3),
    ("sigmoid-all-equal", 64, 8, "sigmoid", 5.0, 0, 1),
]
TOKENS = 4  # cases per row: seeds seed0 .. seed0 + TOKENS - 1


def main():
    for i, (name, n, k, gate, scale, with_bias, levels) in enumerate(CASES):
        for t in range(TOKENS):
            seed = 16 * i + t
            while True:
                r = logits(seed, n, levels)
                bias = bias_of(seed, n) if with_bias else [0.0] * n
                if gate == "softmax":
                    order, w, ok = softmax_case(r, k)
                else:
                    order, w, ok = sigmoid_case(r, bias, k, scale)
                if ok:
                    break
                seed += 1000  # a ranking that rounding could change: another input
            print(f"case {name} {n} {k} {gate} {scale!r} {with_bias} {levels} {seed}")
            print("experts", *order)
            print("weights", *(repr(v) for v in w))


if __name__ == "__main__":
    main()
