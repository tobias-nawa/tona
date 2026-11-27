"""Paired comparison of two runs over the same tasks: McNemar exact test, Newcombe interval, power."""

import math


def wilson(k: int, n: int, z: float = 1.96) -> tuple:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact p of b versus c discordant pairs under p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def newcombe_paired(a: int, b: int, c: int, d: int, z: float = 1.96) -> tuple:
    """Difference of accuracies (variant minus reference) with Newcombe's score interval for paired proportions
    (method 10). a: both right, b: reference right and variant wrong, c: the opposite, d: both wrong."""
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n  # reference, variant
    l1, u1 = wilson(a + b, n, z)
    l2, u2 = wilson(a + c, n, z)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den else 0.0
    diff = p2 - p1
    lo = diff - math.sqrt(max(0.0, (p2 - l2) ** 2 - 2 * phi * (p2 - l2) * (u1 - p1) + (u1 - p1) ** 2))
    hi = diff + math.sqrt(max(0.0, (u2 - p2) ** 2 - 2 * phi * (u2 - p2) * (p1 - l1) + (p1 - l1) ** 2))
    return diff, lo, hi


def tasks_needed(f: float, d: float) -> int:
    """n for a detectable accuracy difference d (fraction) at 80 % power, alpha 0.05: d = 2.8 sqrt(f / n)."""
    return math.ceil(7.84 * f / (d * d))


def detectable(f: float, n: int) -> float:
    return 2.8 * math.sqrt(f / n)
