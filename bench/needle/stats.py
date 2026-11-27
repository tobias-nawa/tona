"""Intervals, the paired exact test and the sample size of the needle test (standard library only)."""

import math
from statistics import NormalDist


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval of k hits in n trials."""
    if n == 0:
        return 0.0, 1.0
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
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def paired_diff(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
    """Difference (second minus first) with a Wald interval from the discordant pairs: b = first right and second
    wrong, c = the opposite, n = all pairs. Var(d) = (b + c - (b - c)^2 / n) / n^2."""
    if n == 0:
        return 0.0, 0.0, 0.0
    d = (c - b) / n
    half = z * math.sqrt(max(0.0, b + c - (b - c) ** 2 / n)) / n
    return d, d - half, d + half


def pairs_needed(psi: float, d: float, alpha: float = 0.05, power: float = 0.8) -> int:
    """Pairs for McNemar to find a difference d when a fraction psi of the pairs is discordant (Connor 1987):
    n = (z_a sqrt(psi) + z_b sqrt(psi - d^2))^2 / d^2."""
    nd = NormalDist()
    za, zb = nd.inv_cdf(1 - alpha / 2), nd.inv_cdf(power)
    return math.ceil((za * math.sqrt(psi) + zb * math.sqrt(max(0.0, psi - d * d))) ** 2 / (d * d))


def detectable(psi: float, n: int, alpha: float = 0.05, power: float = 0.8) -> float:
    """Smallest difference that n pairs resolve at the given discordance (inverse of pairs_needed, by bisection)."""
    lo, hi = 1e-4, min(1.0, psi)
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if pairs_needed(psi, mid, alpha, power) > n else (lo, mid)
    return hi
