"""Hit rate of a cache policy on an TONA_ROUTE_TRACE_P trace, as in research/measurements/2026-10-25-cache-policy.md:
hits over the unions of the experts per layer and pass, counted from `warm` of the run."""

from classlru import ClassLruCache
from policies import LrfuCache, LruCache

N_PRELOAD = 8  # TONA_PREFETCH default


def make_factory(spec: str, slots: int):
    """Spec: lru | lru-admit:HL | lrfu:H | class:N:HL | class-admit:N:HL (HL in tokens, H in passes)."""
    kind, *args = spec.split(":")
    nums = [float(a) for a in args]
    makers = {
        "lru": lambda: LruCache(slots),
        "lru-admit": lambda: LruCache(slots, nums[0]),
        "lrfu": lambda: LrfuCache(slots, nums[0]),
        "class": lambda: ClassLruCache(slots, int(nums[0]), nums[1]),
        "class-admit": lambda: ClassLruCache(slots, int(nums[0]), nums[1], admit=True),
    }
    if kind not in makers:
        raise ValueError(f"unknown policy {spec!r}")
    return makers[kind]


FILL_CAP = 4  # fill_cap_ of the server (mega_build_bufs.cpp): misses that enter per layer and pass while slots are free


def hit_rate(passes, make_cache, warm: float = 0.2, n_pre: int = N_PRELOAD, server_fill: bool = False) -> float:
    """server_fill: the refill rule of the server with TONA_FILL=0 (default): a missed expert enters the cache only
    while slots are free, at most FILL_CAP per layer and pass (ignoring that it sits in its slot one pass later).
    Without it every missed expert enters, as in the first simulation."""
    n_layer = 1 + max(lp.layer for ps in passes for lp in ps)
    caches = [make_cache() for _ in range(n_layer)]
    first = int(len(passes) * warm)
    hits = total = 0
    for t, ps in enumerate(passes):
        for lp in ps:
            cache = caches[lp.layer]
            if t >= first:
                total += len(lp.union)
                hits += sum(e in cache for e in lp.union)
            room = min(FILL_CAP, max(cache.cap - len(cache), 0)) if server_fill else None
            cache.demand(lp.union, lp.ntok, room)
            if lp.layer + 1 < n_layer:
                caches[lp.layer + 1].prefetch(lp.cand, n_pre)
    return hits / max(total, 1)
