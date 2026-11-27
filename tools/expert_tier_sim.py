#!/usr/bin/env python3
"""Offline replay of a TONA_ROUTE_TRACE file over the expert tiers (docs/architecture/expert-tiers.md).

The trace holds, per layer and pass, the union of the experts the pass needs (engine/gpu/mega_select.cpp). The replay
runs a per-layer LRU as the VRAM slot cache, then sends every VRAM miss to the warm tier (pinned RAM, chosen by a
ranking) or to the cold tier (an LRU over the page cache), and counts what reaches the SSD. One RAM room R is split
between pins and page cache; the table shows which split leaves the fewest SSD reads.

Usage: tools/expert_tier_sim.py TRACE PROFILE --tokens N [--slots 84] [--room 6.6 22.6] [--q 0.2]
"""

import argparse
import heapq
import math
import struct
from collections import OrderedDict

EXPERT_MB = 2.05  # one expert of one layer: gate + up + down, Coder IQ1_M (docs/architecture/expert-tiers.md)
N_LAYER, N_EXPERT = 48, 256


def read_profile(path: str) -> list[list[int]]:
    with open(path, "rb") as f:
        magic, layers, experts = struct.unpack("<III", f.read(12))
        if magic != 0x50584549:
            raise ValueError("only profile version 1 (uniform layers) is read here")
        flat = struct.unpack(f"<{layers * experts}I", f.read(4 * layers * experts))
    return [list(flat[ly * experts : (ly + 1) * experts]) for ly in range(layers)]


def read_trace(path: str) -> list[list[tuple[int, ...]]]:
    """Passes, each a list of N_LAYER unions (a new pass starts at layer 0)."""
    passes: list[list[tuple[int, ...]]] = []
    with open(path, "rb") as f:
        buf = f.read()
    i = 0
    while i + 2 <= len(buf):
        layer, n = buf[i], buf[i + 1]
        union = tuple(buf[i + 2 : i + 2 + n])
        i += 2 + n
        if layer == 0 or not passes:
            passes.append([])
        passes[-1].append(union)
    return [p for p in passes if len(p) == N_LAYER]


def vram_misses(passes, profile, slots: int, warm: float):
    """Per pass and layer the experts not in the VRAM cache (LRU, seeded with the profile's top experts, every miss
    enters). Returns the miss lists after the warm-up share and the VRAM hit rate."""
    caches = []
    for ly in range(N_LAYER):
        order = sorted(range(N_EXPERT), key=lambda e: -profile[ly][e])[:slots]
        c: OrderedDict[int, None] = OrderedDict()
        for e in reversed(order):  # the most used is the most recent
            c[e] = None
        caches.append(c)
    first = int(len(passes) * warm)
    out, hits, total = [], 0, 0
    for t, ps in enumerate(passes):
        row = []
        for ly, union in enumerate(ps):
            c = caches[ly]
            miss = [e for e in union if e not in c]
            if t >= first:
                total += len(union)
                hits += len(union) - len(miss)
                row.append(miss)
            for e in union:
                if e in c:
                    c.move_to_end(e)
            for e in miss:
                c[e] = None
                if len(c) > slots:
                    c.popitem(last=False)
        if t >= first:
            out.append(row)
    return out, hits / max(total, 1)


def ranking(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    return sorted(scores, key=lambda k: -scores[k])


def profile_scores(profile, slots: int, q: float):
    s = {}
    for ly in range(N_LAYER):
        top = set(sorted(range(N_EXPERT), key=lambda e: -profile[ly][e])[:slots])
        for e in range(N_EXPERT):
            s[(ly, e)] = profile[ly][e] * (q if e in top else 1.0)
    return s


def miss_scores(miss, lo: int, hi: int):
    s = {(ly, e): 0.0 for ly in range(N_LAYER) for e in range(N_EXPERT)}
    for row in miss[lo:hi]:
        for ly, ms in enumerate(row):
            for e in ms:
                s[(ly, e)] += 1.0
    return s


def tiers(miss, order, pin_n: int, cache_n: int, lo: int = 0):
    """Counts over miss[lo:]: reads served by pins, by the page cache (LRU, cache_n entries) and by the SSD."""
    pinned = set(order[:pin_n])
    pc: OrderedDict[tuple[int, int], None] = OrderedDict()
    warm = ssd = cached = 0
    for row in miss[lo:]:
        for ly, ms in enumerate(row):
            for e in ms:
                k = (ly, e)
                if k in pinned:
                    warm += 1
                elif k in pc:
                    cached += 1
                    pc.move_to_end(k)
                else:
                    ssd += 1
                    pc[k] = None
                    if len(pc) > cache_n:
                        pc.popitem(last=False)
    return warm, cached, ssd


class Lrfu:
    """A cache that evicts the entry with the lowest decayed use count (CRF, half-life in passes). Keys compare by
    log2(count) + t / half_life, which orders entries the same at every later time: a heap with lazy deletion works."""

    def __init__(self, cap: int, half_life: float):
        self.cap, self.h = cap, half_life
        self.state: dict[tuple[int, int], tuple[float, float]] = {}  # key -> (count, last time)
        self.heap: list[tuple[float, tuple[int, int]]] = []

    def touch(self, k, t: float) -> bool:
        hit = k in self.state
        c, t0 = self.state.get(k, (0.0, t))
        c = c * 2 ** (-(t - t0) / self.h) + 1.0
        self.state[k] = (c, t)
        heapq.heappush(self.heap, (math.log2(c) + t / self.h, k))
        return hit

    def evict(self) -> None:
        while len(self.state) > self.cap and self.heap:
            key, k = heapq.heappop(self.heap)
            c, t0 = self.state.get(k, (0.0, 0.0))
            if k in self.state and abs(math.log2(c) + t0 / self.h - key) < 1e-9:
                del self.state[k]


def managed(miss, cap_n: int, half_life: float, lo: int = 0) -> int:
    """SSD reads of one managed RAM tier (all room as one cache, replaced by decayed use count)."""
    cache, ssd = Lrfu(cap_n, half_life), 0
    for t, row in enumerate(miss):
        for ly, ms in enumerate(row):
            for e in ms:
                if not cache.touch((ly, e), t) and t >= lo:
                    ssd += 1
        cache.evict()
    return ssd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("profile")
    ap.add_argument("--tokens", type=int, required=True, help="tokens decoded in the trace (for per-token rates)")
    ap.add_argument("--slots", type=int, default=84)
    ap.add_argument("--room", type=float, nargs="+", default=[6.6, 22.6], help="GB for pins plus page cache")
    ap.add_argument("--q", type=float, default=0.2)
    ap.add_argument("--warm", type=float, default=0.2)
    a = ap.parse_args()
    passes = read_trace(a.trace)
    profile = read_profile(a.profile)
    miss, hit = vram_misses(passes, profile, a.slots, a.warm)
    n_miss = sum(len(m) for row in miss for m in row)
    tok = a.tokens * len(miss) / len(passes)
    print(f"passes {len(passes)} (counted {len(miss)}), tokens counted {tok:.0f}, VRAM hit rate {hit:.3f}")
    print(f"VRAM misses per token {n_miss / tok:.1f} = {n_miss / tok * EXPERT_MB:.0f} MB")
    half = len(miss) // 2
    rankings = {
        "profile": ranking(profile_scores(profile, a.slots, 1.0)),
        f"profile q={a.q}": ranking(profile_scores(profile, a.slots, a.q)),
        "own misses (hindsight)": ranking(miss_scores(miss, 0, len(miss))),
    }
    first_half = ranking(miss_scores(miss, 0, half))
    for room in a.room:
        cap = int(room * 1000 / EXPERT_MB)
        res = [f"hl{hl}:{managed(miss, cap, hl) / tok:5.1f}" for hl in (20, 100, 500, 5000)]
        print(f"managed LFU-decay cache of {room} GB, SSD reads per token: " + "  ".join(res))
    for room in a.room:
        print(f"\nroom {room} GB: SSD reads per token (pin GB | page cache GB), lower is better")
        pins = [0.0] + [room * f for f in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)]
        for name, order in [*rankings.items(), ("online miss counts (1st half -> 2nd half)", first_half)]:
            lo = half if name.startswith("online") else 0
            line = []
            for p in pins:
                pin_n = int(p * 1000 / EXPERT_MB)
                cache_n = int((room - p) * 1000 / EXPERT_MB)
                _w, _c, s = tiers(miss, order, pin_n, cache_n, lo)
                n_tok = tok * (len(miss) - lo) / len(miss)
                line.append(f"{p:4.1f}:{s / n_tok:5.1f}")
            print(f"  {name:42s}" + " ".join(line))


if __name__ == "__main__":
    main()
