#!/usr/bin/env python3
"""Evaluation of an expert trace of tona-ref --experts: int16 [tokens][layers][active].

- union of the experts over b consecutive tokens (cost of a verification batch)
- skew of the usage (share of the most frequent experts among all calls)
- hit rate of an expert cache per layer (LRU and "ideal static" = the most frequent ones in hindsight)

Usage: tools/expert_stats.py TRACE.bin [--layers 48] [--used 10] [--experts 256] [--cache-frac 0.38]
"""

import argparse
import array
import sys
from collections import Counter, OrderedDict


def load_trace(a) -> list[list[set]]:
    data = array.array("h")
    with open(a.trace, "rb") as f:
        data.frombytes(f.read())
    per_tok = a.layers * a.used
    n = len(data) // per_tok
    return [
        [set(data[t * per_tok + ly * a.used : t * per_tok + (ly + 1) * a.used]) for ly in range(a.layers)]
        for t in range(n)
    ]


def print_unions(tr, a) -> None:
    n = len(tr)
    print("\nUnion over b consecutive tokens (experts per layer, mean; disjoint would be b*10):")
    for b in (1, 2, 3, 4, 6, 8, 12, 16):
        if b > n:
            break
        tot = cnt = 0
        for t0 in range(0, n - b + 1):
            for ly in range(a.layers):
                tot += len(set().union(*(tr[t][ly] for t in range(t0, t0 + b))))
                cnt += 1
        u = tot / cnt
        print(
            f"  b={b:2d}: {u:6.1f} experts = {u / a.used:.2f}x one token, {u / (b * a.used) * 100:5.1f} % of disjoint"
        )


def lru_hits(tr, layer: int, cap: int) -> int:
    lru = OrderedDict()
    hits = 0
    for t in range(len(tr)):
        for e in tr[t][layer]:
            if e in lru:
                hits += 1
                lru.move_to_end(e)
            else:
                lru[e] = True
                if len(lru) > cap:
                    lru.popitem(last=False)
    return hits


def print_cache(tr, a) -> None:
    n = len(tr)
    cap = round(a.cache_frac * a.experts)
    print(f"\nSkew and cache ({cap} of {a.experts} experts per layer in VRAM):")
    lru_total = static_hits = calls = 0
    for ly in range(a.layers):
        c = Counter(e for t in range(n) for e in tr[t][ly])
        calls += sum(c.values())
        lru_total += lru_hits(tr, ly, cap)
        static_hits += sum(v for _, v in c.most_common(cap))
    print(f"  ideal static (the {cap} most frequent in hindsight): {static_hits / calls * 100:.1f} % hits")
    print(f"  LRU (cold start): {lru_total / calls * 100:.1f} % hits")
    distinct = sum(len({e for t in range(n) for e in tr[t][ly]}) for ly in range(a.layers)) / a.layers
    print(f"  distinct experts used per layer on average: {distinct:.0f}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Evaluate an expert trace of tona-ref --experts.")
    ap.add_argument("trace")
    ap.add_argument("--layers", type=int, default=48)
    ap.add_argument("--used", type=int, default=10)
    ap.add_argument("--experts", type=int, default=256)
    ap.add_argument("--cache-frac", type=float, default=0.38)
    a = ap.parse_args()
    tr = load_trace(a)
    print(f"{len(tr)} tokens, {a.layers} layers, {a.used} active of {a.experts}")
    print_unions(tr, a)
    print_cache(tr, a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
