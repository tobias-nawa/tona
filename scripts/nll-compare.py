#!/usr/bin/python3 -I
"""Paired comparison of runs of `tona-gpu-gen --multi-check` with TONA_MULTI_DUMP.

Usage: nll-compare.py REFERENCE.txt [VARIANT.txt ...]
Dump lines: conversation position nll_alone nll_together KL(alone||together) top1_equal.

Per conversation and file: -log p together versus alone (mean of the difference with a 95% interval), KL
divergence (mean, 99th percentile, with interval), top-1 agreement. Per variant additionally paired against the
reference: difference of -log p together on the same tokens. Intervals: block bootstrap (blocks of 64 tokens,
because neighboring tokens are correlated), 4,000 draws, fixed seed.
"""

import random
import statistics
import sys
from pathlib import Path

BLOCK, DRAWS = 64, 4000


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        g, pos, na, nm, kl, t1 = line.split()
        rows[(int(g), int(pos))] = (float(na), float(nm), float(kl), int(t1))
    return rows


def boot(values):
    """Mean and 95% interval by block bootstrap."""
    blocks = [values[i : i + BLOCK] for i in range(0, len(values), BLOCK)]
    rnd = random.Random(1)
    means = []
    for _ in range(DRAWS):
        pick = [blocks[rnd.randrange(len(blocks))] for _ in blocks]
        flat = [v for b in pick for v in b]
        means.append(sum(flat) / len(flat))
    means.sort()
    return sum(values) / len(values), means[int(0.025 * DRAWS)], means[int(0.975 * DRAWS)]


def fmt(m, lo, hi, scale=1.0, digits=4):
    return f"{m * scale:+.{digits}f} [{lo * scale:+.{digits}f}, {hi * scale:+.{digits}f}]"


def report_file(f, d, ref, keys):
    ks = [k for k in keys if k in d]
    dn = [d[k][1] - d[k][0] for k in ks]
    kl = [d[k][2] for k in ks]
    t1 = sum(d[k][3] for k in ks) / len(ks)
    kls = sorted(kl)
    print(
        f"  {f}: {len(ks)} tokens, together - alone {fmt(*boot(dn))}, KL {fmt(*boot(kl), digits=5)} "
        f"(p99 {kls[int(0.99 * len(kls))]:.4f}), top-1 equal {100 * t1:.2f} %"
    )
    if d is not ref:
        dv = [d[k][1] - ref[k][1] for k in ks if k in ref]
        dkl = boot([d[k][2] - ref[k][2] for k in ks if k in ref])
        print(f"      against reference (same tokens, together): {fmt(*boot(dv))}, dKL {fmt(*dkl, digits=5)}")


def main():
    files = sys.argv[1:]
    if not files:
        sys.exit(__doc__)
    data = [load(f) for f in files]
    ref = data[0]
    for g in sorted({k[0] for k in ref}):
        print(f"Conversation {'AB'[g]}:")
        keys = sorted(k for k in ref if k[0] == g)
        for f, d in zip(files, data, strict=True):
            report_file(f, d, ref, keys)
        sd = statistics.pstdev([ref[k][1] - ref[k][0] for k in keys])
        print(f"  Spread per token (reference, together - alone): {sd:.4f}")


if __name__ == "__main__":
    main()
