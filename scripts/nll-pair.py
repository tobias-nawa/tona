#!/usr/bin/python3 -I
"""Turn two TONA_TEACHER_NLL files ("pos nll" per line, tona-gpu-gen --teacher --prefill) into the dump format of
scripts/nll-compare.py, so that the same paired block bootstrap applies to prefill settings.

Usage: nll-pair.py REFERENCE.txt VARIANT.txt OUT_PREFIX
Writes OUT_PREFIX.ref.txt (reference against itself) and OUT_PREFIX.var.txt (variant against the reference);
then: nll-compare.py OUT_PREFIX.ref.txt OUT_PREFIX.var.txt. The prefill keeps no logits, so the KL column is 0 and
the top-1 column 1 in both files: read only the Δ(-log p) lines.
"""

import sys
from pathlib import Path


def load(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        pos, nll = line.split()
        rows[int(pos)] = float(nll)
    return rows


def main():
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    ref, var = load(sys.argv[1]), load(sys.argv[2])
    if ref.keys() != var.keys():
        sys.exit("the two files do not cover the same positions")
    out = sys.argv[3]
    Path(out + ".ref.txt").write_text("".join(f"0 {p} {v:.6f} {v:.6f} 0 1\n" for p, v in sorted(ref.items())))
    Path(out + ".var.txt").write_text("".join(f"0 {p} {ref[p]:.6f} {var[p]:.6f} 0 1\n" for p in sorted(var)))


if __name__ == "__main__":
    main()
