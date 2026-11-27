#!/usr/bin/env python3
"""Sanity check of a final variant: each requantized expert resembles the Coder expert it claims to be.

Dequantizes ffn_gate_exps of sampled (layer, expert) pairs from the final file and from the Coder (keep[layer][j]) and
compares cosine similarity against the best other Coder expert. A wrong index mapping shows as a matched cosine no
better than an unmatched one. Needs numpy and llama.cpp's gguf-py (--gguf-py); not part of the unit tests.

  check_experts.py --gguf-py LLAMA/gguf-py --coder-trunk CODER-00001.gguf --final coder-A-256e.gguf --keep keep.json
"""

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from coder_requant import select_experts, sources

MIN_COSINE = 0.8  # same weights at ~2-3.5 bits on both sides; unrelated experts sit near 0
MIN_MARGIN = 0.3  # matched minus best unmatched


def expert(np, quants, reader, layout, name: str, e: int):
    t = layout.by_name[name]
    per = t.nbytes // t.dims[-1]
    raw = reader.read(layout.data_start + t.offset + e * per, per)
    vals = quants.dequantize(np.frombuffer(raw, dtype=np.uint8), t.type)
    return vals.reshape(-1).astype(np.float64)


def cosine(np, a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--gguf-py", required=True), p.add_argument("--coder-trunk", required=True)
    p.add_argument("--final", required=True), p.add_argument("--keep", required=True)
    p.add_argument("--samples", type=int, default=24), p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()
    sys.path.insert(0, args.gguf_py)
    import numpy as np
    from gguf import quants

    keep = select_experts.read_keep(args.keep)
    cr, fr = sources.Reader(Path(args.coder_trunk)), sources.Reader(Path(args.final))
    cl, fl = cr.layout(), fr.layout()
    rng = random.Random(args.seed)
    bad = 0
    for _ in range(args.samples):
        layer = rng.randrange(len(keep))
        j = rng.randrange(len(keep[layer]))
        name = f"blk.{layer}.ffn_gate_exps.weight"
        mine = expert(np, quants, fr, fl, name, j)
        want = keep[layer][j]
        others = rng.sample([e for e in range(cl.by_name[name].dims[-1]) if e != want], 3)
        matched = cosine(np, mine, expert(np, quants, cr, cl, name, want))
        other = max(cosine(np, mine, expert(np, quants, cr, cl, name, e)) for e in others)
        ok = matched >= MIN_COSINE and matched - other >= MIN_MARGIN
        bad += not ok
        verdict = "" if ok else " FAIL"
        print(f"layer {layer:2d} expert {j:3d} <- {want:3d}: cos {matched:.3f}, best other {other:.3f}{verdict}")
    if bad:
        raise SystemExit(f"{bad} of {args.samples} samples failed")


if __name__ == "__main__":
    main()
