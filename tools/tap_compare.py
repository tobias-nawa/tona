#!/usr/bin/env python3
"""Compare the GPU's per-layer tap (TONA_TAP_OUT, gpu/mega_tap.h) with the oracle's (TONA_REF_TRACE_OUT of tona-check).

Both files hold, per pass, int32 {slots, d, nb, pos} and then slots x nb x d float32; slot 2l follows the mixer of layer
l, slot 2l + 1 the whole layer. Prints, per slot, the relative L2 difference ||g - o|| / ||o|| over the steps (median
and maximum) and, per step, the first slot above the threshold.

    tools/tap_compare.py GPU.tap ORACLE.tap [--threshold 1e-3] [--steps 0,1,50]
"""

import argparse
import struct
import sys

import numpy as np


def read(path):
    steps = []
    with open(path, "rb") as f:
        while h := f.read(16):
            slots, d, nb, pos = struct.unpack("<4i", h)
            a = np.fromfile(f, dtype="<f4", count=slots * nb * d)
            steps.append((pos, a.reshape(slots, nb, d)[:, 0, :].astype(np.float64)))
    return steps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu")
    ap.add_argument("oracle")
    ap.add_argument("--threshold", type=float, default=1e-3)
    ap.add_argument("--steps", default="")
    ap.add_argument(
        "--updates", action="store_true", help="also the error of each block's own update (slot minus the slot before)"
    )
    a = ap.parse_args()
    g, o = read(a.gpu), read(a.oracle)
    n = min(len(g), len(o))
    print(f"{len(g)} GPU passes, {len(o)} oracle steps, comparing {n}")
    rel = np.zeros((n, g[0][1].shape[0]))
    for i in range(n):
        gp, gv = g[i]
        op, ov = o[i]
        if gp != op:
            sys.exit(f"step {i}: position {gp} against {op}")
        rel[i] = np.linalg.norm(gv - ov, axis=1) / np.maximum(np.linalg.norm(ov, axis=1), 1e-30)
    print("slot  where        median     p90        max")
    for s in range(rel.shape[1]):
        name = f"L{s // 2:02d} {'mixer' if s % 2 == 0 else 'layer'}"
        print(f"{s:3d}   {name}  {np.median(rel[:, s]):.2e}  {np.percentile(rel[:, s], 90):.2e}  {rel[:, s].max():.2e}")
    first = [int(np.argmax(r > a.threshold)) if (r > a.threshold).any() else -1 for r in rel]
    vals, counts = np.unique(first, return_counts=True)
    print("first slot above", a.threshold, "(slot: steps):", dict(zip(vals.tolist(), counts.tolist(), strict=True)))
    if a.updates:
        print("slot  where        update error ||dg - do|| / ||do||: median, p90 over steps")
        for s in range(1, rel.shape[1]):
            e = []
            for i in range(n):
                dg, do = g[i][1][s] - g[i][1][s - 1], o[i][1][s] - o[i][1][s - 1]
                e.append(np.linalg.norm(dg - do) / max(np.linalg.norm(do), 1e-30))
            name = f"L{s // 2:02d} {'mixer' if s % 2 == 0 else 'moe  '}"
            print(f"{s:3d}   {name}  {np.median(e):.2e}  {np.percentile(e, 90):.2e}")
    for s in [int(x) for x in a.steps.split(",") if x]:
        print(f"step {s}:", " ".join(f"{x:.1e}" for x in rel[s]))


if __name__ == "__main__":
    main()
