#!/usr/bin/env python3
"""Compares logit dumps of one token sequence (tona-logits-dump, llama-logits-dump) position by position.

Usage: logits_compare.py REF.bin OTHER.bin [OTHER.bin ...]

REF is the file the others are measured against. Per OTHER: the share of positions with the same top-1 token, the mean
and largest absolute logit difference (after removing each row's mean, which softmax does not see), KL(REF || OTHER) of
the softmax distributions (mean and largest) and the number of positions where the reference's top-1 is not in the
other's top-5. Positions are matched by their index in the sequence; all dumps must hold the same ones.
"""

import struct
import sys

import numpy as np


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns the position indices and the logits [count][vocab] of a dump."""
    with open(path, "rb") as f:
        vocab, count = struct.unpack("<II", f.read(8))
        rec = np.dtype([("i", "<i4"), ("v", "<f4", (vocab,))])
        data = np.frombuffer(f.read(), dtype=rec, count=count)
    return data["i"], data["v"].astype(np.float64)


def log_softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(axis=1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=1, keepdims=True))


def compare(ref: np.ndarray, other: np.ndarray) -> dict[str, float]:
    a, b = ref - ref.mean(axis=1, keepdims=True), other - other.mean(axis=1, keepdims=True)
    la, lb = log_softmax(ref), log_softmax(other)
    kl = (np.exp(la) * (la - lb)).sum(axis=1)
    top5 = np.argsort(-other, axis=1)[:, :5]
    top1 = ref.argmax(axis=1)
    return {
        "positions": float(len(ref)),
        "top1 equal %": 100.0 * float((top1 == other.argmax(axis=1)).mean()),
        "ref top-1 outside top-5 of other": float((~(top5 == top1[:, None]).any(axis=1)).sum()),
        "mean |dlogit|": float(np.abs(a - b).mean()),
        "max |dlogit|": float(np.abs(a - b).max()),
        "mean KL": float(kl.mean()),
        "max KL": float(kl.max()),
    }


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    idx, ref = load(argv[1])
    for path in argv[2:]:
        i2, other = load(path)
        if not np.array_equal(idx, i2):
            print(f"{path}: other positions than {argv[1]}")
            return 1
        print(f"{argv[1]} vs {path}")
        for key, value in compare(ref, other).items():
            print(f"  {key:34s} {value:.6g}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
