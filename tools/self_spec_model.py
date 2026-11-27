#!/usr/bin/env python3
"""Tokens per ms of a layer-skip draft, alone and behind the MTP head, from the offline acceptance of tona-layer-skip
and the pass costs of the engine.

A cycle = one verify pass over 1 + k + e positions (k from the MTP chain, e from the layer-skip chain), the MTP fill
and k MTP steps, then the layer-skip steps: one that carries the k+1 known tokens through the kept layers as a batch
and e - 1 single steps (one step if k = 0 and e = 1). Memory-bound decode: a kept layer costs its bytes, so one
layer-skip step takes its share of a single-token pass (tools/layer_cost.py), the head included.

Usage: tools/self_spec_model.py RESULT.jsonl --gguf MODEL.gguf --masks MASKS.txt --mtp-survival 0.9,0.8,...
       [--scale 1.77] [--helper-mb MB]
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# generator.h defaults, measured on the reference: verify pass over 1..8 positions, MTP step, MTP fill (ms)
PASS_MS = [0.0, 23.7, 28.9, 32.5, 35.9, 41.7, 46.1, 49.6, 52.6]
MTP_STEP_MS = 1.15
MTP_FILL_MS = 1.3


def load_drafts(path: str) -> dict[str, list[list[int]]]:
    """{mask: [match flags per start position]} from the JSONL of tona-layer-skip."""
    out = defaultdict(list)
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if r["kind"] == "draft":
                out[r["mask"]].append(r["match"])
    return dict(out)


def run_survival(samples: list[list[int]], start: int, length: int) -> float:
    """Fraction of samples whose flags are all 1 on indices start .. start + length - 1 (index i judges token i + 1)."""
    if not samples:
        return 0.0
    ok = sum(1 for m in samples if len(m) >= start + length and all(m[start : start + length]))
    return ok / len(samples)


def survival_curve(samples: list[list[int]], depth: int) -> list[float]:
    """S_j = P(the first j drafts all match), j = 1..depth."""
    return [run_survival(samples, 0, j) for j in range(1, depth + 1)]


def pass_ms(width: int) -> float:
    """Verify pass over `width` positions; beyond the table the last step is extended linearly."""
    if width < len(PASS_MS):
        return PASS_MS[width]
    return PASS_MS[-1] + (width - (len(PASS_MS) - 1)) * (PASS_MS[-1] - PASS_MS[-2])


def rate(k: int, e: int, mtp_s: list[float], skip: list[list[int]], step_ms: float, scale: float = 1.0) -> float:
    """Tokens per ms for an MTP chain of k and a layer-skip extension of e (either may be 0). `step_ms` is one
    layer-skip step, `scale` the factor of the class on every verify pass (cold experts: 1.77 at the 32 GB class)."""
    if k > len(mtp_s):
        raise ValueError("mtp_s shorter than k")
    s_k = 1.0 if k == 0 else mtp_s[k - 1]
    tokens = 1.0 + sum(mtp_s[:k]) + s_k * sum(run_survival(skip, k, i) for i in range(1, e + 1))
    cost = pass_ms(1 + k + e) * scale
    if k:
        cost += MTP_FILL_MS + k * MTP_STEP_MS
    return tokens / (cost + e * step_ms)


def best(mtp_s: list[float], skip: list[list[int]], step_ms: float, scale: float = 1.0, kmax: int = 7, emax: int = 4):
    """(rate, k, e) of the best scheme; e = 0 is the MTP-only baseline."""
    ks = range(0, min(kmax, len(mtp_s)) + 1)
    return max((rate(k, e, mtp_s, skip, step_ms, scale), k, e) for k in ks for e in range(0, emax + 1))


def breakeven_step_ms(mtp_s: list[float], skip: list[list[int]], scale: float = 1.0, hi: float = 60.0) -> float:
    """Largest layer-skip step cost (ms) at which the best scheme with an extension still beats MTP-only; 0 if none."""
    base = best(mtp_s, [], 0.0, scale)[0]
    lo = 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if best(mtp_s, skip, mid, scale)[0] > base * 1.0001:
            lo = mid
        else:
            hi = mid
    return lo


def parse_mask(line: str, n_layer: int = 48) -> dict:
    """{"name", "kept": set, "n_used": int | None (None: the model's), "helper": str | None} of a masks-file line."""
    name, *terms = line.split()
    m = {"name": name, "kept": set(range(n_layer)), "n_used": None, "helper": None}
    for t in terms:
        if t.startswith("skip="):
            m["kept"] -= {int(x) for x in t[5:].split(",")}
        elif t.startswith("topk="):
            m["n_used"] = int(t[5:])
        elif t == "routed=off":
            m["n_used"] = 0
        elif t.startswith("helper="):
            m["helper"] = t[7:]
            m["n_used"] = 0
    return m


def step_ms(c: dict, mask: dict, helper_bytes: int = 0, scale: float = 1.0) -> float:
    """One layer-skip step in ms: the single-token pass (PASS_MS[1]) split by bytes; the expert bytes cost `scale`
    times as much per byte as the dense bytes would at that class' extra (cold experts), see module docstring."""
    from layer_cost import decode_bytes

    n_used = c["n_used"] if mask["n_used"] is None else mask["n_used"]
    layers = [r for r in c["layers"] if r["layer"] in mask["kept"]]
    dense = c["tail"] + sum(r["dense"] for r in layers) + helper_bytes * len(layers)
    experts = sum(n_used * r["expert"] for r in layers)
    full_experts = c["n_used"] * sum(r["expert"] for r in c["layers"])
    total = decode_bytes(c)
    return PASS_MS[1] * (dense + experts) / total + (scale - 1) * PASS_MS[1] * experts / full_experts


def main() -> int:
    from gguf_info import read_gguf
    from layer_cost import layer_costs

    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--masks", required=True, help="the masks file given to tona-layer-skip")
    ap.add_argument("--mtp-survival", required=True, help="S_1..S_k of the MTP chain, comma separated")
    ap.add_argument("--scale", type=float, default=1.0, help="pass cost factor of the class (reference 1, 32 GB 1.77)")
    ap.add_argument("--helper-mb", type=float, default=0.0, help="bytes of one layer's helper, MB (Q4 weights)")
    a = ap.parse_args()
    mtp = [float(x) for x in a.mtp_survival.split(",")]
    c = layer_costs(read_gguf(a.gguf))
    drafts = load_drafts(a.path)
    r0 = best(mtp, [], 0.0, a.scale)
    print(f"MTP only: {r0[0] * 1000:.1f} tok/s (k={r0[1]}) model, scale {a.scale}")
    for line in Path(a.masks).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        mask = parse_mask(line)
        samples = drafts.get(mask["name"], [])
        ms = step_ms(c, mask, int(a.helper_mb * 1e6) if mask["helper"] else 0, a.scale)
        r1 = best(mtp, samples, ms, a.scale)
        curve = [round(x, 2) for x in survival_curve(samples, 6)]
        print(
            f"{mask['name']}: n={len(samples)} step {ms:.1f} ms, S={curve}, with skip {r1[0] * 1000:.1f} tok/s "
            f"(k={r1[1]}, e={r1[2]}, {100 * (r1[0] / r0[0] - 1):+.1f} %)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
