#!/usr/bin/env python3
"""Bytes one decode token reads per layer of a GGUF model, from the tensor directory only (no weights loaded).

Per layer: dense bytes (attention or DeltaNet, hyper-connection, router, shared expert) and the bytes of one routed
expert; a decode token reads `dense + n_used * expert` of every layer it runs. Plus the output head and the final
hyper-connection, which every pass reads once however many layers are skipped.

Usage: tools/layer_cost.py FILE.gguf [--json]
(the second shard of a split model is not needed: the tensor directory of shard 1 lists every layer's tensors only
when it holds them, so pass the shard that holds the layers you ask about)
"""

import argparse
import json
import re
import sys

from gguf_info import read_gguf, tensor_bytes

LAYER = re.compile(r"^blk\.(\d+)\.(.+)$")


def layer_costs(g: dict) -> dict:
    """Returns {"layers": [{"layer", "dense", "expert", "kind"}...], "tail": bytes, "n_used": int, "n_expert": int}."""
    kv = g["kv"]
    arch = kv["general.architecture"]
    n_used = int(kv[f"{arch}.expert_used_count"])
    n_expert = int(kv[f"{arch}.expert_count"])
    layers: dict[int, dict] = {}
    tail = 0
    for t in g["tensors"]:
        b = tensor_bytes(t)
        if b is None:
            raise ValueError(f"tensor {t['name']}: unknown type {t['type']}")
        m = LAYER.match(t["name"])
        if not m:
            # token_embd is a row lookup, not read per token
            if not t["name"].startswith("token_embd"):
                tail += b
            continue
        row = layers.setdefault(int(m.group(1)), {"dense": 0, "expert": 0, "kind": "deltanet"})
        if "_exps." in m.group(2):
            row["expert"] += b // n_expert
        else:
            row["dense"] += b
        if m.group(2).startswith("attn_q."):
            row["kind"] = "qsa"
    return {
        "layers": [{"layer": i, **layers[i]} for i in sorted(layers)],
        "tail": tail,
        "n_used": n_used,
        "n_expert": n_expert,
    }


def decode_bytes(c: dict, kept: set[int] | None = None, n_used: int | None = None) -> int:
    """Bytes of one decode token that runs the layers in `kept` (all if None) with `n_used` routed experts."""
    n = c["n_used"] if n_used is None else n_used
    return c["tail"] + sum(r["dense"] + n * r["expert"] for r in c["layers"] if kept is None or r["layer"] in kept)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    c = layer_costs(read_gguf(a.path))
    if a.json:
        print(json.dumps(c, indent=1))
        return 0
    print(f"{len(c['layers'])} layers, top-{c['n_used']} of {c['n_expert']}")
    print(f"tail (head, final HC) {c['tail'] / 1e6:.1f} MB")
    print("layer kind      dense MB  expert MB  token MB")
    for r in c["layers"]:
        tok = (r["dense"] + c["n_used"] * r["expert"]) / 1e6
        print(f"{r['layer']:5d} {r['kind']:9s} {r['dense'] / 1e6:8.2f} {r['expert'] / 1e6:10.3f} {tok:9.2f}")
    print(f"full token: {decode_bytes(c) / 1e9:.3f} GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
