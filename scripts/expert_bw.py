#!/usr/bin/env python3
"""Achieved bandwidth of the expert matvec steps per quant type and pass width.

Reads the log of `tona-gpu-gen MODEL PROMPT --mega --batch-bench N` run with TONA_MEGA_PROF=1 TONA_PROF_LAYERS=1
(per-layer step times `moe.gate_up#l`, `moe.down#l`, the mean GPU-computed experts `nu#l` of each layer and, of
them, the ones read over PCIe `gd#l`) and the
GGUF directory of the model, and prints per (step, type) and width: time, bytes read by the GPU, GB/s. Bytes are
exact: experts read from VRAM in that layer (computed on the GPU, minus the GPU-direct ones that read pinned host
memory over PCIe) times the bytes of one expert (gate and up rows for moe.gate_up, down rows for moe.down). Experts
computed on the CPU are not counted; the step time of a layer with GPU-direct experts includes their PCIe reads, so run
it with TONA_PLAN=cpu (no GPU-direct experts) for a clean VRAM figure.

Usage: scripts/expert_bw.py MODEL.gguf LOG [LOG ...]    (several logs: median over their profiles)
"""

import re
import statistics
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
# type name -> (values per block, bytes per block), engine/model/gguf.cpp
BLOCK = {
    "IQ3_XXS": (256, 98),
    "IQ2_S": (256, 82),
    "IQ3_S": (256, 110),
    "IQ4_XS": (256, 136),
    "IQ4_NL": (32, 18),
    "Q2_0": (64, 18),
}
CEILING_GBS = 315.0  # measured VRAM read bandwidth of the reference GPU (docs/backends/reference-platform.md)


def layer_types(model):
    """Per layer: (gate type, up type, down type, in length of gate, in length of down, rows gate, rows down)."""
    out = subprocess.run(
        [sys.executable, str(ROOT / "tools/gguf_info.py"), "--tensors", model],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    layers = defaultdict(dict)
    for m in re.finditer(r"blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight\s+(\S+)\s+\[(\d+), (\d+), (\d+)\]", out):
        layers[int(m[1])][m[2]] = (m[3], int(m[4]), int(m[5]))
    return layers


def expert_bytes(t):
    name, inlen, rows = t
    values, nbytes = BLOCK[name]
    return rows * inlen // values * nbytes


def parse(log):
    """Profiles of a log: {width: {label: value}} (last profile per width wins; repeats are separate calls)."""
    prof, cur = [], None
    for line in Path(log).read_text().splitlines():
        if m := re.match(r"Profile B = (\d+): ([\d.]+) ms", line):
            cur = {"w": int(m[1]), "total": float(m[2]), "v": {}}
            prof.append(cur)
        elif cur and (m := re.match(r"\s+(\S+#\d+)\s+(\d+) ", line)):
            cur["v"][m[1]] = float(m[2])
    return prof


def main():
    model, logs = sys.argv[1], sys.argv[2:]
    layers = layer_types(model)
    acc = defaultdict(lambda: defaultdict(list))
    for log in logs:
        for p in parse(log):
            for k, v in p["v"].items():
                acc[p["w"]][k].append(v)
    print("| step | type | W | layers | experts per pass | MB per pass | ms per pass | GB/s | % of 315 GB/s |")
    print("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for w in sorted(acc):
        mean = {k: statistics.median(v) for k, v in acc[w].items()}  # median: one stalled profile must not count
        for step, parts in (("moe.gate_up", ("gate", "up")), ("moe.down", ("down",))):
            groups = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
            for lay, d in layers.items():
                key = d[parts[0]][0]
                g = groups[key]
                nu = mean.get(f"nu#{lay}", 0.0) / 1e3 - mean.get(f"gd#{lay}", 0.0) / 1e3  # VRAM hits
                g[0] += 1
                g[1] += nu
                g[2] += nu * sum(expert_bytes(d[p]) for p in parts) / 1e6
                g[3] += mean.get(f"{step}#{lay}", 0.0) / 1e3
            tot = [0, 0.0, 0.0, 0.0]
            for key, g in sorted(groups.items()):
                row(step, key, w, g)
                tot = [a + b for a, b in zip(tot, g, strict=True)]
            row(step, "all", w, tot)


def row(step, key, w, g):
    gbs = g[2] / g[3] if g[3] else 0.0  # MB per ms = GB/s
    print(
        f"| {step} | {key} | {w} | {g[0]} | {g[1]:.0f} | {g[2]:.0f} | {g[3]:.2f} | {gbs:.0f} "
        f"| {100 * gbs / CEILING_GBS:.0f} |"
    )


if __name__ == "__main__":
    main()
