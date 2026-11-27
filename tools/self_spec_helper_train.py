#!/usr/bin/env python3
"""Distils the draft-side helper FFNs of the layer-skip drafter: per layer one SwiGLU FFN d -> h -> d that
approximates the routed experts
(variant A) or the whole MoE block (variant B) of the Coder from the data `tona-layer-skip --moe-dump` writes.

Dump layout (F16): [token][layer][x, routed, shared][d]. The helper file layout is in engine/cpu/ref_helper.h.

Usage: tools/self_spec_helper_train.py --data DUMP... --val DUMP... --h 256 --variant A --out helper.bin [--layers 0-47]
       [--epochs 150] [--device cuda|cpu]
"""

import argparse
import json
import struct
import sys
from pathlib import Path

N_LAYER = 48
D = 2560
MAGIC = b"TONAHLP1"


def load_dump(path: str, n_layer: int = N_LAYER, d: int = D):
    """Memory-mapped [token, layer, 3, d] F16 array of one dump file."""
    import numpy as np

    raw = np.memmap(path, dtype=np.float16, mode="r")
    per_token = n_layer * 3 * d
    if raw.size % per_token:
        raise ValueError(f"{path}: size is not a whole number of tokens")
    return raw.reshape(-1, n_layer, 3, d)


def layer_data(dumps, layer: int, variant: str):
    """(x, y) of one layer over all dump files as float32 arrays; y is routed (A) or routed + shared (B)."""
    import numpy as np

    xs, ys = [], []
    for a in dumps:
        xs.append(np.asarray(a[:, layer, 0, :], dtype=np.float32))
        y = np.asarray(a[:, layer, 1, :], dtype=np.float32)
        if variant == "B":
            y = y + np.asarray(a[:, layer, 2, :], dtype=np.float32)
        ys.append(y)
    return np.concatenate(xs), np.concatenate(ys)


def rel_error(pred, target) -> float:
    """||pred - target||^2 / ||target||^2 summed over the set."""
    num = float(((pred - target) ** 2).sum())
    den = float((target**2).sum())
    return num / max(den, 1e-30)


def train_layer(train, val, h: int, epochs: int, device: str, seed: int = 0):
    """Returns (gate, up, down as float32 arrays, validation relative error of the best epoch)."""
    import torch

    (x, y), (xv, yv) = train, val
    torch.manual_seed(seed)
    d = x.shape[1]
    dev = torch.device(device)
    xt, yt = torch.tensor(x, device=dev), torch.tensor(y, device=dev)
    xvt, yvt = torch.tensor(xv, device=dev), torch.tensor(yv, device=dev)
    scale = float(yt.pow(2).mean().sqrt()) + 1e-12
    gate = torch.nn.Linear(d, h, bias=False).to(dev)
    up = torch.nn.Linear(d, h, bias=False).to(dev)
    down = torch.nn.Linear(h, d, bias=False).to(dev)
    down.weight.data.mul_(0.1)  # a small start: the untrained helper stays near the shared-only draft (error 1)
    params = [*gate.parameters(), *up.parameters(), *down.parameters()]
    opt = torch.optim.AdamW(params, lr=2e-3, weight_decay=0.01)
    steps = epochs * max(1, (len(xt) + 255) // 256)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    def forward(a):
        return down(torch.nn.functional.silu(gate(a)) * up(a))

    best, best_w = float("inf"), None
    for _ in range(epochs):
        perm = torch.randperm(len(xt), device=dev)
        for i in range(0, len(xt), 256):
            idx = perm[i : i + 256]
            pred = forward(xt[idx])
            tgt = yt[idx]
            loss = ((pred - tgt) / scale).pow(2).mean() + 0.1 * (
                1 - torch.nn.functional.cosine_similarity(pred, tgt, dim=1).mean()
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            err = rel_error(forward(xvt), yvt)
        if err < best:
            best = err
            best_w = tuple(m.weight.detach().cpu().numpy().copy() for m in (gate, up, down))
    return (*best_w, best)


def write_helper(path: str, layers: list, whole: bool, d: int = D) -> None:
    """layers: per layer (gate [h, d], up [h, d], down [d, h]) float32."""
    h = layers[0][0].shape[0]
    with open(path, "wb") as f:
        f.write(MAGIC + struct.pack("<4I", len(layers), d, h, int(whole)))
        for gate, up, down in layers:
            for w in (gate, up, down):
                f.write(w.astype("<f4").tobytes())


def parse_layers(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", nargs="+", required=True)
    ap.add_argument("--val", nargs="+", required=True)
    ap.add_argument("--h", type=int, required=True)
    ap.add_argument("--variant", choices=["A", "B"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", default=f"0-{N_LAYER - 1}")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    train = [load_dump(p) for p in a.data]
    val = [load_dump(p) for p in a.val]
    layers, report = [], []
    for layer in parse_layers(a.layers):
        x, y = layer_data(train, layer, a.variant)
        xv, yv = layer_data(val, layer, a.variant)
        # the shared-only draft predicts zero for the routed part (A) and the shared part for B
        base = rel_error(0 * yv, yv) if a.variant == "A" else rel_error(_shared(val, layer), yv)
        gate, up, down, err = train_layer((x, y), (xv, yv), a.h, a.epochs, a.device)
        layers.append((gate, up, down))
        report.append({"layer": layer, "h": a.h, "variant": a.variant, "val_rel_err": err, "baseline_rel_err": base})
        print(json.dumps(report[-1]), flush=True)
    if len(layers) == N_LAYER:
        write_helper(a.out, layers, a.variant == "B")
    Path(a.out + ".json").write_text(json.dumps(report))
    return 0


def _shared(dumps, layer: int):
    import numpy as np

    return np.concatenate([np.asarray(a[:, layer, 2, :], dtype=np.float32) for a in dumps])


if __name__ == "__main__":
    sys.exit(main())
