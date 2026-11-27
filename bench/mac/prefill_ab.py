#!/usr/bin/env python3
"""Prefill A/B on the Mac: tona-bench processes with alternating TONA_PREFILL_CHUNK arms (A B A B ...), the thermal
state (pmset -g therm) and free memory before and after each process, one JSON line per process into OUT.

Usage: bench/mac/prefill_ab.py OUT.jsonl --model M.gguf --arms 0,512 [--procs 3] [--prompt 1024] [--rounds 2]
An arm is CHUNK or BUILD:CHUNK (BUILD a directory under work/build, default mtl).
"""

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(__file__))
from synth_runs import ENV, ROUND, memory


def therm() -> str:
    """The warnings pmset recorded; 'none' while it has recorded none"""
    out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, env=ENV, check=False).stdout
    lines = [ln.strip() for ln in out.splitlines() if ln.strip() and "No " not in ln and "Note" not in ln]
    return "; ".join(lines) or "none"


def run(a: argparse.Namespace, arm: str) -> dict:
    build, chunk = arm.split(":") if ":" in arm else ("mtl", arm)
    cmd = [f"work/build/{build}/tona-bench", a.model, "-p", str(a.prompt), "-d", str(a.decode), "-b", "1",
           "-r", str(a.rounds), "--ctx", str(a.ctx)]  # fmt: skip
    rec = {"build": build, "chunk": int(chunk), "therm_before": therm(), "before": memory()}
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=ENV | {"TONA_PREFILL_CHUNK": chunk}, check=False)
    rec |= {"seconds": round(time.time() - t0, 1), "therm_after": therm(), "after": memory(), "exit": out.returncode}
    rec["rounds"] = [
        {"round": n, "prefill_tps": float(p), "decode_tps": float(d)} for n, p, d in ROUND.findall(out.stdout)
    ]
    if out.returncode:
        rec["error"] = out.stderr.strip()[-500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--model", required=True)
    ap.add_argument("--arms", default="0,512")
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--decode", type=int, default=16, help="decoded tokens after the prompt")
    ap.add_argument("--ctx", type=int, default=4096)
    ap.add_argument("--pause", type=float, default=30)
    a = ap.parse_args()
    order = [arm for _ in range(a.procs) for arm in a.arms.split(",")]
    for i, arm in enumerate(order):
        rec = run(a, arm)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        pre = [r["prefill_tps"] for r in rec["rounds"] if r["round"] != "warmup"]
        print(f"{arm}: prefill {pre}, therm {rec['therm_after']}", flush=True)
        if i + 1 < len(order):
            time.sleep(a.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
