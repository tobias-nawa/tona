#!/usr/bin/env python3
"""Runs the process list of research/measurements/2026-11-11-mac-synth.md: tona-bench per row, a pause between rows,
free memory and swap before and after each, one JSON line per process (rounds, warmup, memory) into OUT.

Usage: bench/mac/synth_runs.py OUT.jsonl [--pause S] [--rows 1,2,...] (rows 11 to 20: the second set)
"""

import argparse
import json
import os
import re
import subprocess
import time

SYNTH = "work/models/synth/synth-a-e{}.gguf"
ROWS = {
    1: (128, "mtl", 1), 2: (128, "mtl", 2), 3: (128, "mtl", 3), 4: (128, "mtl", 4),
    5: (64, "mtl", 1), 6: (64, "mtl", 4), 7: (128, "mtl-fast", 1), 8: (128, "mtl-fast", 4),
    9: (128, "mtl", 1), 10: (128, "mtl", 4),
    # second set, the real expert count of variant A, 128 as control
    11: (256, "mtl", 1), 12: (256, "mtl", 2), 13: (256, "mtl", 3), 14: (256, "mtl", 4),
    15: (256, "mtl-fast", 1), 16: (256, "mtl-fast", 4), 17: (128, "mtl", 1), 18: (128, "mtl", 4),
    19: (256, "mtl", 1), 20: (256, "mtl", 4),
}  # fmt: skip
# Only the variables the plan names reach the program
ENV = {k: v for k, v in os.environ.items() if not k.startswith("TONA_")} | {"TONA_SEQS": "1", "LC_ALL": "C"}
ROUND = re.compile(r"(round \d+|warmup): prefill ([\d.]+) tok/s, decode ([\d.]+) tok/s")


def memory() -> dict:
    """Swap in use and free + inactive pages (what the system can hand out without swapping), in GB; page-ins and
    swap-ins since boot, whose growth over a process shows weights read again from the SSD"""
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True, env=ENV, check=False).stdout
    used = float(re.search(r"used = ([\d.]+)M", swap).group(1)) / 1024
    vm = subprocess.run(["vm_stat"], capture_output=True, text=True, check=False).stdout
    page = int(re.search(r"page size of (\d+)", vm).group(1))
    pages = {k: int(v) for k, v in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", vm)}
    counts = {k.lower(): int(v) for k, v in re.findall(r"(Pageins|Swapins):\s+(\d+)", vm)}
    return {"swap_gb": round(used, 2), "available_gb": round(sum(pages.values()) * page / 1e9, 2)} | counts


def run_row(row: int) -> dict:
    experts, build, width = ROWS[row]
    cmd = [f"work/build/{build}/tona-bench", SYNTH.format(experts), "-p", "256", "-d", "64", "-b", str(width),
           "-r", "3", "--ctx", "4096"]  # fmt: skip
    before = memory()
    t0 = time.time()
    out = subprocess.run(cmd, capture_output=True, text=True, env=ENV, check=False)
    rec = {"row": row, "experts": experts, "build": build, "width": width, "seconds": round(time.time() - t0, 1),
           "before": before, "after": memory(), "exit": out.returncode, "rounds": []}  # fmt: skip
    for name, pre, dec in ROUND.findall(out.stdout):
        rec["rounds"].append({"round": name, "prefill_tps": float(pre), "decode_tps": float(dec)})
    if out.returncode:
        rec["error"] = out.stderr.strip()[-500:]
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("--pause", type=float, default=30)
    ap.add_argument("--rows", default="1,2,3,4,5,6,7,8,9,10")
    a = ap.parse_args()
    rows = [int(r) for r in a.rows.split(",")]
    for i, row in enumerate(rows):
        rec = run_row(row)
        with open(a.out, "a") as f:
            f.write(json.dumps(rec) + "\n")
        dec = [r["decode_tps"] for r in rec["rounds"] if r["round"] != "warmup"]
        print(f"row {row}: {rec['experts']} experts, {rec['build']}, B={rec['width']}: decode {dec}", flush=True)
        if i + 1 < len(rows):
            time.sleep(a.pause)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
