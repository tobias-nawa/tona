#!/usr/bin/env python3 -I
"""Tables of the needle test.

report.py summary FILE.jsonl ...                       hit rate with Wilson interval per variant and length
report.py compare --a off --b top1024 FILE.jsonl ...   paired difference b - a per length (McNemar, exact)
report.py plan [--pairs N]                             pairs needed for a given difference
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import stats

BIN_NAMES = ["0-10 %", "25 %", "50 %", "75 %", "90-100 %"]


def load(files: list[str]) -> list[dict]:
    rows = []
    for f in files:
        for line in Path(f).read_text().splitlines():
            r = json.loads(line)
            if "hit" in r:
                rows.append(r)
    return rows


def rate(rows: list[dict]) -> str:
    k, n = sum(r["hit"] for r in rows), len(rows)
    lo, hi = stats.wilson(k, n)
    return f"{k:>3}/{n:<3} {100 * k / max(n, 1):5.1f} % [{100 * lo:4.1f}, {100 * hi:5.1f}]"


def group(rows: list[dict], key) -> dict:
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return dict(sorted(out.items()))


def variant_key(r: dict) -> tuple:
    return r["label"] + ("+hard" if r.get("hard") else ""), r["length"]


def summary(rows: list[dict]) -> None:
    print("variant  length   all                      single                   multi                  texts")
    for (label, length), g in group(rows, variant_key).items():
        single = [r for r in g if r["mode"] == "single"]
        multi = [r for r in g if r["mode"] == "multi"]
        print(f"{label:<8} {length:>7}  {rate(g)}  {rate(single)}  {rate(multi)}  {len({r['text'] for r in g})}")
    print("\nby depth (all needles):")
    print("variant  length  " + "  ".join(f"{b:<27}" for b in BIN_NAMES))
    for (label, length), g in group(rows, variant_key).items():
        cells = [rate([r for r in g if r["bin"] == b]) for b in range(len(BIN_NAMES))]
        print(f"{label:<8} {length:>7}  " + "  ".join(cells))


def pair_key(r: dict) -> tuple:
    return r["length"], r["text"], r["needle"], r["mode"], bool(r.get("hard"))


def pair_row(ra: dict, rb: dict, keys: list) -> str:
    only_a = sum(ra[k] and not rb[k] for k in keys)
    only_b = sum(rb[k] and not ra[k] for k in keys)
    both = sum(ra[k] and rb[k] for k in keys)
    d, lo, hi = stats.paired_diff(only_a, only_b, len(keys))
    p = stats.mcnemar_exact(only_a, only_b)
    none = len(keys) - both - only_a - only_b
    diff = f"{100 * d:+6.1f} [{100 * lo:+6.1f}, {100 * hi:+6.1f}]"
    return f"{len(keys):>4}  {both:>5} {only_a:>7} {only_b:>7} {none:>5}   {diff}   {p:.3f}"


def compare(rows: list[dict], a: str, b: str) -> None:
    ra = {pair_key(r): r["hit"] for r in rows if r["label"] == a}
    rb = {pair_key(r): r["hit"] for r in rows if r["label"] == b}
    print(f"{b} minus {a}, paired over the needles both have; pairs: both right / only {a} / only {b} / both wrong")
    print("length  mode    pairs   both  only-a  only-b  none   diff pp [95 %]            McNemar p")
    for length in sorted({k[0] for k in ra if k in rb}):
        for mode in ("single", "multi", "all"):
            keys = [k for k in ra if k in rb and k[0] == length and mode in (k[3], "all")]
            print(f"{length:>7} {mode:<7} {pair_row(ra, rb, keys)}")


def plan(pairs: int | None) -> None:
    print("pairs for McNemar (alpha 0.05, power 0.8) to find a difference d; psi = share of discordant pairs")
    print("psi    " + "".join(f"d={d:.2f}  " for d in (0.10, 0.15, 0.20, 0.25)))
    for psi in (0.15, 0.25, 0.35, 0.50):
        print(f"{psi:.2f}   " + "".join(f"{stats.pairs_needed(psi, d):>6}  " for d in (0.10, 0.15, 0.20, 0.25)))
    if pairs:
        found = [f"{100 * stats.detectable(psi, pairs):.0f} pp at psi {psi:.2f}" for psi in (0.15, 0.25, 0.35, 0.5)]
        print(f"\nwith {pairs} pairs the smallest difference found: " + ", ".join(found))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["summary", "compare", "plan"])
    ap.add_argument("files", nargs="*")
    ap.add_argument("--a", help="first variant (label)")
    ap.add_argument("--b", help="second variant (label)")
    ap.add_argument("--pairs", type=int)
    args = ap.parse_args()
    if args.cmd == "plan":
        return plan(args.pairs)
    rows = load(args.files)
    if args.cmd == "summary":
        summary(rows)
    else:
        compare(rows, args.a, args.b)


if __name__ == "__main__":
    main()
