#!/usr/bin/python3 -I
"""Pair runs against a reference run and print markdown tables (accuracy, flips, McNemar, Newcombe, time, power).

Usage: analyze.py REF_RUN RUN... [--select FILE] [--families A,B]
--select: JSON from select_tasks.py; only those task ids count.
Output per run and family: n, accuracy, discordant pairs (b: lost, c: gained), f = (b + c) / n, exact McNemar p,
95 % interval of the accuracy difference (Newcombe, paired), median seconds per task.
"""

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import stats


def load(label: str) -> dict:
    path = common.WORK / "runs" / f"{label}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if "correct" in r}


def ratio(x: float, n: int) -> float:
    return x / n if n else 0.0


def compare(ref: dict, run: dict, fam: str, ids) -> dict:
    keys = [i for i in sorted(ids) if i in ref and i in run and ref[i]["family"] == fam]
    pairs = [(bool(ref[i]["correct"]), bool(run[i]["correct"])) for i in keys]
    a, b, c = pairs.count((True, True)), pairs.count((True, False)), pairs.count((False, True))
    n = len(keys)
    diff, lo, hi = stats.newcombe_paired(a, b, c, n - a - b - c) if n else (0, 0, 0)
    secs = [run[i]["wall_s"] for i in keys] or [0]
    return {
        "n": n, "acc": ratio(a + c, n), "ref_acc": ratio(a + b, n), "b": b, "c": c, "f": ratio(b + c, n),
        "p": stats.mcnemar_exact(b, c), "diff": diff, "lo": lo, "hi": hi, "sec": statistics.median(secs),
    }  # fmt: skip


def row(label: str, fam: str, r: dict) -> str:
    return (
        f"| {label} | {fam} | {r['n']} | {100 * r['acc']:.1f} | {r['b']} | {r['c']} | {r['f']:.3f} | {r['p']:.3f} | "
        f"{100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, {100 * r['hi']:+.1f}] | {r['sec']:.1f} |"
    )


def main():
    args = sys.argv[1:]
    sel = None
    fams = "AB"
    if "--select" in args:
        i = args.index("--select")
        sel = set(common.read_json(Path(args[i + 1]))["ids"])
        del args[i : i + 2]
    if "--families" in args:
        i = args.index("--families")
        fams = args[i + 1].replace(",", "")
        del args[i : i + 2]
    ref = load(args[0])
    print("| run | family | n | acc % | b (lost) | c (gained) | f | McNemar p | diff pp [95 % Newcombe] | s/task |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for label in args[1:]:
        run = load(label)
        for fam in fams:
            print(row(label, fam, compare(ref, run, fam, sel if sel is not None else run.keys())))


if __name__ == "__main__":
    main()
