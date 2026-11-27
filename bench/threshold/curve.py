#!/usr/bin/python3 -I
"""Dose-response: accuracy loss per stage against the quality metrics (the analysis fixed before the stage runs).

Usage: curve.py REF_RUN STAGES.json [--select FILE]
STAGES.json (nll_summary.py): {"stages": [{"label": [run labels], "full": number of full layers, "lm": {...},
              "bench": {...}, "repo": {"pydash": {...}, "boltons": {...}}}, ...]}  (mild to strong); each axis has
              mean/lo/hi of delta-NLL, kl, top1.
Options: --axis lm|bench (default lm: 32K texts; bench: tails of the two benchmark contexts), --repo NAME (tasks of
that repo only, with the delta-NLL of that repo's own tail), --select FILE.
Per stage and family: loss in pp against REF_RUN (paired, over all tasks unless --select), 95 % Newcombe interval,
exact McNemar p, and the zone: "safe" when the lower bound of the difference is above the equivalence margin
(A -3 pp, B -5 pp: the resolution the plan asked for), "damage" when the upper bound is below 0, else "open".
The first stage (mild to strong) in zone "damage" is the first with a significant loss. Rank correlation (Spearman)
of the loss with the delta-NLL and with the KL over the stages is printed last.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze
import common

MARGIN = {"A": -0.03, "B": -0.05}


def zone(fam: str, r: dict) -> str:
    if r["hi"] < 0:
        return "damage"
    return "safe" if r["lo"] > MARGIN[fam] else "open"


def rank(xs: list) -> list:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    out = [0.0] * len(xs)
    for r, i in enumerate(order):
        out[i] = float(r)
    return out


def spearman(x: list, y: list) -> float:
    rx, ry = rank(x), rank(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry, strict=True))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else 0.0


def parse(args: list) -> tuple:
    opts = {"--axis": "lm", "--repo": None, "--select": None}
    for opt in opts:
        if opt in args:
            i = args.index(opt)
            opts[opt] = args[i + 1]
            del args[i : i + 2]
    sel = set(common.read_json(Path(opts["--select"]))["ids"]) if opts["--select"] else None
    return args, opts["--axis"], opts["--repo"], sel


def print_row(s: dict, label: str, fam: str, r: dict, q: dict) -> None:
    print(
        f"| {s['full']} ({label}) | {fam} | {r['n']} | {100 * r['diff']:+.1f} [{100 * r['lo']:+.1f}, "
        f"{100 * r['hi']:+.1f}] | {r['b']} / {r['c']} | {r['p']:.3f} | {zone(fam, r)} | "
        f"{q['mean']:+.4f} [{q['lo']:+.4f}, {q['hi']:+.4f}] | {q['kl']:.4f} | {100 * q['top1']:.1f} |"
    )


def main():
    args, axis, repo, sel = parse(sys.argv[1:])
    ref = {k: v for k, v in analyze.load(args[0]).items() if repo is None or v["repo"] == repo}
    stages = common.read_json(Path(args[1]))["stages"]
    print("| full layers | family | n | loss pp [95 %] | b / c | McNemar p | zone | dNLL mean [95 %] | KL | top-1 % |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    series = {"A": [], "B": []}
    for s in stages:
        q = s["repo"][repo] if repo else s[axis]
        if q is None:
            continue
        for fam in "AB":
            for label in s["label"]:
                if not (common.WORK / "runs" / f"{label}.done").exists():
                    continue  # not run (yet)
                run = analyze.load(label)
                r = analyze.compare(ref, run, fam, sel if sel is not None else run.keys())
                print_row(s, label, fam, r, q)
            series[fam].append((r["diff"], q["mean"], q["kl"]))
    for fam, v in series.items():
        loss = [-d for d, _, _ in v]
        print(
            f"Spearman, family {fam}: loss vs dNLL {spearman(loss, [x for _, x, _ in v]):+.2f}, "
            f"loss vs KL {spearman(loss, [k for _, _, k in v]):+.2f} ({len(v)} stages)"
        )


if __name__ == "__main__":
    main()
