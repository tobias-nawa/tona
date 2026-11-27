#!/usr/bin/python3 -I
"""Summarise the delta-NLL runs of nll_all.sh: per stage and text, pooled over py / cpp / sh, and over the two
benchmark contexts. Writes WORK/nll/stages.json for curve.py and prints markdown tables.

Delta = -log p of the stage minus -log p of the reference on the same tokens (4,096 tail tokens per text, 64-token block
bootstrap with 4,000 draws, scripts/nll-compare.py); KL(reference || stage) and top-1 agreement are means.
"""

import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

NLL = common.WORK / "nll"
STAGES = [  # (tag, full layers, benchmark run labels)
    ("p0,1,3,4,6,7,9,10", 8, ["s8"]),
    ("p0,2,4,6,8,10", 6, ["s6", "s6b"]),
    ("p0,2,5,7,10", 5, ["s5"]),
    ("p0,3,6,9", 4, ["s4", "s4b"]),
    ("p0,6", 2, ["s2"]),
    ("p0", 1, ["s1"]),
]
GROUPS = {"lm": ["py-32768-4096", "cpp-32768-4096", "sh-32768-4096"], "bench": ["pydash-%d-4096", "boltons-%d-4096"]}


def load_compare():
    p = common.ROOT / "scripts" / "nll-compare.py"
    spec = importlib.util.spec_from_file_location("nll_compare", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(mod, names: list, tag: str) -> list:
    out = []
    for n in names:
        path = NLL / f"{n}-{tag}.txt"
        if path.exists() and (NLL / f"{tag}.done").exists():  # a stage still running has a partial file
            out += list(mod.load(path).values())
    return out


def stats_of(mod, r: list) -> dict:
    if not r:
        return {}
    m, lo, hi = mod.boot([x[1] - x[0] for x in r])
    return {
        "mean": m, "lo": lo, "hi": hi, "n": len(r),
        "kl": sum(x[2] for x in r) / len(r), "top1": sum(x[3] for x in r) / len(r),
    }  # fmt: skip


def bench_names() -> list:
    out = []
    for r, pat in zip(("pydash", "boltons"), GROUPS["bench"], strict=True):
        n = common.read_json(common.WORK / "ctx" / f"{r}-ctx.json")["tokens"]
        out.append(pat % (n - 4096))
    return out


def axes(mod, tag: str, names: dict) -> dict:
    """Quality axes of one stage for curve.py: pooled 32K texts ("lm"), pooled benchmark tails, per repo."""
    keys = ("mean", "lo", "hi", "kl", "top1")
    pick = lambda s: {k: s[k] for k in keys} if s else None  # noqa: E731
    per_repo = {
        r: pick(stats_of(mod, rows(mod, [n], tag))) for r, n in zip(("pydash", "boltons"), names["bench"], strict=True)
    }
    return {
        "lm": pick(stats_of(mod, rows(mod, names["lm"], tag))),
        "bench": pick(stats_of(mod, rows(mod, names["bench"], tag))),
        "repo": per_repo,
    }


def fmt(s: dict) -> str:
    if not s:
        return "-"
    return f"{s['mean']:+.4f} [{s['lo']:+.4f}, {s['hi']:+.4f}] | {s['kl']:.4f} | {100 * s['top1']:.1f}"


def main():
    mod = load_compare()
    names = {"lm": GROUPS["lm"], "bench": bench_names()}
    print("| full layers | texts | n tokens | dNLL mean [95 %] | KL | top-1 equal % |")
    print("|---|---|---|---|---|---|")
    out = []
    for tag, full, labels in STAGES:
        pooled = {g: stats_of(mod, rows(mod, ns, tag)) for g, ns in names.items()}
        for n in GROUPS["lm"] + names["bench"]:
            s = stats_of(mod, rows(mod, [n], tag))
            if s:
                print(f"| {full} | {n} | {s['n']} | {fmt(s)} |")
        for g, s in pooled.items():
            if s:
                print(f"| {full} | **pooled {g}** | {s['n']} | {fmt(s)} |")
        out.append({"label": labels, "full": full, **axes(mod, tag, names)})
    common.write_json(NLL / "stages.json", {"stages": out})


if __name__ == "__main__":
    main()
