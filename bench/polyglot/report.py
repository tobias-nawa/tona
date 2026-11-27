#!/usr/bin/env python3 -I
"""Pass rates with 95 % Wilson intervals per run file, paired exact McNemar tests between runs, tokens and time.

    report.py NAME=FILE.jsonl [NAME=FILE.jsonl ...]   (the first run is the reference of the paired tests)

Only exercises present in every file are compared. Standard library only.
"""

import json
import math
import sys
from pathlib import Path

Z = 1.959964


def wilson(k: int, n: int) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return c - h, c + h


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial test on the discordant pairs (b: only first passes, c: only second passes)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def load(path: str) -> dict[str, dict]:
    return {r["id"]: r for r in map(json.loads, Path(path).read_text().splitlines())}


def by_lang(rows: list[dict], key: str) -> str:
    out = []
    for lang in sorted({r["id"].split("/")[0] for r in rows}):
        sub = [r for r in rows if r["id"].startswith(lang + "/")]
        out.append(f"{lang} {sum(r[key] for r in sub)}/{len(sub)}")
    return ", ".join(out)


def summarize(name: str, r: dict, ids: list[str]) -> None:
    rows = [r[i] for i in ids]
    att = [a for x in rows for a in x["attempts"]]
    for key in ("passed1", "passed2"):
        k = sum(x[key] for x in rows)
        lo, hi = wilson(k, len(rows))
        pct = f"{100 * k / len(rows):.1f} % [{100 * lo:.1f}, {100 * hi:.1f}]"
        print(f"{name} {key}: {k}/{len(rows)} = {pct}  ({by_lang(rows, key)})")
    toks = sorted(a["completion_tokens"] or 0 for a in att)
    secs = sum(a["seconds"] for a in att)
    trunc = sum(a["finish"] == "length" for a in att)
    print(
        f"{name}: attempts {len(att)}, generated tokens median {toks[len(toks) // 2]} max {toks[-1]} "
        f"total {sum(toks)}, truncated {trunc}, wall {secs / 60:.1f} min"
    )


def paired(names: list[str], runs: dict, ids: list[str]) -> None:
    for other in names[1:]:
        for key in ("passed1", "passed2"):
            b = sum(runs[names[0]][i][key] and not runs[other][i][key] for i in ids)
            c = sum(runs[other][i][key] and not runs[names[0]][i][key] for i in ids)
            p = mcnemar_exact(b, c)
            print(f"{names[0]} vs {other} {key}: only first {b}, only second {c}, exact McNemar p = {p:.3f}")


def main() -> int:
    runs = {a.split("=", 1)[0]: load(a.split("=", 1)[1]) for a in sys.argv[1:]}
    ids = sorted(set.intersection(*(set(r) for r in runs.values())))
    print(f"{len(ids)} exercises in common")
    for name, r in runs.items():
        summarize(name, r, ids)
    paired(list(runs), runs, ids)
    return 0


if __name__ == "__main__":
    sys.exit(main())
