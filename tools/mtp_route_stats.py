#!/usr/bin/python3 -I
"""Keep list for the cut of the MTP head from its routing on coding load.

Reads the log of TONA_MTP_ROUTE_STATS (Metal backend: one line per MTP entry, "expert:weight" for each of the
chosen experts), ranks the experts by routing mass (sum of weights) and keeps the top N. Stability: the same
ranking over the first and the second half of the entries, the overlap of their keep sets and the mass each half
puts on the keep set of the whole. Output: {"keep": [...], "mass": {expert: share}, ...} for
tools/coder_requant/pipeline.py slice-mtp --keep-list.

mtp_route_stats.py LOG [--keep 164] [--out keep.json]
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def read(path: str) -> list[list[tuple[int, float]]]:
    rows = []
    for line in Path(path).read_text().splitlines():
        if line.strip():
            rows.append([(int(e), float(w)) for e, w in (p.split(":") for p in line.split())])
    return rows


def mass(rows: list[list[tuple[int, float]]]) -> Counter:
    m: Counter = Counter()
    for r in rows:
        for e, w in r:
            m[e] += w
    return m


def top(m: Counter, n: int) -> list[int]:
    return sorted(e for e, _ in sorted(m.items(), key=lambda kv: (-kv[1], kv[0]))[:n])


def share(m: Counter, keep: list[int]) -> float:
    total = sum(m.values())
    return sum(m[e] for e in keep) / total if total else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--keep", type=int, default=164)
    ap.add_argument("--out")
    a = ap.parse_args()
    rows = read(a.log)
    half = len(rows) // 2
    m, m1, m2 = mass(rows), mass(rows[:half]), mass(rows[half:])
    keep, k1, k2 = top(m, a.keep), top(m1, a.keep), top(m2, a.keep)
    total = sum(m.values())
    doc = {
        "keep": keep,
        "entries": len(rows),
        "experts_seen": len(m),
        "mass_on_keep": share(m, keep),
        "halves": {
            "overlap_of_keep_sets": len(set(k1) & set(k2)),
            "first_half_mass_on_keep": share(m1, keep),
            "second_half_mass_on_keep": share(m2, keep),
            "second_half_mass_on_first_half_keep": share(m2, k1),
        },
        "mass": {str(e): m[e] / total for e in sorted(m)},
    }
    text = json.dumps(doc, indent=1)
    if a.out:
        Path(a.out).write_text(text + "\n")
    print(json.dumps({k: v for k, v in doc.items() if k not in ("keep", "mass")}, indent=1))


if __name__ == "__main__":
    main()
