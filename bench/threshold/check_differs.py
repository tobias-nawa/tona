#!/usr/bin/python3 -I
"""Plausibility check of a variant run: it must differ from the reference run, else the switch did nothing.

Usage: check_differs.py REF_RUN VARIANT_RUN   (exit 1 when every reply is identical)
Found the hard way: a stale server binary without TONA_IDX_SHARE gave 360 of 360 identical replies and identical decode
times (2026-10-29), which looks like "no quality loss" and is not.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def load(label: str) -> dict:
    path = common.WORK / "runs" / f"{label}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines()) if "reply" in r}


def main():
    ref, var = load(sys.argv[1]), load(sys.argv[2])
    ids = ref.keys() & var.keys()
    same = sum(ref[i]["reply"] == var[i]["reply"] for i in ids)
    print(f"{sys.argv[2]}: {same} of {len(ids)} replies identical to {sys.argv[1]}")
    if ids and same == len(ids):
        print("ABORT: the variant changed nothing; check the build and the environment", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
