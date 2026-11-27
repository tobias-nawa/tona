#!/usr/bin/python3 -I
"""Selection rule, fixed before the first variant run (README.md): by reference behaviour only, never by variants.

Usage: select_tasks.py SELECTION_RUN CLOSED_BOOK_RUN -> WORK/tasks/selected-<SELECTION_RUN>.json
A task is kept when the selection run (a reference run) solved it and the closed-book run (reference, repository
removed from the prompt) did not. The first drops tasks the reference never solves (a variant cannot lose them), the
second drops tasks that are solved without the long context (QSA damage cannot show). The comparison reference of the
pairing is another run, so that the selection does not bias it (regression to the mean).
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def load(label: str) -> dict:
    path = common.WORK / "runs" / f"{label}.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    sel, closed = load(sys.argv[1]), load(sys.argv[2])
    keep = [i for i, r in sel.items() if r.get("correct") and not closed.get(i, {}).get("correct")]
    out = common.WORK / "tasks" / f"selected-{sys.argv[1]}.json"
    common.write_json(out, {"selection_run": sys.argv[1], "closed_book_run": sys.argv[2], "ids": sorted(keep)})
    fam = {f: sum(1 for i in keep if sel[i]["family"] == f) for f in "AB"}
    print(f"kept {len(keep)} of {len(sel)}: {fam}")


if __name__ == "__main__":
    main()
