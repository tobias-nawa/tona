#!/usr/bin/python3 -I
"""Family B: write the removed body of a function; the repo's own tests decide (score.py runs them in the sandbox).

Usage: gen_b.py REPO   -> WORK/tasks/REPO-B.jsonl   (one task per hole of the shared context)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common

ASK = (
    "Reply with the complete definition of the function (signature, docstring and body) in one Python code block "
    "and nothing else."
)


def question(h: dict) -> str:
    f = h["func"]
    return (
        f'In {f["path"]}, the body of the function "{f["qualname"]}" was replaced by "...". '
        f"Write the function so that the package's tests pass."
    )


def main():
    repo = sys.argv[1]
    ctx = common.read_json(common.WORK / "ctx" / f"{repo}-ctx.json")
    out = common.WORK / "tasks" / f"{repo}-B.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for i, h in enumerate(sorted(ctx["holes"], key=lambda h: h["pos"])):
            task = {"id": f"{repo}-B{i:03d}", "family": "B", "repo": repo, "kind": "body", "question": question(h)}
            task.update(src_path=h["func"]["path"], src_line=h["func"]["start"], src_pos=h["pos"], func=h["func"])
            fh.write(json.dumps(task, ensure_ascii=False) + "\n")
    print(f"{repo}: {len(ctx['holes'])} tasks")


if __name__ == "__main__":
    main()
