#!/usr/bin/python3 -I
"""Write the shared context of a repo: the planned files with the bodies of the chosen functions replaced by `...`.

Usage: build.py REPO     (needs WORK/ctx/REPO-holes.json from holes.py)
Writes WORK/ctx/REPO.txt (the text of the system message after the instruction) and WORK/ctx/REPO-ctx.json
(files with holes, char offsets, token count, hole list). Deterministic: the same inputs give the same bytes.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import ctxlib


def main():
    repo = common.REPOS[sys.argv[1]]
    d = common.WORK / "ctx"
    spec = common.read_json(d / f"{repo.name}-holes.json")
    plan = ctxlib.plan_files(repo, spec["budget"])
    assert [p for p, _ in plan] == spec["files"], "plan changed since holes.py ran"
    files, _ = ctxlib.holed_files(plan, [ctxlib.Func(**h["func"]) for h in spec["holes"]])
    text, offsets = ctxlib.assemble(files)
    (d / f"{repo.name}.txt").write_text(text)
    common.write_json(
        d / f"{repo.name}-ctx.json",
        {
            "repo": repo.name,
            "commit": repo.commit,
            "license": repo.license,
            "tokens": common.count_tokens(text),
            "chars": len(text),
            "offsets": offsets,
            "files": files,
            "holes": spec["holes"],
        },
    )
    print(f"{repo.name}: {len(text)} chars, {common.count_tokens(text)} tokens, {len(spec['holes'])} holes")


if __name__ == "__main__":
    main()
