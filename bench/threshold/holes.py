#!/usr/bin/python3 -I
"""Choose the functions whose bodies are removed from the context (family B), by tests and position only.

Usage: holes.py REPO
1. run the suite with coverage in the sandbox, 2. plan the files of the context, 3. candidates: 3-35 body lines, at
least 80 % of them executed, 4. mutation check in the sandbox: with the body replaced by `raise NotImplementedError`
at least one test must fail, 5. every function that passes becomes a hole, except those within 2,400 tokens of the end
(QSA selects nothing below 2,051 positions, and a task there would be local). The file budget is adjusted until the
context WITH the holes is about TARGET tokens, so the prompt stays within 64K. Writes WORK/ctx/REPO-holes.json.
"""

import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common
import ctxlib
import sandbox

MIN_LINES, MAX_LINES, MIN_COVERED = 3, 35, 0.8
TAIL_MARGIN = 2400  # > QSA_MIN plus the question after the context
TARGET, TOLERANCE = 62500, 600  # tokens of the holed context; plus instruction and question it stays below 64K


def covered_fraction(cov: dict, path: str, f: ctxlib.Func) -> float:
    executed, missing = cov.get(path, [[], []])
    span = range(f.body_start, f.body_end + 1)
    e = sum(1 for x in executed if x in span)
    m = sum(1 for x in missing if x in span)
    return e / (e + m) if e + m else 0.0


def candidates(files, cov) -> list:
    out = []
    for path, src in files:
        for f in ctxlib.functions_of(path, src):
            sized = MIN_LINES <= f.body_lines <= MAX_LINES and f.body_start > f.start
            if sized and covered_fraction(cov, path, f) >= MIN_COVERED:
                out.append(f)
    return out


def fails_without_body(repo, files, f: ctxlib.Func) -> bool:
    src = dict(files)[f.path]
    stub = ctxlib.hole_text(src, f).replace(" " * f.col + "...\n", " " * f.col + "raise NotImplementedError\n", 1)
    res = sandbox.run_tests(repo, {f.path: stub}, timeout=90)
    return res["rc"] not in (0, None) and not res["timed_out"]


def holed_context(files, funcs):
    holed, where = ctxlib.holed_files(files, funcs)
    text, offsets = ctxlib.assemble(holed)
    return holed, where, text, offsets, common.count_tokens(text)


def settle(repo, cov, cache, budget) -> tuple:
    """Plan with this budget; returns (files, holes with positions, tokens of the holed context)."""
    files = ctxlib.plan_files(repo, budget)
    todo = [f for f in candidates(files, cov) if (f.path, f.start, f.body_end) not in cache]
    with ThreadPoolExecutor(4) as ex:
        for f, ok in zip(todo, ex.map(lambda f: fails_without_body(repo, files, f), todo), strict=True):
            cache[(f.path, f.start, f.body_end)] = ok
    funcs = [f for f in candidates(files, cov) if cache[(f.path, f.start, f.body_end)]]
    for _ in range(2):  # dropping holes near the end moves the end a little; once more is enough
        holed, where, text, offsets, total = holed_context(files, funcs)
        pos = {
            (f.path, f.qualname): ctxlib.line_token_pos(holed, offsets, text, f.path, where[(f.path, f.qualname)])
            for f in funcs
        }
        keep = [f for f in funcs if pos[(f.path, f.qualname)] <= total - TAIL_MARGIN]
        if len(keep) == len(funcs):
            break
        funcs = keep
    holes = [{"func": f.__dict__, "pos": pos[(f.path, f.qualname)]} for f in funcs]
    return files, holes, total


def main():
    repo = common.REPOS[sys.argv[1]]
    base = sandbox.run_tests(repo, {}, coverage=True)
    assert base["rc"] == 0, base["tail"]
    cache, budget = {}, common.CONTEXT_TOKENS
    for _ in range(5):
        files, holes, total = settle(repo, base["coverage"], cache, budget)
        print(f"budget {budget}: {len(files)} files, {len(holes)} holes, {total} tokens with holes", flush=True)
        if abs(total - TARGET) <= TOLERANCE:
            break
        budget += TARGET - total
    common.write_json(
        common.WORK / "ctx" / f"{repo.name}-holes.json",
        {"commit": repo.commit, "budget": budget, "files": [p for p, _ in files], "holes": holes},
    )
    print(f"{repo.name}: {len(holes)} holes, {total} tokens, last file {files[-1][0]}")


if __name__ == "__main__":
    main()
