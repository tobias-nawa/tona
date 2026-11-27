#!/usr/bin/python3 -I
"""Score family B: put the generated body into the real file and run the repo's tests in the sandbox.

Usage: score.py LABEL            adds `correct` (and `why`) to the family B lines of WORK/runs/LABEL.jsonl
       score.py --selftest REPO  feeds the original function text as the "reply" of every task; all must pass
A task passes when the whole test suite of the repo passes with that one body replaced (the other holes keep their
original bodies: tasks are independent of each other).
"""

import ast
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answers
import common
import ctxlib
import sandbox

PARALLEL = 4
TEST_TIMEOUT = 90


def substitute(repo: common.Repo, f: ctxlib.Func, reply: str) -> tuple:
    """(new file text or None, reason). Line numbers refer to the unmodified file in the repository."""
    full = (common.repo_dir(repo) / f.path).read_text()
    body = answers.extract_body(reply, f.name, f.col)
    if body is None:
        return None, "no body in the reply"
    lines = full.splitlines(keepends=True)
    new = "".join(lines[: f.body_start - 1]) + body + "".join(lines[f.body_end :])
    try:
        ast.parse(new)
    except SyntaxError as e:
        return None, f"syntax error: {e.msg}"
    return new, ""


def score_one(repo: common.Repo, task: dict, reply: str) -> dict:
    f = ctxlib.Func(**task["func"])
    new, why = substitute(repo, f, reply)
    if new is None:
        return {"correct": False, "why": why}
    res = sandbox.run_tests(repo, {f.path: new}, timeout=TEST_TIMEOUT)
    ok = res["rc"] == 0
    return {"correct": ok, "why": "" if ok else ("timeout" if res["timed_out"] else res["tail"][-300:])}


def tasks_of(repo: str) -> dict:
    path = common.WORK / "tasks" / f"{repo}-B.jsonl"
    return {t["id"]: t for t in map(json.loads, path.read_text().splitlines())}


def rescore_a(rows: list) -> None:
    """Family A is scored again from the stored replies, so a fix of the normalisation applies to every run alike."""
    cache = {}
    for r in rows:
        if r["family"] == "A" and "reply" in r:
            t = cache.setdefault(r["repo"], {x["id"]: x for x in map(json.loads, _lines(r["repo"], "A"))})[r["id"]]
            r["answer"] = answers.first_line(r["reply"])
            r["correct"] = answers.is_correct(t["kind"], t["answer"], r["reply"])


def _lines(repo: str, fam: str) -> list:
    return (common.WORK / "tasks" / f"{repo}-{fam}.jsonl").read_text().splitlines()


def score_file(label: str) -> None:
    path = common.WORK / "runs" / f"{label}.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    todo = [r for r in rows if r["family"] == "B" and "reply" in r]  # always again: the scorer may have been fixed
    rescore_a(rows)
    cache = {}

    def work(r):
        t = cache.setdefault(r["repo"], tasks_of(r["repo"]))[r["id"]]
        r.update(score_one(common.REPOS[r["repo"]], t, r["reply"]))

    with ThreadPoolExecutor(PARALLEL) as ex:
        list(ex.map(work, todo))
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    print(f"{label}: scored {len(todo)} family B tasks, {sum(r['correct'] for r in todo)} correct")


def selftest(repo_name: str) -> None:
    repo = common.REPOS[repo_name]
    bad = []

    def work(t):
        f = ctxlib.Func(**t["func"])
        full = (common.repo_dir(repo) / f.path).read_text().splitlines(keepends=True)
        # indentation of a method relative to its class is kept; the extraction takes the def by name
        text = "".join(full[f.start - 1 : f.body_end])
        # the last function of a file may lack the final newline, which would glue the closing fence to the code
        reply = "```python\n" + text + ("" if text.endswith("\n") else "\n") + "```"
        r = score_one(repo, t, reply)
        if not r["correct"]:
            bad.append((t["id"], r["why"]))

    with ThreadPoolExecutor(PARALLEL) as ex:
        list(ex.map(work, tasks_of(repo_name).values()))
    print(f"{repo_name}: selftest failures {len(bad)}")
    for b in bad:
        print(b)


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest(sys.argv[2])
    else:
        score_file(sys.argv[1])
