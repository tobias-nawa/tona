#!/usr/bin/python3 -I
"""Ask the tasks of one or more repos to a running server (greedy) and write one JSONL line per task.

Usage: run.py URL LABEL REPOS [--limit N] [--families A,B] [--closed-book] [--shuffle SEED]
REPOS: comma-separated. Output: WORK/runs/LABEL.jsonl (resumes: ids already in the file are skipped). Family A lines
carry `correct`; family B lines are scored afterwards by score.py (they need the sandbox).
The context goes into the system message and the question into the last user message, so that the server's
checkpoint at the start of the last message (prompt cache) serves every task of a repo after the first.
"""

import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import answers
import common

INSTRUCTION = (
    "You are given files of the Python package {name} ({license} licence). Each file starts with a line "
    '"### FILE: <path>". In some functions the body was replaced by "...". Answer the question after the files.'
)
ASK = {
    "A": "Answer with the exact value or name only, on one line, without explanation.",
    "B": "Reply with the complete definition of the function (signature, docstring and body) in one Python code "
    "block and nothing else.",
}
# B: 700 cut 22 of 60 replies of the pilot (research/measurements/2026-10-30-threshold-pilot.md); the longest hole is
# 35 lines, so 2,000 only stops loops.
MAX_TOKENS = {"A": 64, "B": 2000}


def messages(ctx: dict, text: str, task: dict, closed_book: bool) -> list:
    head = INSTRUCTION.format(name=ctx["repo"], license=ctx["license"])
    system = head if closed_book else f"{head}\n\n{text}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"{task['question']}\n{ASK[task['family']]}"},
    ]


def post(url: str, body: dict, timeout: int = 3600) -> dict:
    req = urllib.request.Request(url, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def ask(url: str, msgs: list, family: str) -> dict:
    body = {
        "model": "x",
        "messages": msgs,
        "max_tokens": MAX_TOKENS[family],
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    try:
        r = post(url + "/v1/chat/completions", body)
    except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
        return {"error": str(e), "wall_s": time.time() - t0}
    return {
        "reply": r["choices"][0]["message"]["content"] or "",
        "finish": r["choices"][0].get("finish_reason"),
        "timings": r["timings"],
        "prompt_tokens": r["usage"]["prompt_tokens"],
        "wall_s": time.time() - t0,
    }


def load_tasks(repo: str, families: str) -> list:
    out = []
    for fam in families.split(","):
        path = common.WORK / "tasks" / f"{repo}-{fam}.jsonl"
        out += [json.loads(line) for line in path.read_text().splitlines()]
    return out


def done_ids(path: Path) -> set:
    if not path.exists():
        return set()
    return {json.loads(line)["id"] for line in path.read_text().splitlines() if '"error"' not in line}


def run_repo(url, repo, out, opts) -> None:
    ctx = common.read_json(common.WORK / "ctx" / f"{repo}-ctx.json")
    text = (common.WORK / "ctx" / f"{repo}.txt").read_text()
    skip = done_ids(out)
    tasks = [t for t in load_tasks(repo, opts["families"]) if t["id"] not in skip]
    if opts["shuffle"] is not None:  # a different order is a different cache history: a repeat that is not a copy
        random.Random(opts["shuffle"]).shuffle(tasks)
    for t in tasks[: opts["limit"]]:
        res = ask(url, messages(ctx, text, t, opts["closed_book"]), t["family"])
        line = {k: t[k] for k in ("id", "family", "repo", "kind", "src_pos", "src_path")}
        line["local"] = t.get("local", False)
        line.update(res)
        if t["family"] == "A" and "reply" in res:
            line["answer"] = answers.first_line(res["reply"])
            line["correct"] = answers.is_correct(t["kind"], t["answer"], res["reply"])
        with out.open("a") as f:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")
        tm = res.get("timings", {})
        print(
            f"{t['id']} correct={line.get('correct')} cache_n={tm.get('cache_n')} prompt_n={tm.get('prompt_n')} "
            f"prompt_ms={tm.get('prompt_ms', 0):.0f} predicted_n={tm.get('predicted_n')} wall={res['wall_s']:.1f}s",
            flush=True,
        )


def main():
    args = sys.argv[1:]
    url, label, repos = args[0].rstrip("/"), args[1], args[2].split(",")
    opts = {"limit": 10**9, "families": "A,B", "closed_book": "--closed-book" in args, "shuffle": None}
    if "--shuffle" in args:
        opts["shuffle"] = int(args[args.index("--shuffle") + 1])
    if "--limit" in args:
        opts["limit"] = int(args[args.index("--limit") + 1])
    if "--families" in args:
        opts["families"] = args[args.index("--families") + 1]
    out = common.WORK / "runs" / f"{label}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    for repo in repos:
        run_repo(url, repo, out, opts)


if __name__ == "__main__":
    main()
