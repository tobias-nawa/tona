#!/usr/bin/env python3 -I
"""Polyglot coding exercises against an OpenAI-compatible server (Aider polyglot method, own harness).

    run.py reference --data DIR --valid FILE            run every reference solution in the sandbox, write the valid ids
    run.py run --data DIR --valid FILE --url URL --out FILE.jsonl [--limit N] [--sampling JSON] [--max-tokens N]

Method: per exercise the model gets the instructions and the stub file(s) and answers with whole files; the tests run in
the sandbox; on failure the model gets the test output once and answers again (pass@2 as in Aider; pass@1 is the first
attempt). Order: the valid ids shuffled with a fixed seed, so a run cut short is a random subset and runs of different
models see the same prefix. One JSON line per exercise, a rerun with the same --out resumes. Standard library only.
"""

import argparse
import json
import random
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox
import tasks

SEED = 139  # fixed so every model sees the same order


def ordered(data: Path, valid: Path | None, langs: tuple[str, ...] = tasks.LANGS) -> list[tasks.Exercise]:
    allx = [ex for lang in langs for ex in tasks.load(data, lang)]
    if valid is not None and valid.exists():
        keep = set(valid.read_text().split())
        allx = [ex for ex in allx if ex.id in keep]
    random.Random(SEED).shuffle(allx)
    return allx


def chat(url: str, messages: list, sampling: dict, max_tokens: int, model: str) -> dict:
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "stream": False, **sampling}
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=3600) as r:
        d = json.load(r)
    ch = d["choices"][0]
    msg = ch["message"]
    return {
        "text": msg.get("content") or "",
        "finish": ch.get("finish_reason"),
        "completion_tokens": d.get("usage", {}).get("completion_tokens"),
        "prompt_tokens": d.get("usage", {}).get("prompt_tokens"),
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "seconds": time.perf_counter() - t0,
        "timings": d.get("timings"),
    }


def attempt(ex: tasks.Exercise, args, messages: list) -> tuple[dict, bool, str]:
    r = chat(args.url, messages, json.loads(args.sampling), args.max_tokens, args.model)
    files = tasks.parse_reply(r["text"], ex.solution)
    r["files_parsed"] = sorted(files)
    if r["finish"] == "length":
        return r, False, "The answer was cut off (token limit)."
    if not files:
        return r, False, "No file in the requested FILE: block form was found in the answer."
    ok, out = sandbox.run_tests(ex, files)
    return r, ok, out


def solve(ex: tasks.Exercise, args) -> dict:
    messages = [{"role": "user", "content": tasks.first_prompt(ex)}]
    rec = {"id": ex.id, "attempts": [], "passed1": False, "passed2": False, "passed_at": None}
    for n in range(1, args.attempts + 1):
        r, ok, out = attempt(ex, args, messages)
        rec["attempts"].append({**r, "passed": ok, "test_tail": "" if ok else out[-1500:]})
        if ok:
            rec["passed1"] = rec["passed1"] or n == 1
            rec["passed2"] = True  # within the attempts allowed (2 by default: Aider's pass@2)
            rec["passed_at"] = n
            break
        messages += [
            {"role": "assistant", "content": r["text"]},
            {"role": "user", "content": tasks.RETRY.format(output=out)},
        ]
    return rec


def cmd_reference(args) -> int:
    good = []
    for ex in ordered(Path(args.data), None, tuple(args.langs.split(","))):
        ok, out = sandbox.reference(ex)
        print(ex.id, "ok" if ok else "FAIL", flush=True)
        if ok:
            good.append(ex.id)
        else:
            print(out[-400:], file=sys.stderr)
    Path(args.valid).write_text("\n".join(sorted(good)) + "\n")
    print(f"{len(good)} valid exercises")
    return 0


def cmd_run(args) -> int:
    out = Path(args.out)
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    pool = ordered(Path(args.data), Path(args.valid), tuple(args.langs.split(",")))[args.start :]
    todo = [ex for ex in pool if ex.id not in done][: args.limit]
    for ex in todo:
        rec = solve(ex, args)
        with out.open("a") as f:
            f.write(json.dumps(rec) + "\n")
        toks = sum(a["completion_tokens"] or 0 for a in rec["attempts"])
        print(f"{ex.id} pass1={rec['passed1']} pass2={rec['passed2']} tokens={toks}", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mode", choices=["reference", "run"])
    p.add_argument("--data", required=True)
    p.add_argument("--valid", required=True)
    p.add_argument("--langs", default=",".join(tasks.LANGS), help="tracks to draw from (csharp has its own data dir)")
    p.add_argument("--url", default="http://127.0.0.1:8099")
    p.add_argument("--model", default="default")
    p.add_argument("--out")
    p.add_argument("--limit", type=int, default=10**9)
    p.add_argument("--sampling", default="{}", help="JSON of sampling fields merged into the request body")
    p.add_argument("--start", type=int, default=0, help="skip the first N exercises of the fixed order")
    p.add_argument("--attempts", type=int, default=2, help="tries per exercise, each after the test output")
    p.add_argument("--max-tokens", type=int, default=16384)
    args = p.parse_args()
    return cmd_reference(args) if args.mode == "reference" else cmd_run(args)


if __name__ == "__main__":
    sys.exit(main())
