#!/usr/bin/env python3 -I
"""Asks the fixed chat prompts (bench/chat/prompts.txt) once each and stores the answers.

    run.py URL OUT.jsonl [SAMPLING_JSON]

Blind scoring: `run.py blind OUT1.jsonl OUT2.jsonl ... > blind.md` prints every prompt with the answers of all
models in a fixed pseudo-random order under neutral labels A, B, ...; the key (label to model) goes to stderr and is
kept apart from the file the scorer reads. Standard library only.
"""

import json
import random
import sys
import time
import urllib.request
from pathlib import Path

PROMPTS = Path(__file__).with_name("prompts.txt")


def prompts() -> list[tuple[str, str]]:
    rows = [ln.split("\t", 1) for ln in PROMPTS.read_text().splitlines() if ln and not ln.startswith("#")]
    return [(a, b) for a, b in rows]


def ask(url: str, text: str, sampling: dict) -> dict:
    body = {"model": "default", "messages": [{"role": "user", "content": text}], "max_tokens": 8192, **sampling}
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=1800) as r:
        d = json.load(r)
    return {
        "answer": d["choices"][0]["message"].get("content") or "",
        "finish": d["choices"][0].get("finish_reason"),
        "tokens": d.get("usage", {}).get("completion_tokens"),
        "seconds": time.perf_counter() - t0,
    }


def collect(url: str, out: Path, sampling: dict) -> None:
    done = {json.loads(ln)["id"] for ln in out.read_text().splitlines()} if out.exists() else set()
    for pid, text in prompts():
        if pid in done:
            continue
        with out.open("a") as f:
            f.write(json.dumps({"id": pid, "prompt": text, **ask(url, text, sampling)}) + "\n")
        print(pid, flush=True)


def blind(files: list[str]) -> None:
    runs = {Path(f).stem: {r["id"]: r for r in map(json.loads, Path(f).read_text().splitlines())} for f in files}
    names = list(runs)
    rng = random.Random(139)
    for pid, text in prompts():
        order = names[:]
        rng.shuffle(order)
        print(f"## {pid}\n\n{text}\n")
        for label, name in zip("ABCDEFGH", order, strict=False):
            print(f"### {pid} {label}\n\n{runs[name][pid]['answer'].strip()}\n")
            print(f"{pid} {label} = {name}", file=sys.stderr)


def main() -> int:
    if sys.argv[1] == "blind":
        blind(sys.argv[2:])
        return 0
    sampling = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
    collect(sys.argv[1], Path(sys.argv[2]), sampling)
    return 0


if __name__ == "__main__":
    sys.exit(main())
