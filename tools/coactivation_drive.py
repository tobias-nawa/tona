#!/usr/bin/env python3
"""Workload driver for the co-activation traces: sends the prompts of four groups to a running server, one
request at a time, greedy, and writes one JSON line per request (group, prompt length, tokens, draft counters) in the
order the requests ran, so that the route log of the server can be cut into requests and groups.

Groups: codeA (bench-set code, held out from the profile), codeB (new code prompts), text (new prose prompts, English
and German), long (two requests with about 20k tokens of context: source code and prose). Code and text have 400
tokens each, the long requests 300.

Usage: tools/coactivation_drive.py URL OUT.jsonl [--root REPO]
"""

import argparse
import json
import sys
import urllib.request
from pathlib import Path

CODE_B = [
    "Write a Python class implementing an LRU cache with O(1) get and put. Only code.",
    "Write a JavaScript function that debounces another function, with a usage example. Only code.",
    "Write a Java method that detects a cycle in a singly linked list, with a JUnit test.",
    "Write a C function that reverses a UTF-8 string in place. Only code.",
    "Write a Kotlin data class for an order with a function that computes the total including tax. Only code.",
    "Write a Rust program that reads lines from stdin and prints word frequencies sorted by count.",
    "Write a Python script with argparse that converts a CSV file to JSON lines. Only code.",
    "Write a SQL schema for a blog with users, posts and comments, plus an index for the latest posts per user.",
    "Write a Dockerfile and a docker-compose.yml for a Flask app with a Postgres database.",
    "Write a Go function that runs N workers over a channel of jobs and collects the results in order.",
]
TEXT = [
    "Explain how a hash table handles collisions, in plain language, in a few paragraphs.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Fasse die Geschichte der Eisenbahn in Deutschland in einigen Absaetzen zusammen.",
    "What are the main causes of the French Revolution? Answer in a structured essay.",
    "Translate into German and explain the grammar: 'The committee has postponed the decision until next week.'",
    "Give advice on how to prepare for a job interview as a software engineer, as a numbered list with reasons.",
    "Erklaere den Unterschied zwischen Inflation und Deflation und nenne je ein Beispiel.",
    "Solve step by step: a train leaves at 9:00 at 80 km/h, a second one at 10:00 at 120 km/h. When does it catch up?",
    "Compare TCP and UDP and say when you would choose which.",
    "Describe how photosynthesis works, then how it differs in C4 plants.",
]
LONG_CODE = "Here is source code. Find three possible bugs or risks in it and explain each in a few sentences.\n\n"
LONG_TEXT = "Here are research notes. Summarize the main decisions and open questions in about 10 bullet points.\n\n"
LONG_CHARS = 60000  # about 18 to 20k tokens


def load_bench_set_code(root: Path) -> list[str]:
    """The CODE list of scripts/bench-set.py, read without running it."""
    src = (root / "scripts" / "bench-set.py").read_text()
    ns: dict = {}
    start = src.index("CODE = [")
    end = src.index("TEXT = [")
    exec(src[start:end], ns)  # noqa: S102 (a literal list of our own file)
    return ns["CODE"]


def long_prompts(root: Path) -> list[str]:
    code = "".join(p.read_text() for p in sorted((root / "engine" / "gpu").glob("mega_*.cpp")))[:LONG_CHARS]
    notes = "".join(p.read_text() for p in sorted((root / "research").glob("[0-4]*.md")))[:LONG_CHARS]
    return [LONG_CODE + code, LONG_TEXT + notes]


def request(url: str, prompt: str, max_tokens: int) -> dict:
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
    return {**out["timings"], "prompt_tokens": out.get("usage", {}).get("prompt_tokens")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parent.parent))
    args = ap.parse_args()
    root = Path(args.root)
    jobs = [("codeA", p, 400) for p in load_bench_set_code(root)]
    jobs += [("codeB", p, 400) for p in CODE_B]
    jobs += [("text", p, 400) for p in TEXT]
    jobs += [("long", p, 300) for p in long_prompts(root)]
    with open(args.out, "w") as f:
        for group, prompt, n in jobs:
            t = request(args.url, prompt, n)
            rec = {"group": group, "chars": len(prompt), "max_tokens": n, **t}
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(group, t.get("prompt_tokens"), t.get("predicted_n"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
