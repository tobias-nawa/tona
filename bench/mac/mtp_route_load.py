#!/usr/bin/python3 -I
"""Coding load for the routing statistics of the MTP head: tasks x languages, greedy, up to 400 tokens each, against a
tona-server started with --mtp and TONA_MTP_ROUTE_STATS. Runs until the server has decoded --tokens tokens.

mtp_route_load.py [--url URL] [--tokens 50000]
"""

import argparse
import itertools
import json
import time
import urllib.request

LANGS = ["Python", "TypeScript", "Go", "Rust", "C++", "Java", "C#", "Bash", "SQL", "Kotlin", "Ruby", "PHP"]
TASKS = [
    "a function that merges overlapping intervals, with tests",
    "an LRU cache with get and put in O(1)",
    "a command line tool that counts word frequencies in a file",
    "a small HTTP JSON API with one GET and one POST endpoint",
    "a parser for semantic version strings, with tests",
    "a thread-safe bounded queue",
    "a function that validates and normalizes email addresses",
    "a retry helper with exponential backoff and jitter",
    "a binary search tree with insert, delete and in-order traversal",
    "a CSV reader that groups rows by a column and sums another",
    "a rate limiter (token bucket)",
    "a refactoring of a long function into smaller ones, with an example before and after",
]


def run(url: str, prompt: str) -> int:
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        url + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.load(r)["timings"]["predicted_n"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--tokens", type=int, default=50000)
    a = ap.parse_args()
    done, t0 = 0, time.time()
    styles = ["Only code.", "With comments and error handling. Only code."]
    for i, (style, task, lang) in enumerate(itertools.product(styles, TASKS, LANGS)):
        if done >= a.tokens:
            break
        n = run(a.url, f"Write {task} in {lang}. {style}")
        done += n
        print(f"{i + 1}: {lang}, {n} tokens, {done} in total, {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
