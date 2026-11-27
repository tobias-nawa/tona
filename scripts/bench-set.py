#!/usr/bin/python3 -I
"""Decode over many prompts (11 code, 3 text), greedy, up to 400 tokens each. For changes that shift the rounding
and with it the text: the drafter then hits more or less by chance per prompt, which is noise over 3 prompts.
Sum of tokens / sum of time per group. The same prompts as the offline calibration of the drafter priors (apart
from bench-openai.py).

bench-set.py URL [model]   (model matters for llama-server only)
API key (tona-server with --api-key-file): the file named in TONA_API_KEY_FILE.
"""

import json
import os
import sys
import urllib.request
from pathlib import Path

URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
MODEL = sys.argv[2] if len(sys.argv) > 2 else "x"
CODE = [
    "Write a Python function that merges overlapping intervals. Only code.",
    "Write a TypeScript React component for a searchable dropdown list. Only code.",
    "Write a Go HTTP handler that returns JSON with the server time, with a test.",
    "Write a C++ class for a thread-safe bounded queue using condition variables.",
    "Write a Bash script that finds the 10 largest files under a directory.",
    "Write SQL to find the top 3 customers by revenue per month from an orders table.",
    "Refactor this Python code to use a dataclass:\nclass P:\n    def __init__(self, x, y):\n        self.x = x\n"
    "        self.y = y\n    def __repr__(self):\n        return f'P({self.x}, {self.y})'",
    "Write a Rust function that parses a semantic version string like 1.2.3-beta, with unit tests. Only code.",
    "Write pytest tests for a function slugify(s) that lowercases and replaces spaces with dashes.",
    "Fix the bug: def fib(n): return fib(n-1) + fib(n-2)",
    "Write a Python asyncio script that downloads 5 URLs concurrently with aiohttp.",
]
TEXT = [
    "Explain the difference between processes and threads in a few paragraphs.",
    # German on purpose: the set covers a non-English prompt
    "Erkläre kurz, wie Git rebase funktioniert und wann man es benutzt.",
    "Describe the CAP theorem and give one example system for each trade-off.",
]

KEY_FILE = os.environ.get("TONA_API_KEY_FILE", "")
KEY = Path(KEY_FILE).read_text().strip() if KEY_FILE and KEY_FILE != "none" and os.path.exists(KEY_FILE) else ""


def run(p):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 400,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    headers = {"content-type": "application/json"}
    if KEY:
        headers["authorization"] = "Bearer " + KEY
    req = urllib.request.Request(URL + "/v1/chat/completions", json.dumps(body).encode(), headers)
    with urllib.request.urlopen(req, timeout=900) as r:
        t = json.load(r)["timings"]
    return t["predicted_n"], t["predicted_ms"] / 1000, t.get("draft_n", 0), t.get("draft_n_accepted", 0)


for name, ps in (("Code", CODE), ("Text", TEXT)):
    n = s = dn = da = 0
    per = []
    for p in ps:
        a, b, c, d = run(p)
        n, s, dn, da = n + a, s + b, dn + c, da + d
        per.append(a / b)
    # verify passes ~ tokens - accepted drafts (one target token per pass)
    acc = f", {n / max(n - da, 1):.2f} tokens per pass, {da}/{dn} drafts accepted" if dn else ""
    print(f"{name}: {n} Tokens in {s:.1f} s = {n / s:.1f} tok/s decode (per prompt {min(per):.0f}-{max(per):.0f}){acc}")
