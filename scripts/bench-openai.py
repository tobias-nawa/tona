#!/usr/bin/env python3
"""Decode measurement against an OpenAI-compatible server (e.g. llama.cpp) with the same three prompts as
scripts/bench-decode.sh: /v1/chat/completions, thinking off (chat_template_kwargs), 200 tokens, greedy.
Decode rate from the server's timings (predicted_per_second, llama.cpp format).
Plus "long": ~13K tokens of code (scripts/bench-long-code.txt) with a random line in front so no prefix cache
applies; gives the prefill rate (prompt_n / prompt_ms) and decode after a long prompt.
Usage: scripts/bench-openai.py URL MODEL [runs, default 3] [cases, e.g. long or csv,long]"""

import json
import os
import random
import statistics
import sys
import time
import urllib.request

URL, MODEL = sys.argv[1].rstrip("/"), sys.argv[2]
RUNS = int(sys.argv[3]) if len(sys.argv) > 3 else 3
CASES = sys.argv[4].split(",") if len(sys.argv) > 4 else ["csv", "rust", "explain", "long"]


PROMPTS = {
    "csv": "Write a Python function that parses a CSV file and returns the average of each numeric column. Only code.",
    "rust": "Write a Rust function that checks whether a string is a valid IPv4 address, with tests. Only code.",
    "explain": "Explain how a hash map handles collisions, in a few paragraphs.",
}
# fixed copy (start of engine/gpu/prefill.hip at 7c48965) so the prompt does not change with the code
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench-long-code.txt")) as f:
    CODE = f.read()
LONG = "Summarize what the following HIP code does, as a bullet list.\n\n```cpp\n" + CODE + "\n```"


def run(prompt):
    body = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 200,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        URL + "/v1/chat/completions", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        t = json.load(r)["timings"]
    return (
        t["predicted_per_second"],
        t["predicted_n"],
        t["prompt_ms"] / 1000,
        t.get("draft_n_accepted", 0) / max(t.get("draft_n", 0), 1),
        t.get("prompt_n", 0),
    )


print(f"== {time.strftime('%Y-%m-%d %H:%M')} {URL} {MODEL}, {RUNS} runs")
for name, p in PROMPTS.items():
    if name not in CASES:
        continue
    rates = []
    for _ in range(RUNS):
        tps, n, pre, acc, _ = run(p)
        rates.append(tps)
        print(f"   {name}: {tps:.1f} tok/s ({n} tokens, prompt {pre:.2f} s, drafts {acc:.0%} accepted)", flush=True)
    rates.sort()
    print(
        f"{name:5} tok/s: median {statistics.median(rates):.1f}, "
        f"range {rates[0]:.1f}-{rates[-1]:.1f} (n = {len(rates)})"
    )

pre_rates, dec = [], []
for _ in range(RUNS if "long" in CASES else 0):
    tps, n, pre, acc, pn = run(f"Run {random.randrange(10**9)}.\n" + LONG)
    pre_rates.append(pn / pre)
    dec.append(tps)
    print(
        f"   long: prefill {pn} tokens in {pre:.2f} s = {pn / pre:.0f} tok/s, decode {tps:.1f} tok/s ({n} tokens)",
        flush=True,
    )
pre_rates.sort(), dec.sort()
if dec:
    print(
        f"long  prefill tok/s: median {statistics.median(pre_rates):.0f}, "
        f"range {pre_rates[0]:.0f}-{pre_rates[-1]:.0f}; "
        f"decode after: median {statistics.median(dec):.1f}, range {dec[0]:.1f}-{dec[-1]:.1f} (n = {len(dec)})"
    )
