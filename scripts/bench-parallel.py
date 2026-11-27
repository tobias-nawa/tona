#!/usr/bin/python3 -I
"""Several conversations at the same time: P clients send the code prompts of bench-set.py, each in a different
order, greedy, up to 400 tokens each. Measures decode overall (sum of tokens / wall clock of the decode phases) and
per client (sum of its tokens / sum of its decode times, server-reported).

bench-parallel.py URL [P] [rounds]   (default 2 clients, 1 round)
Environment: SAMPLING=T[,top_p,top_k] instead of greedy (as in bench-effort.py)
"""

import ast
import json
import os
import sys
import threading
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8091"
P = int(sys.argv[2]) if len(sys.argv) > 2 else 2
ROUNDS = int(sys.argv[3]) if len(sys.argv) > 3 else 1

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench-set.py")) as f:
    src = f.read()
CODE = ast.literal_eval(src[src.index("CODE = [") + len("CODE = ") : src.index("TEXT = [")])


def ask(prompt):
    body = {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 400,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    smp = os.environ.get("SAMPLING")
    if smp:
        v = smp.split(",")
        body["temperature"] = float(v[0])
        if len(v) > 1:
            body["top_p"] = float(v[1])
        if len(v) > 2:
            body["top_k"] = int(v[2])
    else:
        body["temperature"] = 0
    req = urllib.request.Request(
        URL + "/v1/chat/completions",
        json.dumps(body).encode(),
        {"content-type": "application/json", "x-tona-project": "bench"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        j = json.load(r)
    t = j["timings"]
    return t["predicted_n"], t["predicted_n"] / max(t["predicted_per_second"], 1e-9), t0, time.time()


res = [[] for _ in range(P)]


def client(c):
    order = CODE[c * len(CODE) // P :] + CODE[: c * len(CODE) // P]
    for _ in range(ROUNDS):
        for p in order:
            res[c].append(ask(p))


t0 = time.time()
th = [threading.Thread(target=client, args=(c,)) for c in range(P)]
for t in th:
    t.start()
for t in th:
    t.join()
wall = time.time() - t0
tot = sum(n for r in res for n, *_ in r)
print(
    f"{P} clients, {sum(len(r) for r in res)} requests: {tot} tokens in {wall:.1f} s wall clock = "
    f"{tot / wall:.1f} tok/s decode overall (including prefill pauses)"
)
for c, r in enumerate(res):
    n, s = sum(x[0] for x in r), sum(x[1] for x in r)
    print(f"  client {c}: {n} tokens, {n / s:.1f} tok/s decode per request (server time)")
