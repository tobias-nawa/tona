#!/usr/bin/python3 -I
"""Effort versus quality: programming tasks with tests via /v1/messages, per level or thinking budget.

Usage: bench-effort.py URL LEVELS [TASKS]
  LEVELS: comma-separated, each low | medium | high | xhigh | max (output_config.effort as in Claude Code),
          budget:N (thinking enabled, budget_tokens N, level high) or off (no thinking)
  TASKS: names, comma-separated (default: all)
Output per level: passed, thinking tokens (median, max), time; single results as JSON lines in
work/effort/<time>.jsonl. Greedy (no temperature) so the levels are comparable.

The tasks are our own, medium-hard, with edge cases; they measure the difference between the levels, not the
absolute strength of the model. The generated code runs in a subprocess with a time and memory limit.
"""

import importlib.util
import json
import os
import re
import resource
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path


def _load_tasks() -> dict:
    # Loaded by path: -I (isolated mode) keeps the script directory off sys.path.
    spec = importlib.util.spec_from_file_location("effort_tasks", Path(__file__).with_name("effort_tasks.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TASKS


TASKS = _load_tasks()


def ask(url, prompt, level, max_tokens):
    body = {"model": "local", "max_tokens": max_tokens, "messages": [{"role": "user", "content": prompt}]}
    # SAMPLING="T[,top_p,top_k[,seed]]" (0: greedy); unset: send nothing, like Claude Code (server default)
    if os.environ.get("SAMPLING"):
        v = os.environ["SAMPLING"].split(",")
        body["temperature"] = float(v[0])
        if len(v) > 2:
            body.update(top_p=float(v[1]), top_k=int(v[2]))
        if len(v) > 3:
            body["seed"] = int(v[3])
    if level == "off":
        body["thinking"] = {"type": "disabled"}
    elif level.startswith("budget:"):
        body["thinking"] = {"type": "enabled", "budget_tokens": int(level.split(":")[1])}
        body["output_config"] = {"effort": "high"}
    else:
        body["thinking"] = {"type": "adaptive"}
        body["output_config"] = {"effort": level}
    req = urllib.request.Request(url + "/v1/messages", json.dumps(body).encode(), {"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    dt = time.time() - t0
    think = "".join(b.get("thinking", "") for b in out["content"] if b["type"] == "thinking")
    text = "".join(b.get("text", "") for b in out["content"] if b["type"] == "text")
    return think, text, out.get("stop_reason"), out["usage"]["output_tokens"], dt


def count_tokens(url, s):
    body = {"model": "local", "messages": [{"role": "user", "content": s or "."}]}
    req = urllib.request.Request(
        url + "/v1/messages/count_tokens", json.dumps(body).encode(), {"content-type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)["input_tokens"]


def limits():
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_CPU, (30, 30))


def check(code, tests):
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "t.py")
        with open(p, "w") as f:
            f.write(code + "\n\n# --- tests ---\n" + tests)
        try:
            r = subprocess.run(
                [sys.executable, "-I", p],
                cwd=d,
                capture_output=True,
                text=True,
                timeout=60,
                preexec_fn=limits,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return False, "time limit"
        return r.returncode == 0, (r.stderr or r.stdout).strip().splitlines()[-1:] if r.returncode else ""


def run_task(url: str, name: str, level: str, max_tokens: int, base: int) -> dict:
    sig, desc, tests = TASKS[name]
    prompt = (
        f"Write Python 3 code: {sig}.\n\n{desc}\n\nReply with the complete implementation in a single "
        "```python code block (standard library only, no tests, no example usage)."
    )
    think, text, stop, out_tok, dt = ask(url, prompt, level, max_tokens)
    blocks = re.findall(r"```(?:python|py)?\n(.*?)```", text, re.S)
    ok, err = check(blocks[-1], tests) if blocks else (False, "no code block")
    return {
        "level": level,
        "task": name,
        "ok": ok,
        "think_tokens": count_tokens(url, think) - base if think else 0,
        "out_tokens": out_tok,
        "s": round(dt, 1),
        "stop": stop,
        "err": err,
        "sampling": os.environ.get("SAMPLING", "default"),
        "think_tail": think[-600:],
    }


def print_summary(res: dict) -> None:
    print()
    for lv, rows in res.items():
        tt = [r["think_tokens"] for r in rows]
        print(
            f"{lv:>12}: {sum(r['ok'] for r in rows)}/{len(rows)} passed, "
            f"thinking tokens median {statistics.median(tt):.0f}, "
            f"max {max(tt)}, sum {sum(tt)}, time {sum(r['s'] for r in rows):.0f} s"
        )


def main():
    url, levels = sys.argv[1], sys.argv[2].split(",")
    names = sys.argv[3].split(",") if len(sys.argv) > 3 else list(TASKS)
    max_tokens = int(os.environ.get("MAX_TOKENS", "24000"))
    os.makedirs("work/effort", exist_ok=True)
    res = {}
    base = count_tokens(url, "")  # scaffold of the chat template around the text
    with open(time.strftime("work/effort/%Y-%m-%d-%H%M.jsonl"), "a") as log:
        for lv in levels:
            for n in names:
                row = run_task(url, n, lv, max_tokens, base)
                log.write(json.dumps(row, ensure_ascii=False) + "\n")
                log.flush()
                res.setdefault(lv, []).append(row)
                print(
                    f"{lv:>12} {n:<10} {'ok ' if row['ok'] else 'FAIL'} {row['think_tokens']:>6} thinking tokens "
                    f"{row['s']:7.1f} s  {row['err'] or ''}",
                    flush=True,
                )
    print_summary(res)


if __name__ == "__main__":
    main()
