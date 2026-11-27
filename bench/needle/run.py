#!/usr/bin/env python3 -I
"""Greedy needle test in the chat format against a running server.

One conversation per text: the first user message holds the document and the first question, every further question
is a new turn (the server continues from the state it holds, so the document is computed once). Per text: the
"single" needles one by one, then the "multi" needles in one answer. One JSONL line per needle; a finished text is
skipped on a second call, so a run can be resumed.

    run.py --url http://127.0.0.1:8096 --pool corp/code.txt --length 131072 --texts 0,1,2,3 --label off --out out/
"""

import argparse
import json
import random
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haystack


def post(url: str, path: str, body: dict) -> dict:
    req = urllib.request.Request(url + path, json.dumps(body).encode(), {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=7200) as r:
        return json.load(r)


def chat(url: str, messages: list[dict], max_tokens: int) -> tuple[str, dict, float]:
    body = {
        "model": "x",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    t0 = time.time()
    resp = post(url, "/v1/chat/completions", body)
    return resp["choices"][0]["message"]["content"], resp["timings"], time.time() - t0


def count_tokens(url: str, text: str) -> int:
    body = {"model": "x", "max_tokens": 1, "messages": [{"role": "user", "content": text}]}
    return post(url, "/v1/messages/count_tokens", body)["input_tokens"]


def first_message(doc: str, question: str) -> str:
    return f"Here is a long source file.\n\n```\n{doc}\n```\n\n{question}"


def fit_document(url: str, units: list[str], seed: int, target: int, needles: list) -> str:
    """Document whose first message (with the longest question) is within 0.3 % of `target` tokens."""
    probe = "".join(units[:200])[:400_000]
    ratio = len(probe) / count_tokens(url, probe)  # characters per token
    chars = int(target * ratio)
    question = haystack.multi_question(needles, seed)
    for _ in range(6):
        doc = haystack.build_document(units, seed, chars, needles)
        n = count_tokens(url, first_message(doc, question))
        if abs(n - target) <= target * 0.003:
            return doc
        chars = int(chars * target / n)
    return doc


def ask(url: str, history: list[dict], question: str, doc: str | None, max_tokens: int) -> tuple[str, dict]:
    content = first_message(doc, question) if doc is not None else question
    messages = [*history, {"role": "user", "content": content}]
    answer, timings, wall = chat(url, messages, max_tokens)
    history += [messages[-1], {"role": "assistant", "content": answer}]
    timings["wall_s"] = wall
    return answer, timings


def record(meta: dict, n: haystack.Needle, hit: bool, answer: str, t: dict) -> dict:
    keep = {k: t.get(k) for k in ("prompt_n", "cache_n", "prompt_ms", "predicted_n", "predicted_per_second", "wall_s")}
    fields = {"needle": n.name, "mode": n.mode, "bin": n.bin, "frac": round(n.frac, 4), "code": n.code}
    return {**meta, **fields, "hit": hit, "answer": answer.strip()[:400], **keep}


def run_text(args: argparse.Namespace, units: list[str], idx: int, out) -> None:
    needles = haystack.make_needles(idx, args.hard)
    doc = fit_document(args.url, units, idx, args.length, needles)
    meta = {"label": args.label, "length": args.length, "text": idx, "hard": args.hard}
    singles = [n for n in needles if n.mode == "single"]
    random.Random(f"order-{idx}").shuffle(singles)
    history: list[dict] = []
    first_prompt = None
    for n in singles:
        answer, t = ask(args.url, history, haystack.single_question(n), doc if not history else None, 24)
        first_prompt = first_prompt or (t["prompt_n"] + t["cache_n"])
        hit = n.code in re.findall(r"\d+", answer)
        out.write(json.dumps(record(meta, n, hit, answer, t)) + "\n")
        out.flush()
        check_reuse(args, t, first_prompt, len(history))
    multi = [n for n in needles if n.mode == "multi"]
    answer, t = ask(args.url, history, haystack.multi_question(multi, idx), None, 700)
    found = haystack.parse_multi(answer)
    for n in multi:
        out.write(json.dumps(record(meta, n, found.get(n.name) == n.code, answer, t)) + "\n")
    out.write(json.dumps({**meta, "done": True, "prompt_tokens": first_prompt}) + "\n")
    out.flush()


def check_reuse(args: argparse.Namespace, t: dict, first_prompt: int, history_len: int) -> None:
    """From the second turn on the document must come from the cache; otherwise every question costs a prefill."""
    if history_len < 4 or t["prompt_n"] <= 0.1 * first_prompt:
        return
    msg = f"turn {history_len // 2 + 1} computed {t['prompt_n']} of {first_prompt} tokens: no prefix reuse"
    if args.require_cache:
        raise SystemExit("ERROR: " + msg)
    print("WARNING: " + msg, file=sys.stderr, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://127.0.0.1:8096")
    ap.add_argument("--pool", required=True, help="text with '// ===== path =====' file headers (corpus.py)")
    ap.add_argument("--length", type=int, required=True, help="tokens of the first message")
    ap.add_argument("--texts", default="0,1,2", help="seeds of the texts, comma separated")
    ap.add_argument("--label", required=True, help="name of the variant (file name and column in the report)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--hard", action="store_true", help="decoys: old revisions and a lookalike vault per needle")
    ap.add_argument("--require-cache", action="store_true", help="abort when a later turn recomputes the document")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    units = haystack.split_pool(Path(args.pool).read_text())
    for idx in [int(x) for x in args.texts.split(",")]:
        path = args.out / f"{args.label}{'-hard' if args.hard else ''}-{args.length}-t{idx}.jsonl"
        if path.exists() and '"done": true' in path.read_text():
            print(f"{path.name}: done, skipped", flush=True)
            continue
        t0 = time.time()
        with open(path, "w") as out:
            run_text(args, units, idx, out)
        print(f"{path.name}: {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    main()
