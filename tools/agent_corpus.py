#!/usr/bin/env python3
"""Builds a corpus for speculation simulations from Claude Code transcripts (~/.claude/projects/**/*.jsonl).

Each session becomes a sequence of segments: context (user, tool results) and output (text and tool calls of the
assistant), roughly in the Qwen chat format. Thinking blocks are empty in the transcripts and missing.
The output contains private content: write it to work/ only, never commit it.

Format: a line "C <time> <name>" per session, then per segment "S <ctx|out> <bytes>\\n" and the raw bytes.
Usage: tools/agent_corpus.py [--root ~/.claude/projects] [--min-kb 200] --out work/spec/corpus.txt
"""

import argparse
import json
import sys
from pathlib import Path


def blocks(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return content or []


def result_text(c):
    if isinstance(c, str):
        return c
    return "\n".join(b.get("text", "") for b in c or [] if b.get("type") == "text")


def _segments(t: str, m: dict) -> list[tuple[str, str]]:
    """Context and output segments of one transcript message."""
    segs = []
    for b in blocks(m.get("content")):
        bt = b.get("type")
        if t == "user" and bt == "text":
            segs.append(("ctx", f"<|im_start|>user\n{b['text']}<|im_end|>\n<|im_start|>assistant\n"))
        elif t == "user" and bt == "tool_result":
            body = result_text(b.get("content"))
            segs.append(
                (
                    "ctx",
                    f"<|im_end|>\n<|im_start|>user\n<tool_response>\n{body}\n"
                    "</tool_response><|im_end|>\n<|im_start|>assistant\n",
                )
            )
        elif t == "assistant" and bt == "text" and b.get("text"):
            segs.append(("out", b["text"]))
        elif t == "assistant" and bt == "tool_use":
            call = json.dumps({"name": b.get("name"), "arguments": b.get("input")}, ensure_ascii=False)
            segs.append(("out", f"\n<tool_call>\n{call}\n</tool_call>"))
    return segs


def session(path: Path):
    segs, first = [], None
    with path.open(encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("type")
            if t not in ("user", "assistant"):
                continue
            first = first or d.get("timestamp")
            segs += _segments(t, d.get("message") or {})
    return first or "", segs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(Path.home() / ".claude/projects"))
    ap.add_argument("--min-kb", type=int, default=200)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    files = [p for p in Path(a.root).rglob("*.jsonl") if p.stat().st_size >= a.min_kb * 1024]
    sessions = []
    for p in files:
        ts, segs = session(p)
        if any(r == "out" for r, _ in segs):
            sessions.append((ts, p.parent.name + "/" + p.stem, segs))
    sessions.sort()  # chronological: the global tree only knows earlier sessions
    n_out = n_ctx = 0
    with open(a.out, "wb") as f:
        for ts, name, segs in sessions:
            f.write(f"C {ts or '-'} {name}\n".encode())
            for r, text in segs:
                b = text.encode("utf-8", "replace")
                f.write(f"S {r} {len(b)}\n".encode() + b)
                n_out += len(b) if r == "out" else 0
                n_ctx += len(b) if r == "ctx" else 0
    print(f"{len(sessions)} sessions, context {n_ctx / 1e6:.1f} MB, output {n_out / 1e6:.1f} MB -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
