#!/usr/bin/env python3
"""Smoke test for tona-server: an agent round (question -> tool call -> tool result -> answer) through the Anthropic
and the OpenAI API, each with and without streaming. Standard library only.
Usage: tools/server_smoke.py [http://127.0.0.1:8662]"""

import json
import sys
import time
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8662"
FAILS = 0


def post(path, body, stream=False):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(), {"content-type": "application/json"})
    r = urllib.request.urlopen(req, timeout=600)
    if not stream:
        return json.load(r)
    events = []
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            events.append(json.loads(line[6:]))
    return events


def check(what, ok, info=""):
    global FAILS
    print(("ok    " if ok else "FAIL  ") + f" {what}" + (f": {info}" if info and not ok else ""))
    FAILS += not ok


TOOL = {
    "name": "read_file",
    "description": "Read a text file and return its content",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
}
ASK = "What is the hostname stored in /etc/hostname? Use the read_file tool."


def anthropic_stream_to_message(events):
    blocks, cur = [], None
    stop = None
    for e in events:
        if e["type"] == "content_block_start":
            cur = dict(e["content_block"])
            if cur["type"] == "tool_use":
                cur["_json"] = ""
            blocks.append(cur)
        elif e["type"] == "content_block_delta":
            d = e["delta"]
            if d["type"] == "text_delta":
                cur["text"] += d["text"]
            elif d["type"] == "thinking_delta":
                cur["thinking"] += d["thinking"]
            elif d["type"] == "input_json_delta":
                cur["_json"] += d["partial_json"]
        elif e["type"] == "message_delta":
            stop = e["delta"]["stop_reason"]
    for b in blocks:
        if b["type"] == "tool_use":
            b["input"] = json.loads(b.pop("_json") or "{}")
    return {"content": blocks, "stop_reason": stop}


def anthropic(stream):
    msgs = [{"role": "user", "content": ASK}]
    t = time.time()
    body = {"model": "local", "max_tokens": 300, "tools": [TOOL], "messages": msgs, "stream": stream}
    r = post("/v1/messages", body, stream)
    if stream:
        r = anthropic_stream_to_message(r)
    use = [b for b in r["content"] if b["type"] == "tool_use"]
    check(
        f"Anthropic{' stream' if stream else ''}: tool call",
        r["stop_reason"] == "tool_use"
        and use
        and use[0]["name"] == "read_file"
        and use[0]["input"].get("path") == "/etc/hostname",
        json.dumps(r)[:300],
    )
    if not use:
        return
    msgs += [
        {"role": "assistant", "content": r["content"]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": use[0]["id"], "content": "build-box-17"}]},
    ]
    body["messages"] = msgs
    r = post("/v1/messages", body, stream)
    if stream:
        r = anthropic_stream_to_message(r)
    text = "".join(b.get("text", "") for b in r["content"] if b["type"] == "text")
    check(f"Anthropic{' stream' if stream else ''}: answer uses the result", "build-box-17" in text, text[:200])
    print(f"      {time.time() - t:.1f} s")


def _openai_tool_message(r, stream):
    """(finish_reason, assistant message, tool calls) from a streamed or plain response."""
    if stream:
        calls, fin = [], None
        for c in r:
            ch = c["choices"][0]
            calls += ch["delta"].get("tool_calls", [])
            fin = ch.get("finish_reason") or fin
        msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [{k: v for k, v in c.items() if k != "index"} for c in calls],
        }
        return fin, msg, calls
    msg = r["choices"][0]["message"]
    return r["choices"][0]["finish_reason"], msg, msg.get("tool_calls") or []


def _openai_text(r, stream):
    if stream:
        return "".join(c["choices"][0]["delta"].get("content") or "" for c in r)
    return r["choices"][0]["message"]["content"]


def openai(stream):
    tools = [
        {
            "type": "function",
            "function": {"name": TOOL["name"], "description": TOOL["description"], "parameters": TOOL["input_schema"]},
        }
    ]
    msgs = [{"role": "user", "content": ASK}]
    t = time.time()
    label = f"OpenAI{' stream' if stream else ''}"
    body = {"model": "local", "max_tokens": 300, "tools": tools, "messages": msgs, "stream": stream}
    r = post("/v1/chat/completions", body, stream)
    fin, msg, calls = _openai_tool_message(r, stream)
    ok = fin == "tool_calls" and calls and json.loads(calls[0]["function"]["arguments"]).get("path") == "/etc/hostname"
    check(f"{label}: tool call", ok, json.dumps(r)[:300])
    if not calls:
        return
    msgs += [msg, {"role": "tool", "tool_call_id": calls[0]["id"], "content": "build-box-17"}]
    body["messages"] = msgs
    text = _openai_text(post("/v1/chat/completions", body, stream), stream) or ""
    check(f"{label}: answer uses the result", "build-box-17" in text, text[:200])
    print(f"      {time.time() - t:.1f} s")


for f in (anthropic, openai):
    for s in (False, True):
        f(s)
print("all ok" if not FAILS else f"{FAILS} failures")
sys.exit(1 if FAILS else 0)
