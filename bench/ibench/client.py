"""One streamed request against an OpenAI- or Anthropic-compatible API, with timestamps."""

import http.client
import json
import time
from dataclasses import dataclass, field
from urllib.parse import urlsplit

PATHS = {"openai": "/v1/chat/completions", "anthropic": "/v1/messages"}


@dataclass
class Result:
    ok: bool = False
    error: str | None = None
    status: int | None = None
    # times in seconds, time.perf_counter()
    t_start: float = 0.0
    t_first: float | None = None  # first event with output (answer, reasoning or tool call)
    t_last: float | None = None
    t_end: float = 0.0
    chunk_times: list[float] = field(default_factory=list)
    text: str = ""
    reasoning: str = ""
    tool_calls: int = 0
    finish: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cached_tokens: int | None = None
    reasoning_tokens: int | None = None
    timings: dict | None = None  # server-reported timings, llama.cpp field names


def _events(resp):
    """SSE events as (event, data)."""
    event, data = None, []
    while True:
        line = resp.readline()
        if not line:
            break
        line = line.decode("utf-8", "replace").rstrip("\r\n")
        if not line:
            if data:
                yield event, "\n".join(data)
            event, data = None, []
        elif line.startswith(":"):
            continue
        elif line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:"):
            value = line[5:]
            data.append(value[1:] if value.startswith(" ") else value)
    if data:
        yield event, "\n".join(data)


def _openai(r: Result, obj: dict) -> bool:
    out = False
    for ch in obj.get("choices") or []:
        d = ch.get("delta") or {}
        if d.get("content"):
            r.text += d["content"]
            out = True
        rc = d.get("reasoning_content") or d.get("reasoning")
        if rc:
            r.reasoning += rc
            out = True
        for tc in d.get("tool_calls") or []:
            if tc.get("id"):  # only the first fragment of a call carries the id
                r.tool_calls += 1
            out = True
        if ch.get("finish_reason"):
            r.finish = ch["finish_reason"]
    u = obj.get("usage")
    if u:
        r.prompt_tokens = u.get("prompt_tokens", r.prompt_tokens)
        r.completion_tokens = u.get("completion_tokens", r.completion_tokens)
        r.cached_tokens = (u.get("prompt_tokens_details") or {}).get("cached_tokens", r.cached_tokens)
        r.reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens", r.reasoning_tokens)
    if obj.get("timings"):
        r.timings = obj["timings"]
    return out


def _anthropic_usage(r: Result, u: dict) -> None:
    if "input_tokens" in u:
        # Anthropic's input_tokens excludes cache hits
        cached = u.get("cache_read_input_tokens") or 0
        r.prompt_tokens = u["input_tokens"] + cached + (u.get("cache_creation_input_tokens") or 0)
        r.cached_tokens = cached
    if "output_tokens" in u:
        r.completion_tokens = u["output_tokens"]


def _anthropic_delta(r: Result, d: dict) -> bool:
    kind = d.get("type")
    if kind == "text_delta" and d.get("text"):
        r.text += d["text"]
        return True
    if kind == "thinking_delta" and d.get("thinking"):
        r.reasoning += d["thinking"]
        return True
    return kind == "input_json_delta" and bool(d.get("partial_json"))


def _anthropic(r: Result, obj: dict) -> bool:
    out = False
    t = obj.get("type")
    if t == "message_start":
        _anthropic_usage(r, (obj.get("message") or {}).get("usage") or {})
    elif t == "content_block_start":
        if (obj.get("content_block") or {}).get("type") == "tool_use":
            r.tool_calls += 1
            out = True
    elif t == "content_block_delta":
        out = _anthropic_delta(r, obj.get("delta") or {})
    elif t == "message_delta":
        _anthropic_usage(r, obj.get("usage") or {})
        r.finish = (obj.get("delta") or {}).get("stop_reason") or r.finish
    if obj.get("timings"):
        r.timings = obj["timings"]
    return out


def _request_parts(url: str, api: str, body: dict, api_key: str) -> tuple[dict, dict, bytes]:
    body = dict(body, stream=True)
    if api == "openai":
        body.setdefault("stream_options", {"include_usage": True})
        headers = {"Authorization": f"Bearer {api_key}"}
    elif api == "anthropic":
        headers = {"x-api-key": api_key, "anthropic-version": "2023-06-01"}
    else:
        raise ValueError(f"unknown API {api!r}")
    headers["Content-Type"] = "application/json"
    return body, headers, json.dumps(body).encode()


def _consume(r: Result, resp, handle) -> None:
    for _event, data in _events(resp):
        if data == "[DONE]":
            break
        now = time.perf_counter()
        try:
            obj = json.loads(data)
        except ValueError:
            continue
        if obj.get("type") == "error" or "error" in obj:
            r.error = json.dumps(obj.get("error", obj))[:2000]
            break
        if handle(r, obj):
            r.chunk_times.append(now)
            if r.t_first is None:
                r.t_first = now
            r.t_last = now


def stream(url: str, api: str, body: dict, api_key: str = "none", timeout: float = 900.0) -> Result:
    """Streams body to url (without /v1) and measures. Errors end up in Result.error, never as an exception."""
    u = urlsplit(url)
    _, headers, payload = _request_parts(url, api, body, api_key)
    handle = _openai if api == "openai" else _anthropic

    r = Result()
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.connect()
        r.t_start = time.perf_counter()
        conn.request("POST", u.path.rstrip("/") + PATHS[api], payload, headers)
        resp = conn.getresponse()
        r.status = resp.status
        if resp.status != 200:
            r.error = f"HTTP {resp.status}: {resp.read(2000).decode('utf-8', 'replace')}"
            return r
        _consume(r, resp, handle)
        r.ok = r.error is None and r.t_first is not None
        if r.error is None and r.t_first is None:
            r.error = "no output in the stream"
    except (OSError, http.client.HTTPException) as e:
        r.error = f"{type(e).__name__}: {e}"
    finally:
        r.t_end = time.perf_counter()
        conn.close()
    return r
