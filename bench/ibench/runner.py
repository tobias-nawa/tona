"""Runs the cases of a suite against a target and computes the metrics."""

import hashlib
import re
import sys
import time
import uuid

from . import client, stats
from .suite import Case, bodies

# Per-request metrics, in report order
METRICS = [
    "ttft_s",
    "prefill_tps_client",
    "prefill_tps_server",
    "decode_tps_client",
    "decode_tps_server",
    "tokens_per_pass",
    "draft_accept_rate",
    "e2e_s",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "itl_p50_ms",
    "itl_p99_ms",
    "vram_peak_mib",
    "gtt_peak_mib",
    "ram_avail_min_mib",
    "swap_used_max_mib",
    "gpu_busy_mean",
    "power_mean_w",
    "temp_max_c",
    "sclk_mean_mhz",
]


def _client_rates(m: dict, r: client.Result, case: Case, cached) -> int:
    """Adds the client-side latency and rate metrics; returns the completion token count."""
    if r.t_first is not None:
        m["ttft_s"] = r.t_first - r.t_start
        if r.prompt_tokens and m["ttft_s"] > 0:
            # includes HTTP and the first generated token; on cache hits only the uncached rest
            m["prefill_tps_client"] = (r.prompt_tokens - (cached or 0)) / m["ttft_s"]
    n = r.completion_tokens
    if n is None:
        n = len(r.chunk_times)
        m["completion_tokens_estimated"] = True
    if r.t_first is not None and r.t_last is not None:
        span = r.t_last - r.t_first
        if n >= case.min_decode_tokens and span > 0:
            m["decode_tps_client"] = (n - 1) / span
    gaps = [(b - a) * 1000 for a, b in zip(r.chunk_times, r.chunk_times[1:], strict=False)]
    if gaps:
        m["itl_p50_ms"] = stats.percentile(gaps, 0.5)
        m["itl_p99_ms"] = stats.percentile(gaps, 0.99)
    return n


def _draft_metrics(m: dict, t: dict, n: int) -> None:
    if not t.get("draft_n"):
        return
    accepted = t.get("draft_n_accepted") or 0
    produced = t.get("predicted_n") or n
    passes = produced - accepted  # each verify pass yields the accepted drafts plus one token
    if passes > 0:
        m["tokens_per_pass"] = produced / passes
    m["draft_accept_rate"] = accepted / t["draft_n"]


def metrics(r: client.Result, case: Case) -> dict:
    t = r.timings or {}
    cached = r.cached_tokens if r.cached_tokens is not None else t.get("cache_n")
    m = {
        "ok": r.ok,
        "error": r.error,
        "finish": r.finish,
        "prompt_tokens": r.prompt_tokens,
        "cached_tokens": cached,
        "completion_tokens": r.completion_tokens,
        "tool_calls": r.tool_calls,
        "reasoning_tokens": r.reasoning_tokens,
        "reasoning_chars": len(r.reasoning),
        "e2e_s": r.t_end - r.t_start,
        "prefill_tps_server": t.get("prompt_per_second"),
        "decode_tps_server": t.get("predicted_per_second"),
        "timings": r.timings,
    }
    n = _client_rates(m, r, case, cached)
    _draft_metrics(m, t, n)
    m["expect_ok"] = bool(re.search(case.expect, r.text, re.S)) if case.expect and r.ok else None
    m["output_sha"] = hashlib.sha256((r.text + "\0" + r.reasoning).encode()).hexdigest()[:12]
    m["text_head"] = r.text[:400]
    return m


def _fmt(v, nd=1):
    return "-" if v is None else f"{v:.{nd}f}"


def _log(case: Case, label: str, m: dict) -> None:
    if not m["ok"]:
        print(f"[{case.name}] {label}: ERROR {m['error']}", file=sys.stderr)
        return
    exp = {True: "ok", False: "WRONG", None: "-"}[m["expect_ok"]]
    print(
        f"[{case.name}] {label}: TTFT {_fmt(m.get('ttft_s'), 2)} s, "
        f"prefill {_fmt(m.get('prefill_tps_server'))} tok/s, "
        f"decode {_fmt(m.get('decode_tps_server'))} tok/s (client {_fmt(m.get('decode_tps_client'))}), "
        f"{m.get('prompt_tokens')}→{m.get('completion_tokens')} tokens, cache {m.get('cached_tokens')}, "
        f"expected {exp}",
        file=sys.stderr,
    )


def run_case(case: Case, target: dict, mon) -> dict:
    runs = []
    total = case.warmup + case.runs
    for i in range(total):
        measured = i >= case.warmup
        nonce = uuid.uuid4().hex[:12] if case.cache == "cold" else None
        steps = bodies(case, target["api"], target["model"], i, nonce, target.get("extra", {}))
        label = f"run {i - case.warmup + 1}/{case.runs}" if measured else f"warm-up {i + 1}/{case.warmup}"
        run = {"index": i - case.warmup, "warmup": not measured, "nonce": nonce, "steps": []}
        for k, body in enumerate(steps):
            res = client.stream(
                target["url"], target["api"], body, target.get("api_key", "none"), target.get("timeout", 900.0)
            )
            m = metrics(res, case)
            m.update(mon.window(res.t_start, res.t_end))
            run["steps"].append(m)
            _log(case, label + (f", step {k + 1}/{len(steps)}" if len(steps) > 1 else ""), m)
            if not res.ok:
                break
        runs.append(run)
        if case.pause_s:
            time.sleep(case.pause_s)
    return {"case": case.__dict__, "runs": runs, "summary": summarize(case, runs)}


def _step_summary(ms: list[dict]) -> dict:
    ok = [m for m in ms if m["ok"]]
    s = {"n": len(ms), "ok": len(ok), "metrics": {}}
    for key in METRICS:
        v = stats.summarize(m.get(key) for m in ok)
        if v:
            s["metrics"][key] = v
    checks = [m["expect_ok"] for m in ok if m["expect_ok"] is not None]
    s["expect_pass"] = sum(checks) if checks else None
    s["expect_n"] = len(checks)
    s["distinct_outputs"] = len({m["output_sha"] for m in ok})
    return s


def _warnings(case: Case, measured: list[dict], steps: list[dict]) -> list[str]:
    w = []
    if case.cache == "cold":
        hits = [r["steps"][0].get("cached_tokens") or 0 for r in measured if r["steps"]]
        if any(h > 64 for h in hits):
            w.append(f"cold case with cache hits (up to {max(hits)} tokens): the nonce has no effect")
    for s in steps:
        if s["ok"] < s["n"]:
            w.append(f"{s['n'] - s['ok']} of {s['n']} requests failed")
        if s["n"] and "decode_tps_client" not in s["metrics"] and "decode_tps_server" not in s["metrics"]:
            w.append(f"no decode rate: answers shorter than {case.min_decode_tokens} tokens")
    return w


def summarize(case: Case, runs: list[dict]) -> dict:
    measured = [r for r in runs if not r["warmup"]]
    n_steps = max((len(r["steps"]) for r in measured), default=0)
    steps = [_step_summary([r["steps"][k] for r in measured if len(r["steps"]) > k]) for k in range(n_steps)]
    totals = [sum(m["e2e_s"] for m in r["steps"]) for r in measured if r["steps"] and all(m["ok"] for m in r["steps"])]
    return {
        "steps": steps,
        "warnings": _warnings(case, measured, steps),
        "run_e2e_s": stats.summarize(totals),
    }
