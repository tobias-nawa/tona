"""Server side of scripts/ab.py: start and stop tona-server safely, ask it, parse its answers.

Safety rules (AGENTS.md, GPU safety): the server runs in a systemd scope with a memory limit and no swap, so a leak
ends the scope instead of the desktop; on macOS, which has no systemd, a guard thread ends the started PID once its
resident size exceeds the same limit (ab_rss.py). It is stopped by the PID that was started, never by a name; port 8662
(the service) is refused.
"""

import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

from ab_rss import RssGuard, parse_limit
from ab_workloads import LONGDOC, LONGDOC_OFFSET, PROSE_KINDS, WRITE_OFFSET
from make_corpus import BASE_CHARS

HEADS = {
    "sum": "Summarize what the following C++ and HIP code does, as a bullet list.\n\n```cpp\n",
    "write": "Write a new unit test file for the code below, in the style of its tests. Only code.\n\n```cpp\n",
    "prose": "Read the following texts, then explain in detail, in flowing prose, what they say.\n\n```text\n",
    "longdoc": "Read the following document, then write a detailed report on its main topics.\n\n```text\n",
}
HEAD = HEADS["sum"]
FORBIDDEN_PORT = 8662
DEFAULT_PORT = 8097
BENCH_SET_LINE = re.compile(r"^(Code|Text): \d+ Tokens in [\d.]+ s = ([\d.]+) tok/s decode", re.M)


class Corpora(NamedTuple):
    """The corpora the prompts are cut from: the code corpus and, for the prose workloads, the prose corpus."""

    code: str
    prose: str = ""

    def of(self, kind: str) -> str:
        return self.prose if kind in PROSE_KINDS else self.code


@dataclass(frozen=True)
class Arm:
    name: str
    build: str  # directory of the build, work/build/<name> or a path
    env: dict = field(default_factory=dict)
    model: str = ""  # GGUF of this arm; empty: the model of the ServerConfig (both arms run the same model)
    drafter: str = ""  # MTP head of this arm; empty: the drafter of the ServerConfig


def drafter_path(root: Path, name: str) -> str:
    """The MTP head file of --drafter; `none`: no --mtp option (the head is inside the model file)."""
    return "" if name == "none" else str(root / name)


def arms_from(args, root: Path, parse_env) -> dict:
    """Arms A and B from the options --a, --a-env, --a-model, --a-drafter (and --b-...)."""

    def path(p: str) -> str:
        return str(root / p) if p else ""

    opt = vars(args)
    return {
        n: Arm(n, opt[a], parse_env(opt[a + "_env"]), path(opt[a + "_model"]), path(opt[a + "_drafter"]))
        for n, a in (("A", "a"), ("B", "b"))
    }


@dataclass(frozen=True)
class ServerConfig:
    root: Path
    model: str
    drafter: str
    ctx: int
    port: int = DEFAULT_PORT
    memory_max: str = "54G"
    base_env: dict = field(default_factory=dict)  # for both arms, e.g. TONA_DETERMINISTIC=1
    ready_timeout: float = 900.0
    expert_profile: bool = True  # the model's shipped usage profile (toolchain/profiles/<model id>.bin), if it has one


PROFILE_VAR, PROFILE_OFF = "TONA_EXPERT_PROFILE", "none"  # --a-env TONA_EXPERT_PROFILE=none: no profile in that arm


def profile_of(root: Path, model: str) -> Path | None:
    """The shipped expert profile of a model: toolchain/profiles/<model id>.bin, where the id is the name of the folder
    that holds the GGUF (work/models/<id>/, as the installer lays it out and models.toml names it). An exact id, no
    substring of the path: a profile counts the experts of one GGUF; another file's experts are not the ones it
    names."""
    path = root / "toolchain/profiles" / f"{Path(model).parent.name}.bin"
    return path if path.is_file() else None


def build_dir(root: Path, arm: Arm) -> Path:
    path = Path(arm.build)
    return path if path.is_absolute() or len(path.parts) > 1 else root / "work/build" / arm.build


def child_env(cfg: ServerConfig, arm: Arm) -> dict:
    """Environment of the server: no TONA_* from the caller's shell (it would act on both arms unnoticed)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("TONA_")}
    # OMP_NUM_THREADS is not set here: the server sets the physical cores itself (backend/runtime_env.h)
    env["ROCR_VISIBLE_DEVICES"] = "0"
    profile = profile_of(cfg.root, arm.model or cfg.model) if cfg.expert_profile else None
    if profile:
        env["TONA_EXPERT_PROFILE"] = str(profile)
    env.update(cfg.base_env)
    env.update(arm.env)
    if env.get(PROFILE_VAR) == PROFILE_OFF:  # an arm without the profile, to measure what it is worth
        del env[PROFILE_VAR]
    return env


def server_command(cfg: ServerConfig, arm: Arm) -> list:
    binary = build_dir(cfg.root, arm) / "tona-server"
    drafter = arm.drafter or cfg.drafter  # empty: a model without an MTP head file (Bolt, Ornith) gets no --mtp
    server = [
        str(binary),
        arm.model or cfg.model,
        *(["--mtp", drafter] if drafter else []),
        "--port",
        str(cfg.port),
        "--ctx",
        str(cfg.ctx),
    ]
    if sys.platform == "darwin":  # no systemd: start_server watches the resident size instead (ab_rss.py)
        return server
    return [
        "systemd-run", "--user", "--scope", "-q",
        "-p", f"MemoryMax={cfg.memory_max}", "-p", "MemorySwapMax=0",
        *server,
    ]  # fmt: skip


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def check_port(port: int, wait: float = 0.0) -> None:
    """Refuses the service port and a port in use; with `wait`, gives a server that is just shutting down time."""
    if port == FORBIDDEN_PORT:
        raise SystemExit(f"port {FORBIDDEN_PORT} belongs to the service; choose another one with --port")
    deadline = time.monotonic() + wait
    while not port_free(port):
        if time.monotonic() >= deadline:
            raise SystemExit(f"port {port} is in use (another measurement?); choose another one with --port")
        time.sleep(1)


def vram_used_gib() -> float | None:
    """VRAM in use on the discrete GPU (the card with the most VRAM per sysfs), None where sysfs does not say."""
    cards = []
    for d in Path("/sys/class/drm").glob("card[0-9]*/device"):
        try:
            cards.append((int((d / "mem_info_vram_total").read_text()), int((d / "mem_info_vram_used").read_text())))
        except (OSError, ValueError):
            continue
    return max(cards)[1] / 2**30 if cards else None


def wait_gpu_memory_free(limit_gib: float = 3.0, timeout: float = 90.0) -> None:
    """The kernel frees the memory of an ended server a little later (2026-11-17: the next start found 'VRAM only fits 0
    expert slots'); waits until the discrete GPU is back to the desktop's share, or the timeout, then goes on."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        used = vram_used_gib()
        if used is None or used < limit_gib:
            return
        time.sleep(1)


def start_server(cfg: ServerConfig, arm: Arm, log_path: Path) -> subprocess.Popen:
    """Starts the server in its own session (Ctrl+C reaches only ab.py, which then stops it by PID)."""
    binary = build_dir(cfg.root, arm) / "tona-server"
    if not binary.exists():
        raise SystemExit(f"{binary} does not exist (build {arm.build})")
    cmd = server_command(cfg, arm)
    wait_gpu_memory_free()
    with open(log_path, "ab") as log:  # the child keeps its own copy of the descriptor
        proc = subprocess.Popen(
            cmd,
            env=child_env(cfg, arm),
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )
    if sys.platform == "darwin":
        proc.rss_guard = RssGuard(proc, parse_limit(cfg.memory_max), log_path).start()
    return proc


def is_ready(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=3) as r:
            return bool(json.load(r).get("ready"))
    except (urllib.error.URLError, OSError, ValueError):
        return False


def wait_ready(proc: subprocess.Popen, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            guard = getattr(proc, "rss_guard", None)
            why = f" (resident size {guard.tripped >> 20} MiB over the limit)" if guard and guard.tripped else ""
            raise RuntimeError(f"server exited with {proc.returncode} before it was ready{why}")
        if is_ready(port):
            return
        time.sleep(2)
    raise RuntimeError(f"server not ready after {timeout:.0f} s")


def stop_server(proc: subprocess.Popen | None, grace: float = 120.0) -> None:
    """SIGTERM by PID, SIGKILL after the grace period. Waits until the process is gone (GPU memory is free then)."""
    if proc is None:
        return
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(grace)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
    if getattr(proc, "rss_guard", None):
        proc.rss_guard.stop()


def post(url: str, path: str, body: dict, timeout: float = 3600) -> dict:
    data = json.dumps(body).encode()
    req = urllib.request.Request(url + path, data, {"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def chat_body(prompt: str, max_tokens: int) -> dict:
    return {
        "model": "x",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def count_tokens(url: str, text: str) -> int:
    body = {"model": "x", "messages": [{"role": "user", "content": text}]}
    return post(url, "/v1/messages/count_tokens", body)["input_tokens"]


def prompt_for(url: str, corpus: str, target: int, kind: str = "sum", base_chars: int | None = None) -> str:
    """The longest corpus slice (to 2000 characters) whose prompt has fewer than `target` tokens.

    The "sum" slice starts at the beginning of the corpus. The bisection first runs inside the first `base_chars`
    characters (the sources, where it always ran), and only when even those are too short, in the whole corpus: so
    appending text to the corpus does not move a prompt that fitted before.
    """
    start = {"sum": 0, "write": WRITE_OFFSET, "prose": 0, LONGDOC: LONGDOC_OFFSET}[kind]
    head = HEADS[kind]
    body = corpus[start:]

    def tokens(n: int) -> int:
        return count_tokens(url, head + body[:n] + "\n```")

    lo, hi = 1000, len(body)
    if base_chars and kind == "sum" and base_chars < len(body):
        if tokens(base_chars) >= target:
            hi = base_chars
        else:
            lo = base_chars
    while hi - lo > 2000:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if tokens(mid) < target else (lo, mid)
    if lo >= len(body) - 2000 and tokens(len(body)) < target:
        raise RuntimeError(f"the corpus is too short for {target} tokens: extend scripts/make_corpus.py")
    return head + body[:lo] + "\n```"


def parse_completion(resp: dict) -> dict:
    """Rates and the text hash out of a /v1/chat/completions answer."""
    t = resp["timings"]
    text = resp["choices"][0]["message"]["content"]
    return {
        "prompt_n": t["prompt_n"],
        "prefill": t["prompt_n"] / (t["prompt_ms"] / 1000),
        "decode": t["predicted_per_second"],
        "sha": hashlib.sha256(text.encode()).hexdigest()[:16],
    }


def status_totals(url: str) -> dict:
    """The cumulative counters of GET /status (totals.requests, expert_hits, expert_misses); {} if not there yet."""
    with urllib.request.urlopen(f"{url}/status", timeout=10) as r:
        return json.load(r).get("totals", {})


def expert_counts(before: dict, after: dict) -> dict:
    """Expert cache hits and misses of one request: the difference of the cumulative counters of /status."""
    return {k: after.get(k, 0) - before.get(k, 0) for k in ("expert_hits", "expert_misses")}


def pass_counts(before: dict, after: dict) -> dict:
    """Decode passes, generated tokens and MTP draft steps of one request (differences of the /status counters), so a
    run can say how many tokens a pass gave (acceptance) next to its rate."""
    return {k: after.get(k, 0) - before.get(k, 0) for k in ("passes", "gen_tokens", "mtp_passes")}


def totals_after(url: str, requests_before: int, wait: float = 5.0) -> dict:
    """/status once the request is booked (totals.requests grew); the engine books it right after the answer."""
    deadline = time.monotonic() + wait
    totals = status_totals(url)
    while totals.get("requests", 0) <= requests_before and time.monotonic() < deadline:
        time.sleep(0.1)
        totals = status_totals(url)
    return totals


def run_context(
    url: str,
    corpora: Corpora,
    requests: list,
    prompts: dict,
    suffixes: dict | None = None,
    counters: dict | None = None,
) -> tuple:
    """One request each (ab_workloads.Request): (metrics, shas). Metrics prefill@<name><suffix> and decode@<name>
    <suffix> in tok/s; suffixes {name: '/ctx<N>'} tells apart the same request under several --ctx values."""
    metrics, shas = {}, {}
    for req in requests:
        key = (req.kind, req.tokens)
        if key not in prompts:  # tokenizer-dependent only, so one bisection serves every server start
            prompts[key] = prompt_for(url, corpora.of(req.kind), req.tokens, req.kind, BASE_CHARS)
        before = status_totals(url) if counters is not None else {}
        got = parse_completion(post(url, "/v1/chat/completions", chat_body(prompts[key], req.decode)))
        suffix = (suffixes or {}).get(req.name, "")
        if counters is not None:
            after = totals_after(url, before.get("requests", 0))
            counters[f"{req.name}{suffix}"] = {**expert_counts(before, after), **pass_counts(before, after)}
        metrics[f"prefill@{req.name}{suffix}"] = got["prefill"]
        metrics[f"decode@{req.name}{suffix}"] = got["decode"]
        shas[f"{req.name}{suffix}"] = got["sha"]
    return metrics, shas


def parse_bench_set(output: str) -> dict:
    """decode-code and decode-text (tok/s) out of the printout of scripts/bench-set.py."""
    found = {f"decode-{m.group(1).lower()}": float(m.group(2)) for m in BENCH_SET_LINE.finditer(output)}
    if set(found) != {"decode-code", "decode-text"}:
        raise ValueError(f"bench-set output not understood: {output[-300:]!r}")
    return found


def run_bench_set(url: str, root: Path) -> dict:
    out = subprocess.run(
        ["python3", "-I", str(root / "scripts/bench-set.py"), url],
        check=True, capture_output=True, text=True, timeout=3600,
    )  # fmt: skip
    return parse_bench_set(out.stdout)
