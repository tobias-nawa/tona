"""How long a desktop-like GPU submission waits while Tona runs (research/63-desktop-latency.md).
One arm per call.

    python3 tools/desktop_latency.py --arm NAME --run-dir DIR [--env K=V ...] [--canary-priority low|medium|high]

A headless Vulkan client (bench/micro/vram_canary.cpp, built into work/vram/) submits a small fill every 20 ms and
measures how long each submission takes; it stands in for a compositor frame. Tona runs next to it in three phases:
idle (server up, no request), decode (400-token requests) and prefill (2000-token prompts, each one different so that
no prompt cache is hit). Per phase: canary p50, p99, max, the share of submissions under the 16.7 ms frame budget, and
Tona's rates (decode tok/s or prefill tok/s as Tona reports them). No VRAM pressure: no balloon, the canary holds 64
MiB. Stops at the first of the STOP file DIR/STOP, a kernel log line with a GPU reset or ring timeout, a canary error.

Run it only as an exclusive queue job (scripts/gpu_queue.py --kind exclusive): the desktop guard then keeps it away from
a used desktop. Output: DIR/summary.json (sums and times only, no program names).
"""

from __future__ import annotations

import argparse
import json
import signal
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "scripts"))
import ab  # noqa: E402
import ab_server  # noqa: E402
import vram_balloon  # noqa: E402
import vram_level0  # noqa: E402

FRAME_US = 16700  # one frame at 60 Hz
CANARY = ROOT / "work/vram/vram_canary"
DECODE_PROMPT = vram_level0.PROMPT
PREFILL_TOKENS = 2000


def percentile(values: list[float], p: float) -> float | None:
    return vram_level0.percentile(values, p)


def frame_hit_rate(lat_us: list[float], budget_us: float = FRAME_US) -> float | None:
    """Share of submissions that finished within the frame budget."""
    return sum(1 for x in lat_us if x <= budget_us) / len(lat_us) if lat_us else None


def phase_summary(canary: list[tuple[float, float]], rates: list[tuple[float, float]], t0: float, t1: float) -> dict:
    """Canary latency and Tona's rate in the wall-clock window [t0, t1)."""
    lat = [us for t, us in canary if t0 <= t < t1]
    r = [x for t, x in rates if t0 <= t < t1]
    return {
        "seconds": round(t1 - t0, 1),
        "canary_submissions": len(lat),
        "canary_us_p50": percentile(lat, 0.5),
        "canary_us_p99": percentile(lat, 0.99),
        "canary_us_max": max(lat) if lat else None,
        "frame_hit_rate": None if not lat else round(frame_hit_rate(lat), 4),
        "tona_requests": len(r),
        "tona_tok_s_mean": round(statistics.fmean(r), 2) if r else None,
    }


class Worker(threading.Thread):
    """Sends the requests of the current mode; records (end time, rate) for decode and prefill."""

    def __init__(self, url: str, prefill_prompt: str) -> None:
        super().__init__(daemon=True)
        self.url, self.prefill_prompt = url, prefill_prompt
        self.mode = "idle"
        self.stop = threading.Event()
        self.rates: dict[str, list[tuple[float, float]]] = {"decode": [], "prefill": []}
        self.error: str | None = None
        self.n = 0

    def request(self, mode: str) -> None:
        self.n += 1
        if mode == "decode":
            body = ab_server.chat_body(DECODE_PROMPT, 400)
            t = ab_server.post(self.url, "/v1/chat/completions", body, 300)["timings"]
            self.rates["decode"].append((time.time(), t["predicted_per_second"]))
        else:
            body = ab_server.chat_body(f"Request {self.n}.\n" + self.prefill_prompt, 8)  # unique head: no cache hit
            t = ab_server.post(self.url, "/v1/chat/completions", body, 300)["timings"]
            self.rates["prefill"].append((time.time(), t["prompt_n"] / (t["prompt_ms"] / 1000)))

    def run(self) -> None:
        while not self.stop.is_set():
            mode = self.mode
            if mode == "idle":
                time.sleep(0.2)
                continue
            try:
                self.request(mode)
            except Exception as exc:  # a failed request is a finding, not something to hide
                if not self.stop.is_set():
                    self.error = f"request failed: {type(exc).__name__}"
                return


def canary_allowed(priority: str) -> str | None:
    """Starts the canary alone for a second to see whether the driver grants the priority; None if it does."""
    run = subprocess.run(
        [str(CANARY), "--seconds", "1", "--period-ms", "100", "--priority", priority],
        capture_output=True,
        text=True,
        check=False,
    )
    err = [line for line in run.stdout.splitlines() if '"error"' in line]
    return err[0] if run.returncode != 0 and err else None


def run_phases(args: argparse.Namespace, rec: vram_level0.Recorder, worker: Worker, klog) -> dict:
    bounds: dict[str, tuple[float, float]] = {}
    for mode, seconds in (("idle", args.idle_s), ("decode", args.decode_s), ("prefill", args.prefill_s)):
        worker.mode = mode
        t0 = time.time()
        while time.time() < t0 + seconds:
            if (args.run_dir / "STOP").exists() or klog.abort_line or rec.canary_error or worker.error:
                bounds[mode] = (t0, time.time())
                return bounds
            time.sleep(0.5)
        bounds[mode] = (t0, time.time())
    worker.mode = "idle"
    return bounds


def run(args: argparse.Namespace, env: dict) -> dict:
    summary: dict = {"arm": args.arm, "env": env, "canary_priority": args.canary_priority, "build": args.build}
    denied = canary_allowed(args.canary_priority)
    if denied:
        summary["not_run"] = "canary priority not granted: " + denied
        return summary
    ab_server.check_port(args.port)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    klog = vram_balloon.KernelLog()
    klog.start()
    cfg = ab_server.ServerConfig(ROOT, str(ROOT / args.model), str(ROOT / args.drafter), args.ctx, args.port, "54G")
    server = canary = None
    rec = vram_level0.Recorder()
    try:
        server = ab_server.start_server(cfg, ab_server.Arm(args.arm, args.build, env), args.run_dir / "server.log")
        ab_server.wait_ready(server, args.port, cfg.ready_timeout)
        url = f"http://127.0.0.1:{args.port}"
        corpus = ab.load_corpus(ROOT / "work/corpus/code.txt")
        worker = Worker(url, ab_server.prompt_for(url, corpus, PREFILL_TOKENS, "sum"))
        cmd = [str(CANARY), "--seconds", "1800", "--period-ms", "20", "--priority", args.canary_priority]
        cmd += ["--stop-file", str(args.run_dir / "STOP")]
        canary = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
        threading.Thread(target=rec.read_canary, args=(canary,), daemon=True).start()
        worker.start()
        time.sleep(5)  # the first submissions are not a phase
        bounds = run_phases(args, rec, worker, klog)
        worker.stop.set()
        summary["phases"] = {m: phase_summary(rec.canary, worker.rates.get(m, []), *b) for m, b in bounds.items()}
        summary["stop_reason"] = rec.canary_error or worker.error or klog.abort_line
    finally:
        shutdown(canary, server, klog)
    return summary


def shutdown(canary: subprocess.Popen | None, server: subprocess.Popen | None, klog) -> None:
    if canary is not None and canary.poll() is None:
        canary.send_signal(signal.SIGTERM)
        try:
            canary.wait(15)
        except subprocess.TimeoutExpired:
            canary.kill()
    klog.close()
    ab_server.stop_server(server)
    ab_server.wait_gpu_memory_free()


def parse_env(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, value = item.partition("=")
        if not key.startswith("TONA_"):
            raise SystemExit(f"--env {item}: only TONA_* variables")
        out[key] = value
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--canary-priority", default="medium", choices=["low", "medium", "high"])
    ap.add_argument("--build", default="b164t")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--ctx", type=int, default=140000)
    ap.add_argument("--model", default=ab.MODEL)
    ap.add_argument("--drafter", default=ab.DRAFTER)
    ap.add_argument("--idle-s", type=float, default=20.0)
    ap.add_argument("--decode-s", type=float, default=40.0)
    ap.add_argument("--prefill-s", type=float, default=40.0)
    args = ap.parse_args(argv)
    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    args.run_dir.mkdir(parents=True, exist_ok=True)
    summary = run(args, parse_env(args.env))
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if not summary.get("stop_reason") and "not_run" not in summary else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
