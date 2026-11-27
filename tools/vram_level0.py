"""Level 0 of the compositor survival test (research/62): Tona decodes while a polite foreign GPU program takes
VRAM in 64 MiB steps and never takes the card below 1.5 GiB free. One arm per call.

    python3 tools/vram_level0.py --arm default|chunked|margin|both --run-dir DIR [--build b164t] [--port 8099]

Phases (each with its own decode rate and canary latency): base (Tona and the canary only), balloon (steps, then
hold), after (the memory is back). The canary (bench/micro/vram_canary.cpp, built into work/vram/) is a headless
Vulkan client that submits a fill every 100 ms: the compositor's kind of command submission. The balloon
(tools/vram_balloon.py) and a watchdog (tools/vram_watchdog.py) are processes of their own. Everything ends at the first
stop rule of tools/vram_guard.py: the STOP file DIR/STOP, a kernel log line (GPU reset, ring timeout, failed command
submission), a canary error, the time cap. Output: DIR/summary.json, DIR/samples.jsonl; sums and times only, no
program names.

Run it only as an exclusive queue job (scripts/gpu_queue.py --kind exclusive): the desktop guard then keeps it away
from a used desktop. Level 1 (taking the card to full) is NOT part of this tool and must not be built from it.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
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
import vram_guard as vg  # noqa: E402

ARMS = {
    "default": {},
    "chunked": {"TONA_SLOTS_CHUNKED": "1"},
    "margin": {"TONA_GTT_MARGIN_MB": "3072"},
    "both": {"TONA_SLOTS_CHUNKED": "1", "TONA_GTT_MARGIN_MB": "3072"},
}
CANARY = ROOT / "work/vram/vram_canary"
PROMPT = "Write a long, well commented Python module that implements a priority queue, a LRU cache and tests for both."
DECODE_TOKENS = 400


class Recorder:
    """Decode requests and canary submissions with their wall-clock times, filled by two threads."""

    def __init__(self) -> None:
        self.decodes: list[tuple[float, float]] = []  # (end time, decode tok/s)
        self.canary: list[tuple[float, float]] = []  # (time, latency in us)
        self.canary_error: str | None = None
        self.decode_error: str | None = None
        self.stop = threading.Event()

    def decode_loop(self, url: str) -> None:
        while not self.stop.is_set():
            try:
                resp = ab_server.post(url, "/v1/chat/completions", ab_server.chat_body(PROMPT, DECODE_TOKENS), 300)
                self.decodes.append((time.time(), resp["timings"]["predicted_per_second"]))
            except Exception as exc:  # a failed request is a finding; the loop must not hide it by dying silently
                if not self.stop.is_set():  # a request cut off by the shutdown at the end is not a failure
                    self.decode_error = f"decode request failed: {type(exc).__name__}"
                return

    def read_canary(self, proc: subprocess.Popen) -> None:
        for line in proc.stdout:
            try:
                rec = json.loads(line)
            except ValueError:
                continue
            if "error" in rec:
                self.canary_error = f"canary: {rec['error']} ({rec.get('code')})"
            elif "us" in rec:
                self.canary.append((time.time(), rec["us"]))


def evicted_ms(pid: int) -> int | None:
    """Cumulative queue eviction time of a process (KFD), None where the counters are not exposed."""
    total, seen = 0, False
    for path in glob.glob(f"/sys/class/kfd/kfd/proc/{pid}/stats_*/evicted_ms"):
        try:
            total += int(Path(path).read_text())
            seen = True
        except (OSError, ValueError):
            continue
    return total if seen else None


def percentile(values: list[float], p: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(p * len(ordered)))]


def phase_summary(rec: Recorder, t0: float, t1: float, samples: list[dict]) -> dict:
    """Numbers of one phase [t0, t1) in wall-clock time."""
    rates = [r for t, r in rec.decodes if t0 <= t < t1]
    lat = [us for t, us in rec.canary if t0 <= t < t1]
    free = [s["free_mib"] for s in samples if t0 <= s["t"] < t1 and s["free_mib"] is not None]
    return {
        "seconds": round(t1 - t0, 1),
        "decode_requests": len(rates),
        "decode_tok_s_mean": round(statistics.fmean(rates), 2) if rates else None,
        "canary_submissions": len(lat),
        "canary_us_p50": percentile(lat, 0.5),
        "canary_us_p99": percentile(lat, 0.99),
        "canary_us_max": max(lat) if lat else None,
        "free_mib_min": min(free) if free else None,
    }


def balloon_events(run_dir: Path) -> list[dict]:
    path = run_dir / "balloon.jsonl"
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


class Job:
    """One arm: Tona, canary, balloon, watchdog, and the loop that watches them."""

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.dev = vg.pick_card()
        self.rec = Recorder()
        self.klog = vram_balloon.KernelLog()
        self.samples: list[dict] = []
        self.procs: dict[str, subprocess.Popen] = {}
        self.t_start = time.time()
        self.stop_why: str | None = None
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stop_why", "SIGTERM"))

    def sample(self) -> None:
        dev = self.dev
        free = vg.read_free(dev)
        self.samples.append({
            "t": time.time(),
            "free_mib": None if free is None else free // vg.MIB,
            "gtt_used_mib": (vg.read_counter(dev, "mem_info_gtt_used") or 0) // vg.MIB,
        })  # fmt: skip

    def check(self) -> str | None:
        """The stop rules of the orchestrator; the balloon has its own copy of the ones about memory."""
        state = vg.StopState(
            time.time() - self.t_start,
            (self.args.run_dir / "STOP").exists(),
            self.klog.abort_line,
            self.rec.canary_error is not None,
            None,
            vg.read_free(self.dev),
        )
        # the floor rule belongs to the balloon (it is about its next step); here free memory only has to be readable
        return self.stop_why or self.rec.decode_error or vg.stop_reason(state, floor=0)

    def wait(self, seconds: float, until: subprocess.Popen | None = None) -> str | None:
        """Waits `seconds` (or until the process `until` ends), sampling at 2 Hz and checking the stop rules."""
        end = time.time() + seconds
        while time.time() < end:
            self.sample()
            why = self.check()
            if why is not None:
                return why
            if until is not None and until.poll() is not None:
                return None
            time.sleep(0.5)
        return None

    def start_balloon(self) -> subprocess.Popen:
        beat = self.args.run_dir / "balloon.heartbeat"
        beat.unlink(missing_ok=True)
        balloon = subprocess.Popen(
            [sys.executable, str(ROOT / "tools/vram_balloon.py"), "--run-dir", str(self.args.run_dir)]
        )
        self.procs["balloon"] = balloon
        dog_cmd = [sys.executable, str(ROOT / "tools/vram_watchdog.py"), "--heartbeat", str(beat)]
        dog_cmd += ["--parent", str(os.getpid()), "--watch", str(balloon.pid)]
        dog_cmd += ["--kill", str(balloon.pid), "--kill", str(self.procs["canary"].pid)]
        dog = subprocess.Popen(dog_cmd)
        self.procs["watchdog"] = dog
        return balloon

    def stop_balloon(self, balloon: subprocess.Popen) -> None:
        balloon.send_signal(signal.SIGTERM)  # it frees everything on SIGTERM
        try:
            balloon.wait(15)
        except subprocess.TimeoutExpired:
            balloon.kill()
            balloon.wait()


def run_phases(job: Job) -> dict:
    """base, balloon, after; returns the wall-clock bounds of each phase that ran."""
    a = job.args
    bounds: dict[str, tuple[float, float]] = {}
    t0 = time.time()
    job.stop_why = job.wait(a.base_s)
    bounds["base"] = (t0, time.time())
    if job.stop_why:
        return bounds
    t0 = time.time()
    balloon = job.start_balloon()
    job.stop_why = job.wait(a.balloon_max_s, until=balloon)
    if balloon.poll() is None:
        job.stop_balloon(balloon)
    bounds["balloon"] = (t0, time.time())
    if job.stop_why:
        return bounds
    t0 = time.time()
    job.stop_why = job.wait(a.after_s)
    bounds["after"] = (t0, time.time())
    return bounds


def run(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    job = Job(args)
    if job.dev is None:
        raise SystemExit("no discrete GPU in sysfs")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "balloon.jsonl").unlink(missing_ok=True)
    if (args.run_dir / "STOP").exists():
        raise SystemExit("STOP file present, not starting")
    ab_server.check_port(args.port)
    job.klog.start()
    cfg = ab_server.ServerConfig(
        ROOT, str(ROOT / args.model), str(ROOT / args.drafter), args.ctx, args.port, args.memory_max
    )
    # level 0 needs free VRAM above the balloon's floor: the default reserve (1 GiB) leaves less than 1.5 GiB, so every
    # arm keeps more free by the documented knob; the arms differ only by their own variables
    env = {"TONA_VRAM_RESERVE_MB": str(args.reserve_mb), **ARMS[args.arm]}
    arm = ab_server.Arm(args.arm, args.build, env)
    server = None
    summary: dict = {"arm": args.arm, "env": env, "build": args.build}
    try:
        server = ab_server.start_server(cfg, arm, args.run_dir / "server.log")
        ab_server.wait_ready(server, args.port, cfg.ready_timeout)
        summary["free_mib_server_ready"] = (vg.read_free(job.dev) or 0) // vg.MIB
        summary["evicted_ms_before"] = evicted_ms(server.pid)
        canary = subprocess.Popen([str(CANARY), "--seconds", "1800", "--stop-file", str(args.run_dir / "STOP")],
                                  stdout=subprocess.PIPE, text=True)  # fmt: skip
        job.procs["canary"] = canary
        threading.Thread(target=job.rec.read_canary, args=(canary,), daemon=True).start()
        threading.Thread(target=job.rec.decode_loop, args=(f"http://127.0.0.1:{args.port}",), daemon=True).start()
        time.sleep(10)  # the first requests and submissions are not a phase
        bounds = run_phases(job)
        summary["phases"] = {k: phase_summary(job.rec, *v, job.samples) for k, v in bounds.items()}
        summary["evicted_ms_after"] = evicted_ms(server.pid)
    finally:
        job.rec.stop.set()
        shutdown(job, server)
    summary["stop_reason"] = job.stop_why or job.rec.canary_error or job.rec.decode_error
    summary["canary_error"] = job.rec.canary_error
    summary["kernel_abort"] = job.klog.abort_line
    summary["balloon"] = balloon_events(args.run_dir)
    return summary, job.samples


def shutdown(job: Job, server: subprocess.Popen | None) -> None:
    for name in ("balloon", "canary", "watchdog"):
        proc = job.procs.get(name)
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
            try:
                proc.wait(15)
            except subprocess.TimeoutExpired:
                proc.kill()
    job.klog.close()
    ab_server.stop_server(server)
    ab_server.wait_gpu_memory_free()


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--build", default="b164t")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--reserve-mb", type=int, default=3072)
    ap.add_argument("--ctx", type=int, default=140000)
    ap.add_argument("--model", default=ab.MODEL)
    ap.add_argument("--drafter", default=ab.DRAFTER)
    ap.add_argument("--memory-max", default="54G")
    ap.add_argument("--base-s", type=float, default=40.0)
    ap.add_argument("--balloon-max-s", type=float, default=120.0)
    ap.add_argument("--after-s", type=float, default=30.0)
    args = ap.parse_args(argv)
    summary, samples = run(args)
    (args.run_dir / "samples.jsonl").write_text("".join(json.dumps(x) + "\n" for x in samples))
    (args.run_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "balloon"}, indent=2))
    return 0 if summary["stop_reason"] is None else 3


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
