#!/usr/bin/env python3
"""R8 pressure series (docs/contribute/measurement.md): pairs of a 30K and a 130K prompt, every second under pressure.

    scripts/r8_pressure.py --build w163o --pressure-files '/path/to/big/*.gguf' [--pairs 10] [--port 8091]

Question it answers: does a decode stall, abort or run slow when the page cache is busy (memory, pinning and stall
paths of the engine)? Each pair sends a 30K and a 130K prompt from the reference corpus (scripts/make_corpus.py), 200
tokens greedy, to one server (the same one for the whole series; a failed pass is recorded and the server restarted).
The odd pairs run while a reader pulls the --pressure-files (big files the server does not use) through the page cache.

RAM only, capped, no VRAM pressure: the reader runs in its own `systemd-run --user --scope` with RuntimeMaxSec, so it
ends even if this script dies; the server runs in a scope with MemoryMax and MemorySwapMax=0 (scripts/ab_server.py).
The script adds no VRAM load; before every start it waits until the VRAM of a previous server is free.

Per request, one line in OUT/<tag>.jsonl: decode and prefill rate, the eviction time of the KFD processes during the
request and its longest episode (10 ms sampler of evicted_ms), the memory pressure (PSI) and compaction stalls of the
kernel. At the end a summary against the usual expectation (no abort, at most 3 slow of 10 pressured requests, evicted
time below 5 s in total and no episode of 3 s or more in the pressured requests). The port is an argument, never the
port of the running service (8662).
"""

import argparse
import contextlib
import glob
import json
import os
import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ab_server as S
import make_corpus

REPO = Path(__file__).resolve().parent.parent
MODEL = "work/models/coder-iq1_m/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf"
DRAFTER = "work/models/draft/mtp-Qwen3.8-Flash-Next-Q8_0.gguf"
FORBIDDEN_PORTS = (8662, 8097)  # the service and the port that a past agent used for it
SLOW_TOK_S = 55.0  # a decode below this is "slow" (research/measurements/2026-11-01-slow-decode.md)
TARGETS = (30000, 130000)
KFD_EVICTED = "/sys/class/kfd/kfd/proc/*/stats_*/evicted_ms"
VMSTAT_KEYS = ("allocstall_normal", "pgscan_direct", "compact_stall", "compact_isolated", "pgmigrate_success")


def read_int(path: str) -> int:
    try:
        return int(Path(path).read_text())
    except (OSError, ValueError):
        return 0


def evicted_ms() -> int:
    """Eviction time summed over all KFD processes."""
    return sum(read_int(p) for p in glob.glob(KFD_EVICTED))


def psi_memory() -> dict:
    """Stall time in us of the memory pressure (some and full)."""
    out = {}
    for line in Path("/proc/pressure/memory").read_text().splitlines():
        kind, *fields = line.split()
        out[f"memory_{kind}_us"] = int(dict(f.split("=") for f in fields)["total"])
    return out


def vmstat() -> dict:
    table = dict(line.split() for line in Path("/proc/vmstat").read_text().splitlines())
    return {k: int(table.get(k, 0)) for k in VMSTAT_KEYS}


def snapshot() -> dict:
    return {"evicted_ms": evicted_ms(), **psi_memory(), **vmstat()}


class EvictionSampler(threading.Thread):
    """evicted_ms every 10 ms; an episode is a run of samples where it grows (ends after 3 quiet samples)."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.stopping = False
        self.episodes: list = []  # (start epoch s, ms)

    def run(self) -> None:
        prev, grown, quiet, start = evicted_ms(), 0, 0, 0.0
        while not self.stopping:
            time.sleep(0.01)
            now = evicted_ms()
            if now > prev:
                start = start if grown else time.time()
                grown, quiet = grown + now - prev, 0
            elif grown:
                quiet += 1
                if quiet >= 3:
                    self.episodes.append((round(start, 3), grown))
                    grown = 0
            prev = now
        if grown:
            self.episodes.append((round(start, 3), grown))


class PressureReader:
    """Reads the pressure files through the page cache in a loop, in an own scope with a run-time limit."""

    def __init__(self, files: list, max_seconds: int) -> None:
        self.files, self.max_seconds, self.proc = files, max_seconds, None

    def start(self) -> None:
        loop = f"while true; do cat {' '.join(shlex.quote(f) for f in self.files)} > /dev/null; done"
        cmd = ["systemd-run", "--user", "--scope", "-q", "-p", f"RuntimeMaxSec={self.max_seconds}", "bash", "-c", loop]
        self.proc = subprocess.Popen(cmd, start_new_session=True)

    def stop(self) -> None:
        if self.proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
        self.proc.wait()
        self.proc = None
        drop_cache(self.files)


def drop_cache(files: list) -> None:
    """Drops the page cache of the pressure files (posix_fadvise, no root): a pair without pressure starts clean."""
    for f in files:
        fd = os.open(f, os.O_RDONLY)
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        os.close(fd)


def request(url: str, prompt: str) -> dict:
    return S.parse_completion(S.post(url, "/v1/chat/completions", S.chat_body(prompt, 200)))


def measured(url: str, prompt: str, files: list, pressured: bool, max_seconds: int) -> dict:
    """One request with the samplers around it (and the reader, when pressured); the record of the request."""
    reader = PressureReader(files, max_seconds) if pressured else None
    sampler = EvictionSampler()
    before = snapshot()
    sampler.start()
    if reader:
        reader.start()
    t0 = time.time()
    try:
        result = request(url, prompt)
    finally:
        sampler.stopping = True
        sampler.join()
        if reader:
            reader.stop()
    after = snapshot()
    longest = max((ms for _, ms in sampler.episodes), default=0)
    return {
        **result,
        "pressured": pressured,
        "wall_s": round(time.time() - t0, 1),
        "evicted_ms": after["evicted_ms"] - before["evicted_ms"],
        "longest_episode_ms": longest,
        "episodes": len(sampler.episodes),
        "delta": {k: after[k] - before[k] for k in after if k != "evicted_ms"},
    }


class Series:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args, self.files = args, sorted(glob.glob(args.pressure_files))
        if not self.files:
            raise SystemExit(f"--pressure-files {args.pressure_files!r} matches no file")
        out = Path(args.out or REPO / "work/r8" / args.tag)
        out.mkdir(parents=True, exist_ok=True)
        self.jsonl, self.log = out / f"{args.tag}.jsonl", out / f"{args.tag}.server.log"
        self.url = f"http://127.0.0.1:{args.port}"
        corpus = Path(args.corpus)
        corpus = corpus if corpus.is_absolute() else REPO / corpus
        if not corpus.exists():
            corpus.parent.mkdir(parents=True, exist_ok=True)
            corpus.write_text(make_corpus.build(make_corpus.DEFAULT_COMMIT), encoding="utf-8")
        self.corpus = corpus.read_text(encoding="utf-8")
        self.cfg = S.ServerConfig(
            root=REPO, model=args.model, drafter=args.drafter, ctx=args.ctx, port=args.port, base_env={"TONA_DIAG": "1"}
        )
        self.arm = S.Arm("r8", args.build)
        self.proc = None

    def start(self) -> None:
        S.check_port(self.args.port)
        self.proc = S.start_server(self.cfg, self.arm, self.log)
        S.wait_ready(self.proc, self.args.port, self.cfg.ready_timeout)

    def stop(self) -> None:
        S.stop_server(self.proc)
        self.proc = None

    def record(self, row: dict) -> None:
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        print(
            json.dumps({k: row.get(k) for k in ("pair", "target", "pressured", "decode", "evicted_ms", "error")}),
            flush=True,
        )

    def pair(self, i: int) -> None:
        pressured = i % 2 == 1
        for target in TARGETS:
            salt = f"{self.args.tag}-{i}"
            prompt = S.prompt_for(self.url, self.corpus, target)
            prompt = f"Session {salt}.\n" + prompt  # a new prefix per pair: no prompt cache hit between the pairs
            try:
                row = measured(self.url, prompt, self.files, pressured, self.args.press_max_s)
            except Exception as e:  # a failed pass (barrier abort) ends the request with an error: record it, restart
                self.record({"pair": i, "target": target, "pressured": pressured, "error": str(e)})
                self.stop()
                self.start()
                continue
            self.record({"pair": i, "target": target, **row})

    def run(self) -> None:
        self.start()
        try:
            for i in range(self.args.pairs):
                self.pair(i)
        finally:
            self.stop()


def summarize(path: Path) -> int:
    """The summary against the usual expectation; 0 when every point holds."""
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    pressured = [r for r in rows if r["pressured"]]
    errors = [r for r in rows if "error" in r]
    ok = [r for r in pressured if "error" not in r]
    slow = [r for r in ok if r["decode"] < SLOW_TOK_S]
    evicted = sum(r["evicted_ms"] for r in ok)
    longest = max((r["longest_episode_ms"] for r in ok), default=0)
    checks = {
        "no failed request": not errors,
        f"slow decodes (below {SLOW_TOK_S:.0f} tok/s) at most 3 of {len(pressured)} pressured": len(slow) <= 3,
        f"evicted time {evicted / 1000:.1f} s below 5 s in the pressured requests": evicted < 5000,
        f"longest eviction episode {longest / 1000:.1f} s below 3 s": longest < 3000,
    }
    for name, passed in checks.items():
        print(("ok    " if passed else "MISS  ") + name)
    return 0 if all(checks.values()) else 1


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", required=True, help="build directory under work/build, or a path")
    ap.add_argument(
        "--pressure-files", required=True, help="glob of big files the server does not use (the page cache load)"
    )
    ap.add_argument("--pairs", type=int, default=10, help="pairs of 30K and 130K requests (10 pairs: 20 requests)")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--ctx", type=int, default=140000)
    ap.add_argument("--tag", default="r8")
    ap.add_argument("--out", default="", help="directory of the result files (default work/r8/<tag>)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--drafter", default=DRAFTER)
    ap.add_argument("--corpus", default="work/corpus/code.txt", help="made by scripts/make_corpus.py if missing")
    ap.add_argument("--press-max-s", type=int, default=3850, help="RuntimeMaxSec of every reader scope")
    args = ap.parse_args()
    if args.port in FORBIDDEN_PORTS:
        ap.error(f"port {args.port} belongs to the running service")
    return args


def main() -> int:
    args = parse_args()
    series = Series(args)
    series.run()
    return summarize(series.jsonl)


if __name__ == "__main__":
    sys.exit(main())
