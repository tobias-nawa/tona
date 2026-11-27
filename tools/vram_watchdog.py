"""Watchdog of the VRAM balloon (research/62 stop rule 6): a process of its own, so that a hung balloon or a dead
job cannot keep the memory.

    python3 tools/vram_watchdog.py --heartbeat FILE --parent PID --kill PID [--kill PID ...] [--watch PID]

It kills the PIDs given with --kill (SIGKILL, by PID only, never by name) when the heartbeat file, which the balloon
rewrites every second, is older than 5 s, or when the --parent process (the job that started everything) is gone. It
ends quietly when the --watch process (the balloon) has ended on its own. The first 20 s count as start-up (the balloon
loads the HIP runtime before its first beat). Standard library only.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import signal
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vram_guard as vg

START_GRACE_S = 20.0


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def heartbeat_age(path: Path) -> float | None:
    try:
        return time.time() - path.stat().st_mtime
    except OSError:
        return None


def kill_all(pids: list[int]) -> None:
    for pid in pids:
        with contextlib.suppress(ProcessLookupError):
            os.kill(pid, signal.SIGKILL)


def watch(args: argparse.Namespace) -> int:
    started = time.monotonic()
    while True:
        if args.watch is not None and not alive(args.watch):
            return 0
        age = heartbeat_age(args.heartbeat)
        in_grace = age is None and time.monotonic() - started < START_GRACE_S
        if not in_grace and vg.watchdog_kills(age, alive(args.parent)):
            kill_all(args.kill)
            print(f"watchdog: killed {args.kill} (heartbeat age {age})", file=sys.stderr, flush=True)
            return 1
        time.sleep(1.0)


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--heartbeat", required=True, type=Path)
    ap.add_argument("--parent", required=True, type=int)
    ap.add_argument("--kill", required=True, type=int, action="append")
    ap.add_argument("--watch", type=int)
    return watch(ap.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
