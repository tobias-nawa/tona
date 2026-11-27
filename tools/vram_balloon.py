"""A polite second GPU program: takes VRAM in small steps, holds it, gives it back (level 0 of research/62).

    python3 tools/vram_balloon.py --run-dir DIR [--hold-s 20] [--pause-s 2]

It allocates on the discrete GPU with hipMalloc through ctypes, 64 MiB at a time, and reads the free VRAM of that card
from sysfs before EVERY allocation (tools/vram_guard.py): it never takes the card below 1.5 GiB free. It ends at the
first stop rule: a file DIR/STOP, a kernel log line (GPU reset, ring timeout, failed command submission), the time cap,
a failed allocation, the step cap. On SIGTERM or any stop it frees everything. It writes DIR/balloon.heartbeat every
second (the watchdog of tools/vram_level0.py kills it if that stops) and one JSON line per event to DIR/balloon.jsonl
with sums and times only. No memset and no kernel: it adds no GPU work.

Run it only through the level-0 job in the exclusive queue: this tool is the one that must never take the desktop's
memory.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import vram_guard as vg


class Hip:
    """The three HIP calls the balloon needs. The device is checked to be the card whose sysfs counters are read."""

    def __init__(self, dev: Path) -> None:
        root = os.environ.get("ROCM_PATH", "")
        lib = Path(root) / "lib" / "libamdhip64.so" if root else Path("libamdhip64.so")
        self.lib = ctypes.CDLL(str(lib))
        if self.lib.hipInit(0) != 0 or self.lib.hipSetDevice(0) != 0:
            raise RuntimeError("hipInit/hipSetDevice failed")
        buf = ctypes.create_string_buffer(32)
        if self.lib.hipDeviceGetPCIBusId(buf, 32, 0) != 0:
            raise RuntimeError("hipDeviceGetPCIBusId failed")
        if not vg.same_card(buf.value.decode(), dev):
            raise RuntimeError(f"HIP device 0 is {buf.value.decode()}, the guarded card is {vg.bus_id_of(dev)}")

    def alloc(self, size: int) -> int | None:
        ptr = ctypes.c_void_p()
        if self.lib.hipMalloc(ctypes.byref(ptr), ctypes.c_size_t(size)) != 0:
            return None
        return ptr.value

    def release(self, ptr: int) -> None:
        self.lib.hipFree(ctypes.c_void_p(ptr))


class Heartbeat(threading.Thread):
    """Touches a file every second; stops with the process."""

    def __init__(self, path: Path) -> None:
        super().__init__(daemon=True)
        self.path = path
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            self.path.write_text(f"{time.time():.3f}\n")
            self.stop.wait(1.0)


class KernelLog(threading.Thread):
    """Follows the kernel log from now on; keeps the first line that ends a run."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.abort_line: str | None = None
        self.proc = subprocess.Popen(
            ["journalctl", "-k", "-f", "-n", "0", "-o", "cat"],
            stdout=subprocess.PIPE,
            text=True,
            stderr=subprocess.DEVNULL,
        )

    def run(self) -> None:
        for line in self.proc.stdout:
            if vg.kernel_abort(line) and self.abort_line is None:
                self.abort_line = vg.scrub(line)

    def close(self) -> None:
        self.proc.terminate()


def run_ladder(
    read_free: Callable[[], int | None],
    alloc: Callable[[int], int | None],
    check: Callable[[int | None], str | None],
    log: Callable[..., None],
    pause: Callable[[], None],
    max_steps: int = vg.MAX_STEPS,
) -> list[int]:
    """Takes steps until a stop rule or the cap. Returns the pointers held. Every step reads the free VRAM first and
    allocates only if `vram_guard.may_allocate` says so; `check` carries the other stop rules."""
    held: list[int] = []
    for n in range(max_steps):
        free = read_free()
        reason = check(free)
        if reason is None and not vg.may_allocate(free):
            reason = "no headroom: the next step would leave less than the floor free"
        if reason is not None:
            log("stop", step=n, reason=reason, free_mib=None if free is None else free // vg.MIB)
            return held
        t0 = time.monotonic()
        ptr = alloc(vg.STEP_BYTES)
        ms = (time.monotonic() - t0) * 1000
        if ptr is None:
            log("alloc_failed", step=n, ms=round(ms, 1))
            return held
        held.append(ptr)
        log(
            "step",
            step=n + 1,
            held_mib=len(held) * vg.STEP_BYTES // vg.MIB,
            alloc_ms=round(ms, 1),
            free_mib=free // vg.MIB,
        )
        pause()
    log("stop", step=max_steps, reason="step cap")
    return held


def hold(check: Callable[[], str | None], seconds: float, sleep: Callable[[float], None]) -> str | None:
    """Holds the memory for `seconds`, polling the stop rules every half second."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        reason = check()
        if reason is not None:
            return reason
        sleep(0.5)
    return None


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run-dir", required=True, type=Path)
    ap.add_argument("--hold-s", type=float, default=20.0)
    ap.add_argument("--pause-s", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=vg.MAX_STEPS)
    args = ap.parse_args(argv)
    if not 0 < args.max_steps <= vg.MAX_STEPS:
        raise SystemExit(f"--max-steps must be 1..{vg.MAX_STEPS}")
    dev = vg.pick_card()
    if dev is None:
        raise SystemExit("no discrete GPU found in sysfs")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    hip = Hip(dev)
    started = time.monotonic()
    out = (args.run_dir / "balloon.jsonl").open("a")

    def log(event: str, **kw) -> None:
        out.write(json.dumps({"t": round(time.monotonic() - started, 2), "event": event, **kw}) + "\n")
        out.flush()

    klog = KernelLog()
    klog.start()
    beat = Heartbeat(args.run_dir / "balloon.heartbeat")
    beat.start()
    terminated = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: terminated.set())

    def check(free: int | None = None) -> str | None:
        free = vg.read_free(dev) if free is None else free
        if terminated.is_set():
            return "SIGTERM"
        state = vg.StopState(
            time.monotonic() - started, (args.run_dir / "STOP").exists(), klog.abort_line, False, None, free
        )
        return vg.stop_reason(state)

    held: list[int] = []
    try:
        log("start", free_mib=(vg.read_free(dev) or 0) // vg.MIB, floor_mib=vg.FLOOR_BYTES // vg.MIB)
        held = run_ladder(
            lambda: vg.read_free(dev), hip.alloc, check, log, lambda: time.sleep(args.pause_s), args.max_steps
        )
        reason = hold(check, args.hold_s, time.sleep) if held else None
        log("hold_end", held_mib=len(held) * vg.STEP_BYTES // vg.MIB, reason=reason)
    finally:
        for ptr in held:
            hip.release(ptr)
        log("released", free_mib=(vg.read_free(dev) or 0) // vg.MIB)
        beat.stop.set()
        klog.close()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
