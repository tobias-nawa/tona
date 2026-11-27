#!/usr/bin/env python3
"""Read-only logger of the GPU memory counters around a run (research/58-vram-instant-giveback.md, experiment E0).

    python3 tools/vram_watch.py --out work/vram-watch/run1 -- COMMAND [ARGS...]
    python3 tools/vram_watch.py --out work/vram-watch/run1 --seconds 120     # no command: just watch

Samples `mem_info_gtt_used`, `mem_info_gtt_total`, `mem_info_vram_used` and `mem_info_vram_total` of the discrete GPU
(the card with the most VRAM) ten times a second into OUT.csv, follows `journalctl -k` for the messages that show a
failed allocation or an eviction into OUT.kernel.log, and prints a summary when the command ends: the lowest GTT
headroom (total - used) and when it happened, the peak VRAM, and how many kernel lines matched.

It reads sysfs and the kernel log only; it never allocates GPU memory and needs no root. The kernel lines are scrubbed
of process names (summaries and logs hold sums and kernel text only, never a program name).

Standard library only, Python 3.9+, Linux.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

COUNTERS = ("mem_info_gtt_used", "mem_info_gtt_total", "mem_info_vram_used", "mem_info_vram_total")

# Messages that show a failed allocation or an eviction (research/58 section 2): the CS error, the TTM eviction
# errors, the VM validation error, ring timeouts and resets, which would be a different failure.
KERNEL_PATTERN = re.compile(
    r"Not enough memory for command submission|Failed to find memory space|Buffer eviction failed|"
    r"amdgpu_vm_validate\(\) failed|ring .* timeout|GPU reset|evict",
    re.IGNORECASE,
)
# "VM memory stats for proc NAME(PID) task NAME:cs0(PID)", "Process NAME pid N": a program name is never written down
NAME_PATTERN = re.compile(r"\b(proc|process|task|comm)\b[ =:]+\S+", re.IGNORECASE)


def scrub(line: str) -> str:
    """The kernel line without the names of programs."""
    return NAME_PATTERN.sub(lambda m: m.group(1) + " <name>", line)


def pick_card(root: Path = Path("/sys/class/drm")) -> Path | None:
    """The device directory of the card with the most VRAM (the discrete GPU; the iGPU has 2 GiB)."""
    best, size = None, -1
    for dev in sorted(root.glob("card[0-9]*/device")):
        try:
            total = int((dev / "mem_info_vram_total").read_text())
        except (OSError, ValueError):
            continue
        if total > size:
            best, size = dev, total
    return best


def read_counters(dev: Path) -> dict[str, int]:
    out = {}
    for name in COUNTERS:
        try:
            out[name] = int((dev / name).read_text())
        except (OSError, ValueError):
            out[name] = -1
    return out


@dataclass
class Summary:
    samples: int = 0
    min_headroom: int | None = None  # gtt_total - gtt_used, smallest seen
    min_headroom_t: float = 0.0
    max_gtt_used: int = 0
    max_vram_used: int = 0
    vram_total: int = 0
    gtt_total: int = 0
    kernel_lines: int = 0

    def add(self, t: float, c: dict[str, int]) -> None:
        self.samples += 1
        self.gtt_total = max(self.gtt_total, c["mem_info_gtt_total"])
        self.vram_total = max(self.vram_total, c["mem_info_vram_total"])
        self.max_gtt_used = max(self.max_gtt_used, c["mem_info_gtt_used"])
        self.max_vram_used = max(self.max_vram_used, c["mem_info_vram_used"])
        if c["mem_info_gtt_total"] >= 0 and c["mem_info_gtt_used"] >= 0:
            head = c["mem_info_gtt_total"] - c["mem_info_gtt_used"]
            if self.min_headroom is None or head < self.min_headroom:
                self.min_headroom, self.min_headroom_t = head, t

    def text(self) -> str:
        gib = 1 << 30
        head = (
            "n/a" if self.min_headroom is None else f"{self.min_headroom / gib:.2f} GiB at {self.min_headroom_t:.1f} s"
        )
        return (
            f"samples {self.samples}; GTT total {self.gtt_total / gib:.2f} GiB, "
            f"peak used {self.max_gtt_used / gib:.2f} GiB, lowest headroom {head}; "
            f"VRAM total {self.vram_total / gib:.2f} GiB, peak used {self.max_vram_used / gib:.2f} GiB; "
            f"{self.kernel_lines} matching kernel lines"
        )


def follow_kernel(out: Path, summary: Summary, stop: threading.Event) -> subprocess.Popen:
    """Starts `journalctl -k -f` and writes the matching, scrubbed lines; returns the process to terminate."""
    proc = subprocess.Popen(
        ["journalctl", "-k", "-f", "-n", "0", "-o", "short-precise", "--no-pager"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
    )

    def pump() -> None:
        with out.open("w") as f:
            for line in proc.stdout:
                if KERNEL_PATTERN.search(line):
                    summary.kernel_lines += 1
                    f.write(scrub(line.rstrip("\n")) + "\n")
                    f.flush()

    threading.Thread(target=pump, daemon=True).start()
    return proc


def watch(dev: Path, out: Path, hz: float, child: subprocess.Popen | None, seconds: float) -> Summary:
    summary = Summary()
    t0 = time.monotonic()
    period = 1.0 / hz
    with out.with_suffix(".csv").open("w") as csv:
        csv.write("t_s,gtt_used,gtt_total,vram_used,vram_total\n")
        nxt = t0
        while True:
            t = time.monotonic() - t0
            c = read_counters(dev)
            summary.add(t, c)
            csv.write(f"{t:.3f},{c[COUNTERS[0]]},{c[COUNTERS[1]]},{c[COUNTERS[2]]},{c[COUNTERS[3]]}\n")
            if child is not None and child.poll() is not None:
                break
            if child is None and t >= seconds:
                break
            nxt += period
            time.sleep(max(0.0, nxt - time.monotonic()))
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="path prefix: writes OUT.csv and OUT.kernel.log")
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--seconds", type=float, default=60.0, help="without a command: how long to watch")
    ap.add_argument("--card", help="device directory (default: the card with the most VRAM)")
    ap.add_argument("command", nargs="*", help="after --: the command to wrap")
    args = ap.parse_args()
    dev = Path(args.card) if args.card else pick_card()
    if dev is None:
        print("vram_watch: no card with mem_info_vram_total found", file=sys.stderr)
        return 2
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary = Summary()
    jproc = follow_kernel(out.with_suffix(".kernel.log"), summary, threading.Event())
    child = subprocess.Popen(args.command) if args.command else None
    try:
        s = watch(dev, out, args.hz, child, args.seconds)
    finally:
        jproc.terminate()
    s.kernel_lines = summary.kernel_lines
    print("vram_watch:", dev, s.text())
    return child.returncode if child is not None else 0


if __name__ == "__main__":
    sys.exit(main())
