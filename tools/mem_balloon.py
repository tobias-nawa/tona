"""Memory balloon for the RAM governor test (docs/architecture/expert-tiers.md section 5): anonymous memory that
grows at a set rate to a target, holds, and is released, in cycles, like a build or containers started next to the
server. It displaces other programs' memory on purpose: run it only with the maintainer's consent.

Safety, because the machine may stop responding: a hard time limit inside the process (default 45 min, also an
alarm signal), and an emergency brake that releases everything and ends at once when macOS reports critical memory
pressure or the swap in use grows more than --swap-brake-gb over its level at the start. One line per second goes to
the log file as it happens.

Usage: python3 tools/mem_balloon.py LOG [--rates 0.1,1,3] [--targets 10,20] [--phase-s 600] [--hold-s 20]
"""

import argparse
import mmap
import os
import re
import signal
import subprocess
import sys
import time

CHUNK = 64 << 20  # bytes per mapping: released one by one, so a release is immediate


def sysctl(name: str) -> str:
    return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=False).stdout.strip()


def pressure_level() -> int:
    """1 normal, 2 warn, 4 critical (kern.memorystatus_vm_pressure_level); Linux: 0."""
    if sys.platform != "darwin":
        return 0
    try:
        return int(sysctl("kern.memorystatus_vm_pressure_level"))
    except ValueError:
        return 0


def swap_used() -> float:
    """Bytes of swap in use."""
    if sys.platform == "darwin":
        m = re.search(r"used = ([\d.]+)M", sysctl("vm.swapusage"))
        return float(m.group(1)) * 2**20 if m else 0.0
    with open("/proc/meminfo") as f:
        v = {line.split(":")[0]: int(line.split()[1]) for line in f}
    return (v.get("SwapTotal", 0) - v.get("SwapFree", 0)) * 1024.0


class Balloon:
    def __init__(self, log, args):
        self.log, self.args = log, args
        self.maps: list[mmap.mmap] = []
        self.fill = os.urandom(CHUNK)  # incompressible: the compressor must not absorb the balloon
        self.t0 = time.time()
        self.swap0 = swap_used()
        self.last_line = 0.0
        self.state = "start"

    def held(self) -> int:
        return len(self.maps) * CHUNK

    def release(self) -> None:
        while self.maps:
            self.maps.pop().close()

    def check(self) -> None:
        """Brake and time limit; a line per second."""
        now = time.time()
        level, swap = pressure_level(), swap_used()
        if now - self.last_line >= 1.0:
            self.last_line = now
            self.log.write(
                f"{now - self.t0:8.1f} s  {self.state:<12} held {self.held() / 1e9:5.1f} GB  "
                f"pressure {level}  swap +{(swap - self.swap0) / 1e9:5.2f} GB\n"
            )
        if level >= 4:
            self.stop("critical memory pressure")
        if swap - self.swap0 > self.args.swap_brake_gb * 1e9:
            self.stop(f"swap grew more than {self.args.swap_brake_gb} GB")
        if now - self.t0 > self.args.limit_s:
            self.stop("time limit")

    def stop(self, why: str) -> None:
        self.release()
        self.log.write(f"{time.time() - self.t0:8.1f} s  END: {why}, released\n")
        sys.exit(0)

    def wait(self, seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            self.check()
            time.sleep(0.25)

    def grow(self, target_gb: float, rate_gbs: float) -> None:
        self.state = f"grow {rate_gbs:g}GB/s"
        start, have = time.time(), self.held()
        while self.held() < target_gb * 1e9:
            due = have + (time.time() - start) * rate_gbs * 1e9  # bytes the rate allows by now
            while self.held() < min(due, target_gb * 1e9):
                m = mmap.mmap(-1, CHUNK)
                m.write(self.fill)  # touches every page
                self.maps.append(m)
            self.check()
            time.sleep(0.05)

    def cycle(self, target_gb: float, rate_gbs: float) -> None:
        self.log.write(f"{time.time() - self.t0:8.1f} s  cycle to {target_gb:g} GB at {rate_gbs:g} GB/s\n")
        self.grow(target_gb, rate_gbs)
        self.state = "hold"
        self.wait(self.args.hold_s)
        self.release()
        self.state = "released"
        self.wait(self.args.rest_s)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("log")
    ap.add_argument("--rates", default="0.1,1,3", help="GB/s, one phase each")
    ap.add_argument("--targets", default="10,20", help="GB, alternating within a phase")
    ap.add_argument("--phase-s", type=float, default=600)
    ap.add_argument("--hold-s", type=float, default=20)
    ap.add_argument("--rest-s", type=float, default=30, help="pause after each release")
    ap.add_argument("--limit-s", type=float, default=45 * 60, help="hard end, whatever the phases")
    ap.add_argument("--swap-brake-gb", type=float, default=2.0)
    args = ap.parse_args()
    signal.signal(signal.SIGALRM, lambda *_: os._exit(3))  # the last resort: the kernel frees everything
    signal.alarm(int(args.limit_s) + 60)
    with open(args.log, "a", buffering=1) as log:
        b = Balloon(log, args)
        signal.signal(signal.SIGTERM, lambda *_: b.stop("SIGTERM"))
        signal.signal(signal.SIGINT, lambda *_: b.stop("SIGINT"))
        targets = [float(t) for t in args.targets.split(",")]
        for rate in (float(r) for r in args.rates.split(",")):
            end, i = time.time() + args.phase_s, 0
            while time.time() < end:
                b.cycle(targets[i % len(targets)], rate)
                i += 1
        b.stop("all phases done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
