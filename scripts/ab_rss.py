"""Memory limit of the server under scripts/ab.py where systemd-run is missing (macOS).

Linux runs the server in a systemd scope with MemoryMax and MemorySwapMax=0 (ab_server.server_command). macOS has
neither systemd nor cgroups, so a thread polls the resident size of the started PID and ends that PID (SIGTERM, SIGKILL
after a grace period) once it exceeds the limit. The resident size counts the resident pages of the mapped model file
too, as the scope's MemoryMax does; there is no swap limit (macOS swaps the rest of the system, not just the server).
"""

import re
import signal
import subprocess
import threading
import time

UNITS = {"": 1, "K": 1 << 10, "M": 1 << 20, "G": 1 << 30, "T": 1 << 40}


def parse_limit(text: str) -> int:
    """Bytes of a systemd size ("54G", "512M", "1048576"; units of 1024)."""
    m = re.fullmatch(r"\s*(\d+)\s*([KMGT]?)\s*", text.upper())
    if not m:
        raise ValueError(f"memory limit {text!r} is not a size like 54G")
    return int(m.group(1)) * UNITS[m.group(2)]


def rss_bytes(pid: int) -> int | None:
    """Resident size of pid (ps, KiB), None once the process is gone."""
    out = subprocess.run(["ps", "-o", "rss=", "-p", str(pid)], capture_output=True, text=True, check=False)
    text = out.stdout.strip()
    return int(text) * 1024 if out.returncode == 0 and text.isdigit() else None


class RssGuard:
    """Ends proc by its PID when its resident size exceeds limit bytes; `tripped` holds the size it was ended at."""

    def __init__(self, proc: subprocess.Popen, limit: int, log_path=None, period: float = 1.0, grace: float = 10.0):
        self.proc, self.limit, self.log_path = proc, limit, log_path
        self.period, self.grace = period, grace
        self.tripped: int | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> "RssGuard":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive() and self._thread is not threading.current_thread():
            self._thread.join(self.period + self.grace + 5)

    def _run(self) -> None:
        while not self._stop.is_set() and self.proc.poll() is None:
            rss = rss_bytes(self.proc.pid)
            if rss is not None and rss > self.limit:
                self._end(rss)
                return
            self._stop.wait(self.period)

    def _end(self, rss: int) -> None:
        self.tripped = rss
        if self.log_path:
            with open(self.log_path, "a") as log:
                log.write(f"\nab.py: resident size {rss >> 20} MiB over the limit {self.limit >> 20} MiB, ending PID "
                          f"{self.proc.pid}\n")  # fmt: skip
        self.proc.send_signal(signal.SIGTERM)
        deadline = time.monotonic() + self.grace
        while self.proc.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        if self.proc.poll() is None:
            self.proc.kill()
