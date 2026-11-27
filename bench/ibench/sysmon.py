"""Samples VRAM, RAM, swap, GPU load, power, temperature and clock in the background."""

import glob
import statistics
import threading
import time
from pathlib import Path


def _read(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except OSError:
        return None


def _int(path: Path) -> int | None:
    v = _read(path)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def find_gpu() -> Path | None:
    """The discrete GPU: the card with the most VRAM (the iGPU has 512 MiB)."""
    best, size = None, 0
    for dev in glob.glob("/sys/class/drm/card[0-9]*/device"):
        total = _int(Path(dev) / "mem_info_vram_total") or 0
        if total > size:
            best, size = Path(dev).resolve(), total
    return best


def meminfo() -> dict[str, int]:
    """Values from /proc/meminfo in KiB."""
    out = {}
    for line in (_read(Path("/proc/meminfo")) or "").splitlines():
        key, _, rest = line.partition(":")
        parts = rest.split()
        if parts and parts[0].isdigit():
            out[key] = int(parts[0])
    return out


class SysMon:
    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.gpu = find_gpu()
        self.hwmon = None
        self.temp = None
        if self.gpu:
            mons = sorted((self.gpu / "hwmon").glob("hwmon*"))
            self.hwmon = mons[0] if mons else None
        if self.hwmon:
            # prefer the junction temperature, else the first sensor
            for label in sorted(self.hwmon.glob("temp*_label")):
                if _read(label) == "junction":
                    self.temp = self.hwmon / label.name.replace("_label", "_input")
            if self.temp is None and (self.hwmon / "temp1_input").exists():
                self.temp = self.hwmon / "temp1_input"
        self.samples: list[tuple[float, dict]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def sample(self) -> dict:
        s = {}
        if self.gpu:
            g = self.gpu
            if (v := _int(g / "mem_info_vram_used")) is not None:
                s["vram_used_mib"] = v / 2**20
            if (v := _int(g / "mem_info_gtt_used")) is not None:
                s["gtt_used_mib"] = v / 2**20
            if (v := _int(g / "gpu_busy_percent")) is not None:
                s["gpu_busy"] = v
        if self.hwmon:
            h = self.hwmon
            p = _int(h / "power1_average") or _int(h / "power1_input")
            if p is not None:
                s["power_w"] = p / 1e6
            if (v := _int(h / "freq1_input")) is not None:
                s["sclk_mhz"] = v / 1e6
        if self.temp and (v := _int(self.temp)) is not None:
            s["temp_c"] = v / 1000
        m = meminfo()
        if "MemAvailable" in m:
            s["ram_avail_mib"] = m["MemAvailable"] / 1024
        if "SwapTotal" in m and "SwapFree" in m:
            s["swap_used_mib"] = (m["SwapTotal"] - m["SwapFree"]) / 1024
        return s

    def _loop(self):
        while not self._stop.is_set():
            self.samples.append((time.perf_counter(), self.sample()))
            self._stop.wait(self.interval)

    def start(self):
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def window(self, t0: float, t1: float) -> dict:
        """Metrics over the samples between t0 and t1; without a sample in the window, the nearest one."""
        win = [s for t, s in list(self.samples) if t0 <= t <= t1]
        if not win and self.samples:
            win = [min(self.samples, key=lambda ts: abs(ts[0] - t1))[1]]

        def col(key):
            return [s[key] for s in win if key in s]

        out = {}
        for key, name, fn in (
            ("vram_used_mib", "vram_peak_mib", max),
            ("gtt_used_mib", "gtt_peak_mib", max),
            ("ram_avail_mib", "ram_avail_min_mib", min),
            ("swap_used_mib", "swap_used_max_mib", max),
            ("gpu_busy", "gpu_busy_mean", statistics.fmean),
            ("power_w", "power_mean_w", statistics.fmean),
            ("temp_c", "temp_max_c", max),
            ("sclk_mhz", "sclk_mean_mhz", statistics.fmean),
        ):
            values = col(key)
            if values:
                out[name] = fn(values)
        out["samples"] = len(win)
        return out
