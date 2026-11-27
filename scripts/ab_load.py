"""Load of every server start of scripts/ab.py, split into ours, the system's and other programs' (macOS, Linux).

A desktop machine is never idle: a backup, an index run or an update can take the CPU, the GPU or the page cache in
the middle of an arm. Every server start therefore gets a record of what ran beside it, so that a slow start can be
traced afterwards instead of guessed. Sums only, never program names (AGENTS.md rule 2): a process is "ours" (the
server and the measuring process with their children), "system" (an executable under the OS paths) or "other".

Sources without root: CPU time per process (`ps`), GPU time per process (`ioreg` AppUsage of the GPU's user
clients, nanoseconds), page-ins, swap-ins and compressions (`vm_stat`), `kern.memorystatus_level` and whether a
Time Machine backup runs (`tmutil status`). GPU clock and temperature need root (`powermetrics`) and are not read.
The Linux sources (/proc, /sys, DRM fdinfo) are in scripts/ab_load_linux.py.
"""

import os
import plistlib
import re
import subprocess
import sys
import threading
import time

import ab_load_linux

# s; one sample costs about 60 ms of one core on macOS and 33 ms on the Linux reference PC (mean of 100 calls, 530
# processes, 2026-11-14)
INTERVAL = 2.0
GROUPS = ("own", "system", "other")
TRAFFIC = ("pageins", "swapins", "compressions")  # cumulative MiB in a snapshot, per start in the summary
# executables under these paths count as the system's
SYSTEM_PATHS = {
    "darwin": ("/System/", "/usr/", "/sbin/", "/bin/", "/Library/Apple/", "/private/"),
    "linux": ab_load_linux.SYSTEM_PATHS,
}
AUTO = ("darwin",)  # the Linux reference changes only with a measurement there


def enabled(mode: str) -> bool:
    if mode == "on" and sys.platform not in SNAPSHOTS:
        raise SystemExit(f"--load-log on: no load snapshot for {sys.platform} in scripts/ab_load.py")
    return mode == "on" or (mode == "auto" and sys.platform in AUTO)


def run(cmd: list) -> str:
    return subprocess.run(cmd, capture_output=True, text=True, check=False).stdout


def cpu_seconds(text: str) -> float:
    """`ps` time: [[dd-]hh:]mm:ss.cc"""
    days, _, rest = text.rpartition("-")
    total = 0.0
    for part in rest.split(":"):
        total = total * 60 + float(part)
    return total + (int(days) * 86400 if days else 0)


def processes() -> dict:
    """{pid: (ppid, cpu seconds, executable)}"""
    procs = {}
    for line in run(["ps", "-A", "-o", "pid=,ppid=,time=,comm="]).splitlines():
        parts = line.split(None, 3)
        if len(parts) == 4:
            procs[int(parts[0])] = (int(parts[1]), cpu_seconds(parts[2]), parts[3])
    return procs


def gpu_ns() -> dict:
    """{pid: GPU nanoseconds} summed over the user clients of the GPU"""
    try:
        clients = plistlib.loads(
            subprocess.run(
                ["ioreg", "-a", "-r", "-c", "AGXDeviceUserClient", "-d", "1"], capture_output=True, check=False
            ).stdout
        )
    except (plistlib.InvalidFileException, ValueError):
        return {}
    used: dict = {}
    for c in clients:
        m = re.match(r"pid (\d+)", str(c.get("IOUserClientCreator", "")))
        if m:
            ns = sum(int(u.get("accumulatedGPUTime", 0)) for u in c.get("AppUsage", []))
            used[int(m[1])] = used.get(int(m[1]), 0) + ns
    return used


def backup_running() -> bool:
    """A Time Machine backup runs (macOS; elsewhere: never). It pages in and compresses gigabytes beside the server
    (2026-11-15-mac-coder-arena-ab.md: +2.35 GiB compressor during one)."""
    return sys.platform == "darwin" and bool(re.search(r"\bRunning = 1;", run(["tmutil", "status"])))


def wait_backup(limit_s: float = 3 * 3600, every_s: float = 30) -> None:
    """Waits until no backup runs, at most limit_s, then raises: a measurement must not start beside one."""
    end, told = time.monotonic() + limit_s, False
    while backup_running():
        if time.monotonic() > end:
            raise RuntimeError(f"a system backup still runs after {limit_s / 60:.0f} min")
        if not told:
            print("a system backup runs: waiting for its end", file=sys.stderr, flush=True)
            told = True
        time.sleep(every_s)


def memory() -> dict:
    page, vm = 16384, {}
    names = {"Pageins": "pageins", "Swapins": "swapins", "Compressions": "compressions"}
    for line in run(["vm_stat"]).splitlines():
        key, _, value = line.partition(":")
        m = re.search(r"page size of (\d+) bytes", line)
        if m:
            page = int(m[1])
        elif key.strip('"') in names:
            vm[names[key.strip('"')]] = int(value.strip().rstrip(".")) * page / 2**20
    level = run(["sysctl", "-n", "kern.memorystatus_level"]).strip()
    vm["level"] = int(level) if level.isdigit() else None
    vm["backup"] = bool(re.search(r"\bRunning = 1;", run(["tmutil", "status"])))
    return vm


def power() -> dict:
    """Thermal or performance warning, CPU speed limit (pmset prints them only once the system records one), on
    battery, power mode (0 automatic, 1 low power, 2 high power): a throttled GPU waits like a starved one."""
    therm = run(["pmset", "-g", "therm"])
    limit = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", therm)
    mode = re.search(r"^\s*(?:powermode|lowpowermode)\s+(\d+)", run(["pmset", "-g"]), re.M)
    return {
        "warning": "No thermal warning" not in therm or "No performance warning" not in therm,
        "cpu_limit": int(limit[1]) if limit else None,
        "battery": "Battery Power" in run(["pmset", "-g", "batt"]),
        "mode": int(mode[1]) if mode else None,
    }


def darwin_snapshot() -> dict:
    return {"procs": processes(), "gpu": gpu_ns(), "mem": memory(), "power": power()}


# One function per platform, all with the same result:
#   procs {pid: (ppid, CPU seconds, executable path)}, gpu {pid: GPU nanoseconds},
#   mem {"pageins", "swapins", "compressions": cumulative MiB; "level": free-memory level 0-100 or None;
#        "backup": a system backup runs},
#   power {"warning": a thermal or performance warning is recorded, "cpu_limit": CPU speed limit percent or None,
#          "battery": runs on battery, "mode": power mode or None}
SNAPSHOTS = {"darwin": darwin_snapshot, "linux": ab_load_linux.snapshot}


def own_pids(procs: dict, roots: set) -> set:
    own, grew = set(roots), True
    while grew:
        more = {pid for pid, (ppid, _, _) in procs.items() if ppid in own and pid not in own}
        own |= more
        grew = bool(more)
    return own


def group_of(pid: int, procs: dict, own: set) -> str:
    if pid in own:
        return "own"
    exe = procs.get(pid, (0, 0, ""))[2]
    return "system" if exe.startswith(SYSTEM_PATHS.get(sys.platform, ("/",))) else "other"


def deltas(before: dict, after: dict, procs: dict, own: set) -> dict:
    """Per group: what the processes present in both samples used in between (a process that ends in between is
    lost for that interval)."""
    out = dict.fromkeys(GROUPS, 0.0)
    for pid, value in after.items():
        if pid in before:
            out[group_of(pid, procs, own)] += max(0.0, value - before[pid])
    return out


def sample(roots: set) -> dict:
    snap = SNAPSHOTS[sys.platform]()
    procs = snap["procs"]
    return snap | {"t": time.monotonic(), "own": own_pids(procs, roots), "cpu": {p: v[1] for p, v in procs.items()}}


def interval(a: dict, b: dict) -> dict:
    dt = b["t"] - a["t"]
    cpu = deltas(a["cpu"], b["cpu"], b["procs"], b["own"])
    gpu = deltas(a["gpu"], b["gpu"], b["procs"], b["own"])
    row = {f"cpu_{g}": cpu[g] / dt * 100 for g in GROUPS}  # percent of one core
    row |= {f"gpu_{g}": gpu[g] / 1e9 / dt * 100 for g in GROUPS}  # percent of the GPU's time
    row |= {k: b["mem"][k] - a["mem"][k] for k in TRAFFIC if k in a["mem"] and k in b["mem"]}
    row |= {"level": b["mem"]["level"], "backup": b["mem"]["backup"], "seconds": dt}
    return row | {f"power_{k}": v for k, v in b.get("power", {}).items()}


class LoadLog:
    """Samples every INTERVAL s from start() to stop() in a thread; stop() returns the summary of the start, with the
    seconds from the end of the previous start (idle GPU: a clock ramp shows in the first request, PC finding of
    2026-11-12-prefill-bimodal.md) and from the server start to ready."""

    last_stop: float | None = None  # monotonic time the previous start's log ended

    def __init__(self):
        # the server is a child of this process (own session, same parent), so are the bench clients and samplers
        self.roots, self.rows, self.done = {os.getpid()}, [], threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.t0 = time.monotonic()
        self.idle = None if LoadLog.last_stop is None else self.t0 - LoadLog.last_stop
        self.ready_s = None

    def ready(self) -> None:
        self.ready_s = time.monotonic() - self.t0

    def start(self) -> "LoadLog":
        self.thread.start()
        return self

    def loop(self) -> None:
        last = sample(self.roots)
        while not self.done.wait(INTERVAL):
            now = sample(self.roots)
            self.rows.append(interval(last, now))
            last = now

    def stop(self) -> dict:
        self.done.set()
        self.thread.join()
        LoadLog.last_stop = time.monotonic()
        times = {"idle_before_s": self.idle, "ready_s": self.ready_s}
        return summary(self.rows) | {k: None if v is None else round(v, 1) for k, v in times.items()}


def share(rows: list, key: str) -> float:
    """Share of the time in which the row flag `key` was set."""
    return round(sum(r["seconds"] for r in rows if r.get(key)) / sum(r["seconds"] for r in rows), 2)


def lowest(rows: list, key: str):
    values = [r[key] for r in rows if r.get(key) is not None]
    return min(values) if values else None


def summary(rows: list) -> dict:
    """Means of the percentages, sums of the memory traffic (MiB), the lowest memory level and CPU limit, shares of
    the time with a backup, a thermal warning or on battery, the power modes seen."""
    if not rows:
        return {}
    total = sum(r["seconds"] for r in rows)
    out = {k: round(sum(r[k] * r["seconds"] for r in rows) / total, 1) for k in rows[0] if k.startswith(("cpu", "gpu"))}
    out |= {k: round(sum(r.get(k, 0) for r in rows), 1) for k in TRAFFIC}
    out |= {"level_min": lowest(rows, "level"), "backup_share": share(rows, "backup")}
    out |= {"warning_share": share(rows, "power_warning"), "battery_share": share(rows, "power_battery")}
    out["cpu_limit_min"] = lowest(rows, "power_cpu_limit")
    modes = sorted({r["power_mode"] for r in rows if r.get("power_mode") is not None})
    out["power_mode"] = "/".join(map(str, modes)) or None
    out["seconds"] = round(total)
    return out


COLUMNS = [
    ("cpu_own", "CPU ours"), ("cpu_system", "CPU system"), ("cpu_other", "CPU other"),
    ("gpu_own", "GPU ours"), ("gpu_system", "GPU system"), ("gpu_other", "GPU other"),
    ("pageins", "page-ins MiB"), ("swapins", "swap-ins MiB"), ("compressions", "compressed MiB"),
    ("level_min", "memory level min"), ("backup_share", "backup"), ("warning_share", "thermal warning"),
    ("cpu_limit_min", "CPU limit min"), ("battery_share", "on battery"), ("power_mode", "power mode"),
    ("idle_before_s", "idle before s"), ("ready_s", "to ready s"),
]  # fmt: skip


def load_rows(starts: list) -> list:
    """Markdown table, one row per server start (discarded ones included; numbers as in the server logs)."""
    if not starts:
        return []
    head = "| Start | Arm | " + " | ".join(label for _, label in COLUMNS) + " |"
    rows = [head, "|" + "---|" * (len(COLUMNS) + 2)]
    for s in starts:
        cells = ["-" if s.get(k) is None else str(s[k]) for k, _ in COLUMNS]
        rows.append(f"| {s['start']} | {s['arm']} | " + " | ".join(cells) + " |")
    return rows
