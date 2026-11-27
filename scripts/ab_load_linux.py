"""Linux load snapshot for scripts/ab_load.py (same result fields as the macOS one), from /proc and /sys, no root.

Processes: `/proc/<pid>/stat` (ppid, utime + stime). "System" on Linux: a process of another uid, one whose
executable cannot be read (kernel threads, other users) or one under SYSTEM_PATHS (service helpers of the
distribution). Their executables get the pseudo prefix PSEUDO_SYSTEM so `group_of` finds them without a real path.
Our uid's own programs are "other" (a desktop session, a build); the path is only used to group, never logged.

GPU: DRM fdinfo (`drm-client-id`, `drm-engine-gfx` + `drm-engine-compute`, nanoseconds, amdgpu), each client id
once per pid (a client's descriptors repeat the same counters). fdinfo of other users' processes is unreadable
without root, so the GPU share of "system" is a lower bound.

Memory: `pgpgin` (KiB), `pswpin`, `pswpout` (pages) of /proc/vmstat as page-ins, swap-ins and compressions (swap is
zram here, so a swap-out is a compression); level = MemAvailable / MemTotal in percent. `backup` is always False:
no backup service is known on this machine.

Power: on battery if a `BAT*` supply discharges (a desktop has none); `mode` from ACPI platform_profile (low-power 1,
balanced 0, performance 2, as the macOS power modes), None if the file is absent (this machine); `cpu_limit` None
(no unprivileged source); `warning` if an amdgpu hwmon temperature is within WARM_MARGIN_C of its `_crit`.
"""

import os
import re
from pathlib import Path

PROC, SYS = Path("/proc"), Path("/sys")  # tests point these at a fake tree
PSEUDO_SYSTEM = "?system/"
SYSTEM_PATHS = ("/usr/lib/systemd/", "/usr/libexec/", "/usr/sbin/", "/sbin/", PSEUDO_SYSTEM)
# 10 C: amdgpu starts to throttle well before the critical limit (junction 110 C on the reference GPU)
WARM_MARGIN_C = 10
DRM_NODES = ("/dev/dri/",)
GPU_ENGINES = ("drm-engine-gfx", "drm-engine-compute")
PROFILES = {"low-power": 1, "quiet": 1, "cool": 1, "balanced": 0, "balanced-performance": 0, "performance": 2}


def read(path: Path) -> str:
    try:
        return path.read_text()
    except OSError:
        return ""


def parse_stat(text: str, tick: int) -> tuple | None:
    """(ppid, CPU seconds) from a stat line; the comm field may hold spaces and parens, so cut after the last `)`."""
    end = text.rfind(")")
    fields = text[end + 2 :].split()
    if end < 0 or len(fields) < 13:
        return None
    return int(fields[1]), (int(fields[11]) + int(fields[12])) / tick


def executable(pid_dir: Path, uid: int) -> str:
    owner = re.search(r"^Uid:\s+(\d+)", read(pid_dir / "status"), re.M)
    try:
        exe = os.readlink(pid_dir / "exe")
    except OSError:
        return PSEUDO_SYSTEM
    return exe if owner and int(owner[1]) == uid else PSEUDO_SYSTEM + exe


def processes() -> dict:
    """{pid: (ppid, cpu seconds, executable)}"""
    tick, uid, procs = os.sysconf("SC_CLK_TCK"), os.getuid(), {}
    for d in PROC.iterdir():
        if not d.name.isdigit():
            continue
        parsed = parse_stat(read(d / "stat"), tick)
        if parsed:
            procs[int(d.name)] = (*parsed, executable(d, uid))
    return procs


def fdinfo_ns(text: str) -> tuple | None:
    """(client id, GPU nanoseconds) of one fdinfo file, None if it is no DRM client."""
    client = re.search(r"^drm-client-id:\s+(\d+)", text, re.M)
    if not client:
        return None
    busy = sum(int(m[1]) for e in GPU_ENGINES for m in re.finditer(rf"^{e}:\s+(\d+) ns", text, re.M))
    return int(client[1]), busy


def drm_fds(pid_dir: Path) -> list:
    """Descriptors that point at a DRM node. Checking the link first costs far less than reading the fdinfo of
    every descriptor of every process (180 ms against a few ms per snapshot, measured on the reference PC)."""
    try:
        names = os.listdir(pid_dir / "fd")
    except OSError:  # another user's process
        return []
    found = []
    for name in names:
        try:
            if os.readlink(pid_dir / "fd" / name).startswith(DRM_NODES):
                found.append(name)
        except OSError:
            pass  # closed in between
    return found


def gpu_ns(pids) -> dict:
    """{pid: GPU nanoseconds}"""
    used: dict = {}
    for pid in pids:
        clients = {}
        for fd in drm_fds(PROC / str(pid)):
            info = fdinfo_ns(read(PROC / str(pid) / "fdinfo" / fd))
            if info:
                clients[info[0]] = info[1]
        if clients:
            used[pid] = sum(clients.values())
    return used


def memory() -> dict:
    vm = {k: int(v) for k, v in re.findall(r"^(pgpgin|pswpin|pswpout) (\d+)", read(PROC / "vmstat"), re.M)}
    mem = {k: int(v) for k, v in re.findall(r"^(MemTotal|MemAvailable):\s+(\d+)", read(PROC / "meminfo"), re.M)}
    page_kib = os.sysconf("SC_PAGE_SIZE") / 1024
    out = {
        "pageins": vm.get("pgpgin", 0) / 1024,
        "swapins": vm.get("pswpin", 0) * page_kib / 1024,
        "compressions": vm.get("pswpout", 0) * page_kib / 1024,
        "level": None,
        "backup": False,
    }
    if mem.get("MemTotal"):
        out["level"] = round(mem.get("MemAvailable", 0) / mem["MemTotal"] * 100)
    return out


def warm() -> bool:
    for hwmon in (SYS / "class/drm").glob("card*/device/hwmon/hwmon*"):
        for crit in hwmon.glob("temp*_crit"):
            now = read(hwmon / crit.name.replace("_crit", "_input")).strip()
            limit = read(crit).strip()
            if now.isdigit() and limit.isdigit() and int(now) >= int(limit) - WARM_MARGIN_C * 1000:
                return True
    return False


def power() -> dict:
    battery = any(read(b / "status").strip() == "Discharging" for b in (SYS / "class/power_supply").glob("BAT*"))
    profile = read(SYS / "firmware/acpi/platform_profile").strip()
    return {"warning": warm(), "cpu_limit": None, "battery": battery, "mode": PROFILES.get(profile)}


def snapshot() -> dict:
    procs = processes()
    return {"procs": procs, "gpu": gpu_ns(procs), "mem": memory(), "power": power()}
