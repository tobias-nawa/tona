#!/usr/bin/env python3
"""Checks that the machine is quiet before (and after) a measurement.

    python3 tools/quiet_check.py --json > work/quiet-before.json   # before the run
    python3 tools/quiet_check.py --compare work/quiet-before.json  # after the run

Exit code: 0 ok, 1 warning (ask the user to quiet the machine, then check again), 2 do not measure.
With --compare: 0 the load did not change, 3 it changed (mark the run as disturbed or repeat it).

Privacy: the saved output (Markdown, JSON) holds sums and percentages only, never a program name. Which programs
cause the load is told to the user in the conversation by `--ask`; never write that text into a file.

Standard library only, Python 3.9+, Linux, macOS and Windows. It reads, it never changes anything. The thresholds are
below, each with its reason; the sampling window is one second so a busy process shows, a load average (which
lags by a minute) is shown but can only warn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import hw_report as hw
import hw_win32

WINDOW = 1.0  # seconds of sampling: long enough for a busy process to show, short enough to run before every run
GIB = hw.GIB

# (warn, stop) per signal. Reasons: a measurement is meant to carry a spread of about 1 %; anything that takes
# more than a few percent of a shared resource moves the result by about as much, so it warns; the stop values
# are where the machine competes with the run for a unit the engine saturates (cores, GPU, free RAM).
CPU_BUSY_PERCENT = (5, 20)  # of all hardware threads, summed over foreign processes
LOAD_PER_THREAD = (0.25, 0.75)  # lags a minute behind: only ever a warning
GPU_BUSY_PERCENT = (5, 30)  # a desktop compositor is below 5 %
DESKTOP_VRAM_PERCENT = (6, 25)  # of VRAM; the engine plans its VRAM around what is free
RAM_AVAILABLE_PERCENT = (20, 10)  # of RAM; below it the page cache that holds the model is reclaimed
HEAVY_APP = 3  # percent of one core: a process above it is named to the user

# System tools are never named to the user and never asked to be closed (security scanners, indexers, the desktop,
# system services): they run along, their load shows in the sums. Builds are user programs and may be named.
SYSTEM_TOOLS = {
    "baloo_file", "baloo_file_extractor", "tracker-miner-fs-3", "localsearch-3", "localsearch-ext", "tracker-extract-3",
    "mds", "mds_stores", "mdworker", "mdworker_shared", "updatedb", "packagekitd", "clamd", "clamscan", "freshclam",
    "systemd", "systemd-journald", "gnome-shell", "kwin_wayland", "kwin_x11", "plasmashell", "Xorg", "Xwayland",
    "WindowServer", "kernel_task", "syspolicyd", "endpointsecurityd", "dnf", "rpm-ostree",
}  # fmt: skip
OK, WARN, STOP = "ok", "warning", "do not measure"
KERNEL_THREADS = ("kworker", "kswapd", "ksoftirqd", "kcompactd", "migration", "rcu_")  # the user cannot close them
RANK = {OK: 0, WARN: 1, STOP: 2}


def cpu_sample() -> dict:
    """Foreign CPU use over WINDOW: {name: percent of one core}; this process is left out."""
    if not Path("/proc/self/stat").exists():
        rows = [ln.strip().partition(" ") for ln in hw.run(["ps", "-Ao", "pcpu=,comm="]).splitlines()]
        return {Path(c.strip()).name: float(p) for p, _, c in rows if c and float(p or 0) >= 0.5}
    before = hw.cpu_ticks()
    time.sleep(WINDOW)
    hz, me, busy = os.sysconf("SC_CLK_TCK"), os.getpid(), {}
    for pid, (name, ticks) in hw.cpu_ticks().items():
        if pid != me and pid in before and not name.startswith(KERNEL_THREADS):
            busy[name] = busy.get(name, 0) + (ticks - before[pid][1]) / hz / WINDOW * 100
    return {n: round(v, 1) for n, v in busy.items() if v >= 0.5}


def swap_traffic() -> int | None:
    m = re.findall(r"^(?:pswpin|pswpout) (\d+)", hw.read("/proc/vmstat"), re.M)
    return sum(int(x) for x in m) if m else None


def _win32_gpu_cards() -> list:
    return [
        {"busy": g.get("busy_percent"), "vram_gib": g.get("vram_gib"), "used_gib": g.get("vram_used_gib")}
        for g in hw_win32.gpus(hw.run)
        if not g.get("integrated") and (g.get("vram_gib") or g.get("busy_percent") is not None)
    ]


def gpu_cards() -> list:
    """Busy percent and VRAM of discrete GPUs: sysfs (amdgpu), nvidia-smi, ioreg (macOS), Win32."""
    if sys.platform == "win32":
        return _win32_gpu_cards()
    out = []
    for gpu, card in zip(hw.gpus_sysfs(), sorted(Path("/sys/class/drm").glob("card[0-9]*")), strict=False):
        busy = hw.to_int(hw.read(card / "device" / "gpu_busy_percent"))
        if not gpu["integrated"] and gpu["vram_gib"]:
            out.append({"busy": busy, "vram_gib": gpu["vram_gib"], "used_gib": gpu["vram_used_gib"]})
    for row in hw.nvidia_rows():
        out.append({"busy": None, "vram_gib": int(row[1]) / 1024, "used_gib": int(row[2]) / 1024})
    for line in hw.run(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"]).split():
        if line.isdigit() and out and out[-1]["busy"] is None:
            out[-1]["busy"] = int(line)
    util = apple_gpu_busy()
    if util is not None and not out:
        out.append({"busy": util, "vram_gib": None, "used_gib": None})
    return out


def apple_gpu_busy(read=lambda: hw.run(["ioreg", "-r", "-d", "1", "-w0", "-c", "IOAccelerator"]), wait=time.sleep):
    """Busy percent of an Apple GPU, or None. The first read after a pause returns the value of the GPU's last work
    (93 to 98 % right after a run, 0 on the next read 1 s later, research/measurements/2026-11-15-mac-attn-h1.md),
    so the second read counts."""
    if not re.search(r'"Device Utilization %"', read()):
        return None
    wait(1.0)
    util = re.findall(r'"Device Utilization %"\s*=\s*(\d+)', read())
    return max(int(u) for u in util) if util else None


def power() -> dict:
    """{"on_battery": bool | None}; None when there is no battery or it cannot be read."""
    if sys.platform == "win32":
        return hw_win32.power()
    if sys.platform == "darwin":
        text = hw.run(["pmset", "-g", "batt"])
        return {"on_battery": ("Battery Power" in text) if text else None}
    supplies = list(Path("/sys/class/power_supply").glob("*"))
    if not any(hw.read(s / "type") == "Battery" for s in supplies):
        return {"on_battery": None}
    mains = [hw.read(s / "online") for s in supplies if hw.read(s / "type") in ("Mains", "USB")]
    return {"on_battery": ("1" not in mains) if mains else None}


def measure() -> dict:
    swap0 = swap_traffic()
    threads = os.cpu_count() or 1
    if sys.platform == "win32":
        cpu, cpu_busy = {}, hw_win32.cpu_busy_percent(WINDOW)
    else:
        cpu = cpu_sample()
        cpu_busy = round(sum(cpu.values()) / threads, 1)
    swap1 = swap_traffic()
    ram = hw.memory()
    return {
        "taken": time.strftime("%Y-%m-%d %H:%M:%S"),
        "threads": threads,
        "cpu_busy_percent": cpu_busy,
        "busy_processes": dict(sorted(cpu.items(), key=lambda kv: -kv[1])[:5]),
        "loadavg1": round(os.getloadavg()[0], 2) if hasattr(os, "getloadavg") else None,
        "gpus": gpu_cards(),
        "ram_total_gib": ram["total_gib"],
        "ram_available_gib": ram["available_gib"],
        "swap_pages_moved": None if swap0 is None or swap1 is None else swap1 - swap0,
        **power(),
    }


def level(value, limits: tuple, below: bool = False) -> str:
    warn, stop = limits
    if value is None:
        return OK
    if below:
        return STOP if value < stop else WARN if value < warn else OK
    return STOP if value >= stop else WARN if value >= warn else OK


def finding(state: str, text: str, ask: str) -> dict:
    return {"level": state, "what": text, "ask": ask}


def cpu_findings(s: dict) -> list:
    out = []
    heavy = [n for n, v in s["busy_processes"].items() if v >= HEAVY_APP and n not in SYSTEM_TOOLS]
    state = level(s["cpu_busy_percent"], CPU_BUSY_PERCENT)
    if state != OK:
        who = f"close {', '.join(heavy)}" if heavy else "close programs with heavy load"
        out.append(finding(state, f"other programs use {s['cpu_busy_percent']} % of all CPU threads", who))
    if level(s["loadavg1"] and s["loadavg1"] / s["threads"], LOAD_PER_THREAD) != OK:
        out.append(finding(WARN, f"load average {s['loadavg1']} is high (it lags a minute)", "wait a minute"))
    return out


def gpu_findings(s: dict) -> list:
    out = []
    pause = "close browsers, video players and games"
    for g in s["gpus"]:
        state = level(g["busy"], GPU_BUSY_PERCENT)
        if state != OK:
            out.append(finding(state, f"the GPU is {g['busy']} % busy", pause))
        if g["vram_gib"] and g["used_gib"] is not None:
            state = level(g["used_gib"] / g["vram_gib"] * 100, DESKTOP_VRAM_PERCENT)
            if state != OK:
                out.append(finding(state, f"{g['used_gib']} GiB of {g['vram_gib']} GiB VRAM are held by others", pause))
    return out


def system_findings(s: dict) -> list:
    out = []
    if s["ram_total_gib"] and s["ram_available_gib"] is not None:
        state = level(s["ram_available_gib"] / s["ram_total_gib"] * 100, RAM_AVAILABLE_PERCENT, below=True)
        if state != OK:
            out.append(finding(state, f"only {s['ram_available_gib']} GiB of {s['ram_total_gib']} GiB RAM are free",
                               "close memory-hungry programs (browser tabs, virtual machines)"))  # fmt: skip
    if s["swap_pages_moved"]:
        out.append(
            finding(
                WARN, f"the system is swapping ({s['swap_pages_moved']} pages in 1 s)", "close memory-hungry programs"
            )
        )
    if s["on_battery"]:
        out.append(finding(STOP, "the laptop runs on battery and will slow down", "plug in the charger"))
    return out


def evaluate(s: dict) -> dict:
    found = cpu_findings(s) + gpu_findings(s) + system_findings(s)
    worst = max((f["level"] for f in found), key=RANK.get, default=OK)
    return {"state": worst, "findings": found, "signals": s}


def public(result: dict) -> dict:
    """What may be written to a file: no program names, no advice that names them."""
    signals = {k: v for k, v in result["signals"].items() if k != "busy_processes"}
    return {
        "state": result["state"],
        "findings": [{"level": f["level"], "what": f["what"]} for f in result["findings"]],
        "signals": signals,
    }


def ram_by_program() -> dict:
    """{name: GiB resident} summed per program name, Linux only; kernel threads have no resident memory."""
    out: dict = {}
    for status in Path("/proc").glob("[0-9]*/status"):
        text = hw.read(status)
        name, rss = re.search(r"^Name:\s*(.+)$", text, re.M), re.search(r"^VmRSS:\s*(\d+) kB", text, re.M)
        if name and rss:
            out[name.group(1)] = out.get(name.group(1), 0) + int(rss.group(1)) / (1 << 20)
    return out


def vram_by_program() -> dict:
    """{name: GiB of VRAM} from the DRM fdinfo of each process (what the user's own processes show), Linux only."""
    out: dict = {}
    for info in Path("/proc").glob("[0-9]*/fdinfo/*"):
        text = hw.read(info)
        vram = re.search(r"^drm-memory-vram:\s*(\d+) KiB", text, re.M)
        if vram:
            name = hw.read(info.parents[1] / "comm")
            out[name] = max(out.get(name, 0), int(vram.group(1)) / (1 << 20))  # one buffer set is listed per fd
    return out


def is_user_program_linux(pid_dir: Path, uid: int) -> bool:
    """Own process (same UID) that is no service: not in system.slice and not run by a unit of its own. Programs
    that the desktop starts live in app-* scopes or services and count as the user's."""
    if not re.search(rf"^Uid:\s*{uid}\b", hw.read(pid_dir / "status"), re.M):
        return False
    path = next((ln[3:] for ln in hw.read(pid_dir / "cgroup").splitlines() if ln.startswith("0::")), "")
    last = path.rstrip("/").rsplit("/", 1)[-1]
    if "system.slice" in path or "background.slice" in path:
        return False
    return not (last.endswith(".service") and not last.startswith("app-") and "user@" not in last)


def user_program_names() -> set | None:
    """Names of programs the user owns and started: same UID and no service (Linux: cgroup, macOS: app bundle).
    None when it cannot be decided, then only SYSTEM_TOOLS filters."""
    if not hasattr(os, "getuid"):
        return None
    uid = os.getuid()
    if Path("/proc/self/stat").exists():
        return {hw.read(d / "comm") for d in Path("/proc").glob("[0-9]*") if is_user_program_linux(d, uid)} - {""}
    rows = hw.run(["ps", "-Ao", "uid=,comm="]).splitlines()
    names = set()
    for row in rows:
        owner, _, comm = row.strip().partition(" ")
        if owner == str(uid) and ".app/" in comm:
            names.add(Path(comm.strip()).name)
    return names or None


def top_users(s: dict, count: int = 3, mine: set | None = None) -> dict:
    """The biggest user programs by RAM, VRAM and CPU. Terminal and conversation only: never saved, never in JSON."""
    ranked = {"RAM": ram_by_program(), "VRAM": vram_by_program(), "CPU": s["busy_processes"]}
    unit = {"RAM": " GiB", "VRAM": " GiB", "CPU": " % of a core"}
    out = {}
    for kind, table in ranked.items():
        rows = [(n, v) for n, v in table.items() if n not in SYSTEM_TOOLS and not n.startswith(KERNEL_THREADS)]
        rows = [(n, v) for n, v in rows if mine is None or n in mine]  # only the user's own, no services
        top = sorted(rows, key=lambda kv: -kv[1])[:count]
        out[kind] = ", ".join(
            f"{n} ({v:.1f}{unit[kind]})" for n, v in top if v >= (0.3 if kind != "CPU" else HEAVY_APP)
        )
    return {k: v for k, v in out.items() if v}


def free_line(s: dict) -> str:
    parts = []
    if s["ram_available_gib"] is not None:
        parts.append(f"{s['ram_available_gib']} GiB RAM")
    for g in s["gpus"]:
        if g["vram_gib"] and g["used_gib"] is not None:
            parts.append(f"{g['vram_gib'] - g['used_gib']:.1f} GiB VRAM")
    if s["cpu_busy_percent"] is not None:
        parts.append(f"the CPU is {s['cpu_busy_percent']} % busy")
    if not parts:
        return "Right now no live load numbers are available."
    return "Right now " + (", ".join(parts[:-1]) + " are free; " if len(parts) > 1 else "") + parts[-1] + "."


def request(result: dict, users: dict | None = None, minutes: int = 20) -> str:
    """What the agent says to the user before a measurement, in the conversation or on the terminal. It names the
    user's biggest programs: spoken only, never saved. The user decides what to close; system tools (virus
    scanners, indexers, the desktop) are never named and keep running. A refusal is fine: the agent measures anyway
    and the report carries the numbers that were available."""
    s = result["signals"]
    lines = [free_line(s)]
    if users:
        lines.append("The biggest programs of yours: " + "; ".join(f"{k}: {v}" for k, v in users.items()) + ".")
    extra = [f["ask"] for f in result["findings"] if f["ask"].startswith(("plug", "wait"))]
    if result["state"] == OK and not extra:
        return " ".join([*lines, "That is quiet enough to measure."])
    lines.append(
        "I recommend closing browsers, games and similar programs for the measurement"
        + (f", and to {' and '.join(dict.fromkeys(extra))}" if extra else "")
        + f", and leaving the computer alone for the next {minutes} minutes. You decide what you close; virus "
        "scanners and system tools can keep running. Answer 'ok' when you are done and I will check again, "
        "or 'measure' to go ahead as it is: the report then states the available RAM, VRAM and CPU."
    )
    return " ".join(lines)


def compare(before: dict, after: dict) -> dict:
    """The run is disturbed when the machine got worse during it: a higher state or a finding that is new."""
    new = [
        f["what"]
        for f in after["findings"]
        if f["level"] != OK and f["what"] not in {x["what"] for x in before["findings"]}
    ]
    worse = RANK[after["state"]] > RANK[before["state"]]
    return {"changed": bool(new) or worse, "before": before["state"], "after": after["state"], "new_findings": new}


def render(result: dict, cmp: dict | None = None) -> str:
    s = result["signals"]
    lines = [f"- Quiet check ({s['taken']}): {result['state']}"]
    lines += [f"  - {f['level']}: {f['what']}" for f in result["findings"]]
    if cmp:
        verdict = "DISTURBED, repeat or mark the run" if cmp["changed"] else "unchanged"
        lines.append(f"- Quiet check before and after the run: {cmp['before']} -> {cmp['after']}, {verdict}")
        lines += [f"  - new: {w}" for w in cmp["new_findings"]]
    return hw.scrub("\n".join(lines), "", "") + "\n"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output (save it as the 'before' file)")
    ap.add_argument(
        "--ask", action="store_true", help="the request for the user; may name programs: say it, never save it"
    )
    ap.add_argument("--compare", metavar="BEFORE.json", help="after the run: compare with the saved 'before' result")
    args = ap.parse_args(argv)
    result = evaluate(measure())
    if args.ask:
        sys.stdout.write(request(result, top_users(result["signals"], mine=user_program_names())) + "\n")
        return RANK[result["state"]]
    shown = public(result)
    cmp = compare(json.loads(Path(args.compare).read_text()), shown) if args.compare else None
    if args.json:
        sys.stdout.write(hw.scrub(json.dumps({**shown, "compare": cmp}, indent=2)) + "\n")
    else:
        sys.stdout.write(render(shown, cmp))
    if cmp:
        return 3 if cmp["changed"] else 0
    return RANK[result["state"]]


if __name__ == "__main__":
    sys.exit(main())
