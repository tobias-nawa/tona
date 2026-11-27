#!/usr/bin/env python3
"""Prints the "Test conditions" block for pull requests and measurement files.

    python3 tools/hw_report.py                       # Markdown
    python3 tools/hw_report.py --json                # machine-readable
    python3 tools/hw_report.py --model work/models/coder-iq1_m/<file>.gguf

Standard library only, Python 3.9+, Linux and macOS. It reads, it never changes anything and never starts the
engine. Every field that cannot be determined is "unknown" (an honest gap beats a guess). The home directory and the
user name never appear in the output, so the block can be pasted into a public pull request.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UNKNOWN = "unknown"
GIB = 1 << 30
PCI_VENDORS = {0x1002: "AMD", 0x10DE: "NVIDIA", 0x8086: "Intel"}
SAMPLE_SECONDS = 0.5  # long enough to see a busy process, short enough not to slow the report


def run(cmd: list, timeout: int = 10) -> str:
    """Stdout of a command, "" if it is missing, fails or hangs."""
    if not shutil.which(cmd[0]):
        return ""
    try:
        # C locale: tools such as ps and sysctl print a decimal comma under a German or French locale
        env = {**os.environ, "LC_ALL": "C", "LANG": "C"}
        done = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    except (OSError, subprocess.SubprocessError):
        return ""
    return done.stdout if done.returncode == 0 else ""


def read(path) -> str:
    try:
        return Path(path).read_text(errors="replace").strip()
    except OSError:
        return ""


def scrub(text: str, home: str = "", user: str = "") -> str:
    """Replaces the home directory by `~` and the user name inside paths, so nothing private is published."""
    home = home or str(Path.home())
    user = user or os.environ.get("USER", "")
    if home and home != "/":
        text = text.replace(home, "~")
    if user and len(user) > 2:
        text = re.sub(rf"(?<=/){re.escape(user)}(?=/|$)", "<user>", text)
    return text


def cpu_linux() -> dict:
    info = read("/proc/cpuinfo")
    model = re.search(r"^model name\s*:\s*(.+)$", info, re.M)
    cores = {m for m in re.findall(r"^physical id\s*:\s*(\d+)\n(?:.*\n)*?core id\s*:\s*(\d+)", info, re.M)}
    flags = re.search(r"^flags\s*:\s*(.+)$", info, re.M)
    has = set(flags.group(1).split()) if flags else set()
    return {
        "model": model.group(1).strip() if model else UNKNOWN,
        "cores": len(cores) or None,
        "threads": os.cpu_count(),
        "avx512": "avx512f" in has,
    }


def cpu_macos() -> dict:
    cores = run(["sysctl", "-n", "hw.physicalcpu"]).strip()
    return {
        "model": run(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or platform.processor() or UNKNOWN,
        "cores": int(cores) if cores.isdigit() else None,
        "threads": os.cpu_count(),
        "avx512": False,
    }


def kfd_arches() -> dict:
    """render minor -> gfx name, as installer/core/hw.cpp (gfx_target_version 120000 -> gfx1200)."""
    out = {}
    for node in Path("/sys/class/kfd/kfd/topology/nodes").glob("*"):
        props = dict(line.split(None, 1) for line in read(node / "properties").splitlines() if " " in line)
        version = int(props.get("gfx_target_version", "0") or 0)
        if version > 0:
            out[int(props.get("drm_render_minor", "-1"))] = (
                f"gfx{version // 10000}{version // 100 % 100:x}{version % 100:x}"
            )
    return out


def render_minor(dev: Path) -> int:
    for entry in (dev / "drm").glob("renderD*"):
        return int(entry.name[7:])
    return -1


def to_int(text: str) -> int:
    try:
        return int(text, 0)
    except ValueError:
        return 0


def gpus_sysfs() -> list:
    arches = kfd_arches()
    gpus = []
    for card in sorted(Path("/sys/class/drm").glob("card*")):
        dev = card / "device"
        if "-" in card.name or not (dev / "vendor").exists():
            continue
        vendor, device = to_int(read(dev / "vendor")), to_int(read(dev / "device"))
        total, used = to_int(read(dev / "mem_info_vram_total")), to_int(read(dev / "mem_info_vram_used"))
        gpus.append(
            {
                "vendor": PCI_VENDORS.get(vendor, f"pci:{vendor:04x}"),
                "pci_id": f"{vendor:04x}:{device:04x}",
                "name": UNKNOWN,
                "arch": arches.get(render_minor(dev)),
                "vram_gib": round(total / GIB, 1) if total else None,
                "vram_used_gib": round(used / GIB, 2) if total else None,
                "busy_percent": to_int(read(dev / "gpu_busy_percent")) if total else None,
                "integrated": vendor == 0x8086 or (vendor == 0x1002 and 0 < total < 4 * GIB),
            }
        )
    return gpus


def rocminfo_names() -> dict:
    """gfx name -> marketing name, from `rocminfo` when it is installed."""
    names, arch = {}, ""
    for line in run(["rocminfo"]).splitlines():
        key, _, value = line.strip().partition(":")
        value = value.strip()
        if key == "Name":
            arch = value if value.startswith("gfx") else ""
        elif key == "Marketing Name" and arch:
            names.setdefault(arch, value)  # in an agent block, "Name" comes before "Marketing Name"
            arch = ""
    return names


def nvidia_rows() -> list:
    out = run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used,driver_version",
               "--format=csv,noheader,nounits"])  # fmt: skip
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4 and parts[1].isdigit() and parts[2].isdigit():
            rows.append(parts)
    return rows


def gpus_linux() -> list:
    gpus = gpus_sysfs()
    names = rocminfo_names()
    for gpu in gpus:
        if gpu["arch"] in names and names[gpu["arch"]]:
            gpu["name"] = names[gpu["arch"]]
    rows = nvidia_rows()
    for gpu in (g for g in gpus if g["vendor"] == "NVIDIA"):
        if not rows:
            break
        name, total, used, driver = rows.pop(0)
        gpu.update(name=name, vram_gib=round(int(total) / 1024, 1), vram_used_gib=round(int(used) / 1024, 2))
        gpu["driver"] = driver
    return gpus


def gpus_macos() -> list:
    """system_profiler only. recommendedMaxWorkingSetSize needs a Metal call, so it is left out on purpose."""
    try:
        data = json.loads(run(["system_profiler", "SPDisplaysDataType", "-json"], timeout=30) or "{}")
    except ValueError:
        return []
    ram = int(run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
    return [
        {
            "vendor": g.get("spdisplays_vendor", "Apple").replace("sppci_vendor_", ""),
            "name": g.get("sppci_model", UNKNOWN),
            "cores": g.get("sppci_cores"),
            "vram_gib": round(ram / GIB, 1) if ram else None,  # unified memory: shared with the CPU
            "unified": True,
        }
        for g in data.get("SPDisplaysDataType", [])
    ]


def memory() -> dict:
    if sys.platform == "darwin":
        total = int(run(["sysctl", "-n", "hw.memsize"]).strip() or 0)
        return {"total_gib": round(total / GIB, 1) if total else None, "available_gib": None}
    info = read("/proc/meminfo")
    out = {}
    for key, name in (("MemTotal", "total_gib"), ("MemAvailable", "available_gib")):
        m = re.search(rf"^{key}:\s+(\d+) kB", info, re.M)
        out[name] = round(int(m.group(1)) / (1 << 20), 1) if m else None
    return out


def os_info() -> dict:
    if sys.platform == "darwin":
        name = f"macOS {run(['sw_vers', '-productVersion']).strip()}".strip()
    else:
        m = re.search(r'^PRETTY_NAME="?([^"\n]+)"?', read("/etc/os-release"), re.M)
        name = m.group(1) if m else platform.system()
    return {"name": name, "kernel": platform.release(), "machine": platform.machine()}


def toolkit() -> dict:
    """ROCm or CUDA version where they can be read without running a GPU program."""
    out = {}
    for base in filter(None, (os.environ.get("ROCM_PATH"), "/opt/rocm")):
        version = read(Path(base) / ".info" / "version")
        if version:
            out["rocm"] = version
            break
    hip = (
        run([str(ROOT / "work" / "rocm-10.1" / "bin" / "hipconfig"), "--version"]).strip()
        or run(["hipconfig", "--version"]).strip()
    )
    if hip and "rocm" not in out:
        out["hip"] = hip
    m = re.search(r"CUDA Version:\s*([\d.]+)", run(["nvidia-smi"]))
    if m:
        out["cuda"] = m.group(1)
    amdgpu = read("/sys/module/amdgpu/version")
    if amdgpu:
        out["amdgpu_module"] = amdgpu
    return out


def tona_env() -> dict:
    return {k: v for k, v in sorted(os.environ.items()) if k.startswith("TONA_")}


def cgroup_limits() -> dict:
    """memory.max and memory.swap.max of the own cgroup (the nearest ancestor that sets one)."""
    line = read("/proc/self/cgroup").splitlines()
    path = next((entry[3:] for entry in line if entry.startswith("0::")), "")
    if not path:
        return {}
    cur = Path("/sys/fs/cgroup") / path.lstrip("/")
    out = {}
    while str(cur).startswith("/sys/fs/cgroup"):
        for key, file in (("MemoryMax", "memory.max"), ("MemorySwapMax", "memory.swap.max")):
            value = read(cur / file)
            if key not in out and value and value != "max":
                out[key] = value if not value.isdigit() else f"{int(value) / GIB:.1f}G"
        if cur == Path("/sys/fs/cgroup") or len(out) == 2:
            break
        cur = cur.parent
    return out


def cpu_ticks() -> dict:
    """pid -> (name, utime + stime) from /proc, Linux only."""
    ticks = {}
    for stat in Path("/proc").glob("[0-9]*/stat"):
        m = re.match(r"(\d+) \((.*)\) (.*)", read(stat), re.S)
        if m:
            fields = m.group(3).split()
            ticks[int(m.group(1))] = (m.group(2), int(fields[11]) + int(fields[12]))
    return ticks


def cpu_busy_percent() -> float | None:
    """Share of all hardware threads that other processes use over SAMPLE_SECONDS. A sum: no names are kept,
    which programs run is nobody's business but the user's."""
    if not Path("/proc/self/stat").exists():
        rows = run(["ps", "-Ao", "pcpu="]).split()  # macOS: share over the process lifetime, the best available
        return round(sum(float(x) for x in rows) / (os.cpu_count() or 1), 1) if rows else None
    before = cpu_ticks()
    time.sleep(SAMPLE_SECONDS)
    me = os.getpid()
    used = sum(t - before[pid][1] for pid, (_, t) in cpu_ticks().items() if pid != me and pid in before)
    return round(used / os.sysconf("SC_CLK_TCK") / SAMPLE_SECONDS / (os.cpu_count() or 1) * 100, 1)


def swap_used_gib() -> float | None:
    info = read("/proc/meminfo")
    total, free = (re.search(rf"^{k}:\s+(\d+) kB", info, re.M) for k in ("SwapTotal", "SwapFree"))
    return round((int(total.group(1)) - int(free.group(1))) / (1 << 20), 2) if total and free else None


def foreign_load() -> dict:
    """Load of the machine before the run, as sums only: no process names, no command lines."""
    load = os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)
    return {
        "loadavg": [round(x, 2) for x in load],
        "cpu_busy_percent": cpu_busy_percent(),
        "swap_used_gib": swap_used_gib(),
    }


def sandbox_info(env=None, exists=os.path.exists) -> dict:
    """Whether this looks like a cloud or CI sandbox (CI variables, container markers). Measurements from there do not
    count. A missing GPU is no reason: a CPU-only machine is real target hardware."""
    env = os.environ if env is None else env
    reasons = [f"{v} set" for v in ("CI", "GITHUB_ACTIONS") if env.get(v)]
    reasons += [f"{p} exists" for p in ("/.dockerenv", "/run/.containerenv") if exists(p)]
    return {"sandbox": bool(reasons), "reasons": reasons}


def git(*args: str) -> str:
    return run(["git", "-C", str(ROOT), *args]).strip()


def build_info() -> dict:
    commit = git("rev-parse", "--short", "HEAD")
    return {"commit": commit or UNKNOWN, "dirty": bool(git("status", "--porcelain")) if commit else None}


def pinned_entry(name: str) -> tuple:
    """(size in bytes, SHA-256) pinned in toolchain/models.toml for this file name; (0, "") if it is not listed."""
    toml = read(ROOT / "toolchain" / "models.toml")
    m = re.search(rf'path = "(?:[^"]*/)?{re.escape(name)}".*?size = (\d+), sha256 = "([0-9a-f]{{64}})"', toml)
    return (int(m.group(1)), m.group(2)) if m else (0, "")


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_info(path: str, hash_it: bool) -> dict:
    """Recognizes the model by file name and size against toolchain/models.toml; the SHA-256 only on request."""
    p = Path(path)
    if not p.is_file():
        return {"file": p.name, "error": "not a file"}
    size = p.stat().st_size
    pinned_size, pinned = pinned_entry(p.name)
    sha = file_sha(p) if hash_it else ""
    return {
        "file": p.name,
        "size_gib": round(size / GIB, 1),
        "recognized": bool(pinned) and size == pinned_size,
        "sha256_short": sha[:12] or None,
        "matches_pinned": (sha == pinned) if (sha and pinned) else None,
    }


def collect(model: str = "", hash_model: bool = False) -> dict:
    mac = sys.platform == "darwin"
    report = {
        "cpu": cpu_macos() if mac else cpu_linux(),
        "gpus": gpus_macos() if mac else gpus_linux(),
        "ram": memory(),
        "os": os_info(),
        "toolkit": toolkit(),
        "limits": {"env": tona_env(), "cgroup": {} if mac else cgroup_limits()},
        "foreign_load": foreign_load(),
        "build": build_info(),
    }
    report["sandbox"] = sandbox_info()
    if model:
        report["model"] = model_info(model, hash_model)
    return report


def fmt(value) -> str:
    return UNKNOWN if value is None else str(value)


def gpu_line(g: dict) -> str:
    name = g["name"] if g.get("name") != UNKNOWN else f"{g['vendor']} {g.get('pci_id', '')}".strip()
    parts = [name]
    if g.get("arch"):
        parts.append(g["arch"])
    if g.get("vram_gib") is not None:
        kind = "unified memory" if g.get("unified") else "VRAM"
        parts.append(f"{g['vram_gib']} GiB {kind}")
    if g.get("vram_used_gib") is not None:
        parts.append(f"{g['vram_used_gib']} GiB held by others before the run")
    if g.get("busy_percent") is not None:
        parts.append(f"{g['busy_percent']} % busy before the run")
    if g.get("integrated"):
        parts.append("integrated")
    return ", ".join(parts)


def hardware_lines(r: dict) -> list:
    cpu, ram = r["cpu"], r["ram"]
    lines = [f"- CPU: {cpu['model']}, {fmt(cpu['cores'])} cores / {fmt(cpu['threads'])} threads"]
    lines += [f"- GPU: {gpu_line(g)}" for g in r["gpus"]] or ["- GPU: none detected"]
    avail = f", {fmt(ram['available_gib'])} GiB available before the run" if ram["available_gib"] is not None else ""
    lines.append(f"- RAM: {fmt(ram['total_gib'])} GiB{avail}")
    return lines


def software_lines(r: dict) -> list:
    tk = r["toolkit"]
    kit = ", ".join(f"{k} {v}" for k, v in tk.items()) or "no toolkit found"
    drivers = sorted({g["driver"] for g in r["gpus"] if g.get("driver")})
    return [
        f"- OS: {r['os']['name']}, kernel {r['os']['kernel']} ({r['os']['machine']})",
        f"- Driver and toolkit: {kit}" + (f", NVIDIA driver {', '.join(drivers)}" if drivers else ""),
    ]


def limit_lines(r: dict) -> list:
    env, cg = r["limits"]["env"], r["limits"]["cgroup"]
    cgroup = ", ".join(f"{k}={v}" for k, v in cg.items()) or "none set"
    return [
        "- TONA_* variables: " + (" ".join(f"`{k}={v}`" for k, v in env.items()) or "none set"),
        f"- cgroup limits of this shell: {cgroup}",
    ]


def load_lines(r: dict) -> list:
    load = r["foreign_load"]
    busy = f"{load['cpu_busy_percent']} %" if load["cpu_busy_percent"] is not None else UNKNOWN
    swap = f", swap in use {load['swap_used_gib']} GiB" if load["swap_used_gib"] is not None else ""
    return [
        f"- Load average (1, 5, 15 min): {', '.join(str(x) for x in load['loadavg'])}",
        f"- CPU use by other programs over {SAMPLE_SECONDS} s: {busy} of all threads{swap}",
    ]


def build_lines(r: dict) -> list:
    b = r["build"]
    state = "unknown" if b["dirty"] is None else ("dirty" if b["dirty"] else "clean")
    lines = [f"- Commit: `{b['commit']}` ({state} tree)"]
    if "model" in r:
        m = r["model"]
        sha = f", sha256 `{m['sha256_short']}`" if m.get("sha256_short") else ""
        note = {True: " (matches toolchain/models.toml)", False: " (DIFFERS from toolchain/models.toml)"}.get(
            m.get("matches_pinned")
        )
        if note is None:
            note = (
                " (recognized by name and size: toolchain/models.toml)"
                if m.get("recognized")
                else ("" if sha else " (unknown model, run with --hash)")
            )
        lines.append(f"- Model: `{m['file']}`, {fmt(m.get('size_gib'))} GiB{sha}{note}")
    return lines


def sandbox_lines(r: dict) -> list:
    box = r.get("sandbox")
    if box is None:
        return []
    return [f"- sandbox: {'yes (' + ', '.join(box['reasons']) + ')' if box['sandbox'] else 'no'}"]


def render_markdown(report: dict, home: str = "", user: str = "") -> str:
    lines = ["### Test conditions", ""]
    for part in (hardware_lines, software_lines, limit_lines, load_lines, sandbox_lines, build_lines):
        lines += part(report)
    return scrub("\n".join(lines), home, user) + "\n"


def render_json(report: dict, home: str = "", user: str = "") -> str:
    return scrub(json.dumps(report, indent=2, sort_keys=True), home, user) + "\n"


def main(argv: list | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--model", metavar="FILE", help="model file to report (recognized by name and size)")
    ap.add_argument("--hash", action="store_true", help="also compute the model's SHA-256 (tens of GB take minutes)")
    args = ap.parse_args(argv)
    report = collect(args.model or "", args.hash)
    sys.stdout.write(render_json(report) if args.json else render_markdown(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
