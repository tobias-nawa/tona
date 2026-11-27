"""Snapshot of the environment so a measurement can be traced later."""

import http.client
import json
import platform
import subprocess
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

from . import sysmon

REPO = Path(__file__).resolve().parents[2]


def _cmd(*args: str) -> str | None:
    try:
        p = subprocess.run(args, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return p.stdout.strip() if p.returncode == 0 else None


def _read(path) -> str | None:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def _get_json(url: str, path: str, timeout: float = 5.0):
    u = urlsplit(url)
    conn = http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)
    try:
        conn.request("GET", u.path.rstrip("/") + path)
        resp = conn.getresponse()
        data = resp.read(200_000)
        if resp.status != 200:
            return None
        return json.loads(data)
    except (OSError, http.client.HTTPException, ValueError):
        return None
    finally:
        conn.close()


def _os_name() -> str | None:
    for line in (_read("/etc/os-release") or "").splitlines():
        if line.startswith("PRETTY_NAME="):
            return line.split("=", 1)[1].strip('"')
    return None


def _cpu() -> str | None:
    for line in (_read("/proc/cpuinfo") or "").splitlines():
        if line.startswith("model name"):
            return line.split(":", 1)[1].strip()
    return None


def _gpu() -> dict:
    dev = sysmon.find_gpu()
    if not dev:
        return {}
    out = {"sysfs": str(dev)}
    for key, name in (("vram_total_mib", "mem_info_vram_total"), ("vram_used_mib", "mem_info_vram_used")):
        v = _read(dev / name)
        if v and v.isdigit():
            out[key] = round(int(v) / 2**20)
    for key in ("vbios_version", "current_link_speed", "current_link_width", "power_dpm_force_performance_level"):
        v = _read(dev / key)
        if v:
            out[key] = v
    for line in (_read(dev / "uevent") or "").splitlines():
        if line.startswith("PCI_ID="):
            out["pci_id"] = line.split("=", 1)[1]
    return out


def _packages() -> dict:
    names = ["rocm-runtime", "rocm-hip", "hipblaslt", "rocblas", "kernel"]
    out = {}
    raw = _cmd("rpm", "-q", "--qf", "%{NAME} %{VERSION}-%{RELEASE}\n", *names) or ""
    for line in raw.splitlines():
        parts = line.split()
        if len(parts) == 2:
            out[parts[0]] = parts[1]
    return out


def _git() -> dict:
    head = _cmd("git", "-C", str(REPO), "rev-parse", "--short", "HEAD")
    status = _cmd("git", "-C", str(REPO), "status", "--porcelain")
    return {"commit": head, "dirty": bool(status)}


def collect(target: dict | None) -> dict:
    """Without a target (micro-benchmarks) the server queries are skipped."""
    mem = sysmon.meminfo()
    cpu0 = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    info = {
        "time": datetime.now().astimezone().isoformat(timespec="seconds"),
        "kernel": platform.release(),
        "os": _os_name(),
        "python": platform.python_version(),
        "cpu": _cpu(),
        "cpu_governor": _read(cpu0 / "scaling_governor"),
        "cpu_epp": _read(cpu0 / "energy_performance_preference"),
        "mem_total_gib": round(mem.get("MemTotal", 0) / 2**20, 1),
        "mem_avail_gib": round(mem.get("MemAvailable", 0) / 2**20, 1),
        "swap_used_mib": round((mem.get("SwapTotal", 0) - mem.get("SwapFree", 0)) / 1024),
        "gpu": _gpu(),
        "packages": _packages(),
        "git": _git(),
        "target": {k: v for k, v in target.items() if k != "api_key"} if target else None,
        "server": {},
        "warnings": [],
    }
    w = info["warnings"]
    if target:
        for path in ("/v1/models", "/v1/status", "/props"):
            data = _get_json(target["url"], path)
            if data is not None:
                info["server"][path] = data
        if not info["server"].get("/v1/models"):
            w.append(f"{target['url']}/v1/models does not respond: is the server running?")
    ps = _get_json("http://127.0.0.1:11434", "/api/ps")
    if ps and ps.get("models"):
        w.append("Ollama holds models on the GPU: " + ", ".join(m.get("name", "?") for m in ps["models"]))
    if info["swap_used_mib"] > 256:
        w.append(f"Swap in use before start: {info['swap_used_mib']} MiB")
    if info["mem_avail_gib"] < 8:
        w.append(f"only {info['mem_avail_gib']} GiB RAM available before start")
    if info["git"]["dirty"]:
        w.append("Repository has uncommitted changes")
    return info
