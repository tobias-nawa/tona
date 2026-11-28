"""Windows hardware and load probes for hw_report.py and quiet_check.py.

Standard library only (ctypes, winreg, json, re, platform, time). Pure parsers are split out so Linux CI tests them.
"""

from __future__ import annotations

import ctypes
import json
import os
import platform
import re
import time

UNKNOWN = "unknown"
GIB = 1 << 30
VENDOR_IDS = {"10DE": "NVIDIA", "1002": "AMD", "8086": "Intel"}
PF_AVX512F = 40  # IsProcessorFeaturePresent flag for AVX-512F on Windows 10+


class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [
        ("dwLength", ctypes.c_ulong),
        ("dwMemoryLoad", ctypes.c_ulong),
        ("ullTotalPhys", ctypes.c_ulonglong),
        ("ullAvailPhys", ctypes.c_ulonglong),
        ("ullTotalPageFile", ctypes.c_ulonglong),
        ("ullAvailPageFile", ctypes.c_ulonglong),
        ("ullTotalVirtual", ctypes.c_ulonglong),
        ("ullAvailVirtual", ctypes.c_ulonglong),
        ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
    ]


class _FileTime(ctypes.Structure):
    _fields_ = [("dwLowDateTime", ctypes.c_ulong), ("dwHighDateTime", ctypes.c_ulong)]


class _SystemPowerStatus(ctypes.Structure):
    _fields_ = [
        ("ACLineStatus", ctypes.c_byte),
        ("BatteryFlag", ctypes.c_byte),
        ("BatteryLifePercent", ctypes.c_byte),
        ("SystemStatusFlag", ctypes.c_byte),
        ("BatteryLifeTime", ctypes.c_ulong),
        ("BatteryFullLifeTime", ctypes.c_ulong),
    ]


def _ps(run_fn, command: str, timeout: int = 15) -> str:
    return run_fn(["powershell", "-NoProfile", "-Command", command], timeout=timeout).strip()


def parse_nvidia_smi(out: str) -> list[dict]:
    """Parses `nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version` CSV rows."""
    gpus = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 5 or not parts[1].isdigit() or not parts[2].isdigit():
            continue
        busy = int(parts[3]) if parts[3].isdigit() else None
        gpus.append(
            {
                "vendor": "NVIDIA",
                "name": parts[0] or UNKNOWN,
                "vram_gib": round(int(parts[1]) / 1024, 1),
                "vram_used_gib": round(int(parts[2]) / 1024, 2),
                "busy_percent": busy,
                "driver": parts[4] or UNKNOWN,
                "integrated": False,
            }
        )
    return gpus


def _vendor_from_pnp(pnp: str, name: str) -> str:
    m = re.search(r"VEN_([0-9A-Fa-f]{4})", pnp or "")
    if m:
        return VENDOR_IDS.get(m.group(1).upper(), f"pci:{m.group(1).lower()}")
    upper = (name or "").upper()
    for word, label in (("NVIDIA", "NVIDIA"), ("AMD", "AMD"), ("RADEON", "AMD"), ("INTEL", "Intel")):
        if word in upper:
            return label
    return UNKNOWN


def _video_controller_entry(item: dict) -> dict:
    name = str(item.get("Name") or "").strip() or UNKNOWN
    raw_ram = item.get("AdapterRAM")
    ram = int(raw_ram) if isinstance(raw_ram, (int, float)) and raw_ram > 0 else 0
    vendor = _vendor_from_pnp(str(item.get("PNPDeviceID") or ""), name)
    entry = {
        "vendor": vendor,
        "name": name,
        "vram_gib": round(ram / GIB, 1) if ram else None,
        "vram_used_gib": None,
        "busy_percent": None,
        "integrated": vendor == "Intel",
    }
    driver = str(item.get("DriverVersion") or "").strip()
    if driver and vendor == "NVIDIA":
        entry["driver"] = driver
    return entry


def parse_video_controllers(out: str) -> list[dict]:
    """Parses PowerShell JSON for Win32_VideoController (fallback when nvidia-smi is absent)."""
    try:
        data = json.loads(out or "[]")
    except ValueError:
        return []
    items = [data] if isinstance(data, dict) else data if isinstance(data, list) else []
    return [_video_controller_entry(item) for item in items if isinstance(item, dict)]


def parse_nvcc_version(out: str) -> str:
    """Extracts the CUDA release number from `nvcc --version` output."""
    m = re.search(r"release\s+([\d.]+)", out)
    return m.group(1) if m else ""


def nvcc_toolkit(run_fn) -> str:
    return parse_nvcc_version(run_fn(["nvcc", "--version"]))


def _cpu_brand_registry() -> str:
    try:
        import winreg

        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DESCRIPTION\System\CentralProcessor\0") as key:
            value, _ = winreg.QueryValueEx(key, "ProcessorNameString")
            return str(value).strip()
    except OSError:
        return ""
    except ImportError:
        return ""


def _has_avx512() -> bool:
    try:
        return bool(ctypes.windll.kernel32.IsProcessorFeaturePresent(PF_AVX512F))
    except (AttributeError, OSError):
        return False


def cpu(run_fn) -> dict:
    model = (
        _cpu_brand_registry()
        or _ps(run_fn, "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)")
        or platform.processor()
        or UNKNOWN
    )
    cores_text = _ps(run_fn, "(Get-CimInstance Win32_Processor | Measure-Object -Property NumberOfCores -Sum).Sum")
    return {
        "model": model,
        "cores": int(cores_text) if cores_text.isdigit() else None,
        "threads": os.cpu_count(),
        "avx512": _has_avx512(),
    }


def gpus(run_fn) -> list[dict]:
    smi = run_fn(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,utilization.gpu,driver_version",
            "--format=csv,noheader,nounits",
        ]
    )
    rows = parse_nvidia_smi(smi)
    if rows:
        return rows
    ps_out = _ps(
        run_fn,
        "Get-CimInstance Win32_VideoController | "
        "Select-Object Name,AdapterRAM,DriverVersion,PNPDeviceID | ConvertTo-Json -Compress",
    )
    return parse_video_controllers(ps_out)


def memory() -> dict:
    try:
        ms = _MemoryStatusEx()
        ms.dwLength = ctypes.sizeof(ms)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return {"total_gib": None, "available_gib": None}
        return {"total_gib": round(ms.ullTotalPhys / GIB, 1), "available_gib": round(ms.ullAvailPhys / GIB, 1)}
    except (AttributeError, OSError):
        return {"total_gib": None, "available_gib": None}


def os_info() -> dict:
    return {
        "name": f"Windows {platform.release()}".strip(),
        "kernel": platform.version() or platform.release(),
        "machine": platform.machine(),
    }


def _filetime_int(ft: _FileTime) -> int:
    return (ft.dwHighDateTime << 32) | ft.dwLowDateTime


def _system_times() -> tuple[int, int, int] | None:
    try:
        idle, kernel, user = _FileTime(), _FileTime(), _FileTime()
        ok = ctypes.windll.kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user))
        if not ok:
            return None
        return _filetime_int(idle), _filetime_int(kernel), _filetime_int(user)
    except (AttributeError, OSError):
        return None


def cpu_busy_percent(sample_seconds: float) -> float | None:
    """Whole-machine CPU busy percent over sample_seconds via Win32 GetSystemTimes (no process names)."""
    t0 = _system_times()
    if t0 is None:
        return None
    time.sleep(sample_seconds)
    t1 = _system_times()
    if t1 is None:
        return None
    idle = t1[0] - t0[0]
    total = (t1[1] - t0[1]) + (t1[2] - t0[2])  # kernel time includes idle time
    if total <= 0:
        return None
    return round(max(0.0, (total - idle) / total * 100.0), 1)


def power() -> dict:
    """{"on_battery": bool | None} via Win32 GetSystemPowerStatus."""
    try:
        st = _SystemPowerStatus()
        if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
            return {"on_battery": None}
        if (st.BatteryFlag & 0xFF) == 128 or (st.ACLineStatus & 0xFF) == 255:
            return {"on_battery": None}
        return {"on_battery": st.ACLineStatus == 0}
    except (AttributeError, OSError):
        return {"on_battery": None}
