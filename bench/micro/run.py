#!/usr/bin/env python3
"""Builds and runs the micro-benchmarks and writes a report to research/measurements/.

Usage (after `source toolchain/env.sh`): bench/micro/run.py [--reps 7] [--only vram,pcie,launch,wmma,gemm,ram]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import date
from pathlib import Path

MICRO = Path(__file__).resolve().parent
REPO = MICRO.parents[1]
sys.path.insert(0, str(REPO / "bench"))
from ibench import envinfo, report, stats  # noqa: E402
from ibench.sysmon import SysMon  # noqa: E402

BUILD = REPO / "work" / "build"


# Environment without the toolchain from work/, for comparing with Fedora's ROCm
SYSTEM_ENV = {k: v for k, v in os.environ.items() if k not in ("ROCM_PATH", "HIP_PATH", "VIRTUAL_ENV")}
SYSTEM_ENV["PATH"] = ":".join(p for p in os.environ["PATH"].split(":") if "/work/rocm-" not in p)

COMMON_FLAGS = ["-std=c++20", "-Wall", "-Werror"]


def _compile_hip(compiler: list[str], opt: str, source: str, out: Path, libs: list[str], **kwargs):
    cmd = [*compiler, opt, *COMMON_FLAGS, str(MICRO / source), "-o", str(out), *libs]
    return subprocess.run(cmd, check=kwargs.pop("check", True), **kwargs)


def _build_system() -> tuple[Path, Path | None]:
    hipcc = ["/usr/bin/hipcc", "--offload-arch=gfx1200"]
    gpu = BUILD / "gpubench-system"
    _compile_hip(hipcc, "-O3", "gpubench.hip", gpu, ["-lamdhip64"], env=SYSTEM_ENV)
    gemm = BUILD / "gemm-system"
    # needs hipblas-devel from Fedora; without it the GEMM part is skipped
    r = _compile_hip(hipcc, "-O2", "gemm.hip", gemm, ["-lamdhip64", "-lhipblas"], env=SYSTEM_ENV, check=False)
    if r.returncode != 0:
        print("GEMM cannot be built with the system ROCm (hipblas-devel missing?), skipped", file=sys.stderr)
        return gpu, None
    return gpu, gemm


def _build_venv() -> tuple[Path, Path]:
    env = os.environ
    if "TONA_HIPCXX" not in env:
        sys.exit("run `source toolchain/env.sh` first")
    hipcc = [env["TONA_HIPCXX"], *env["TONA_HIPFLAGS"].split()]
    libs = env["TONA_HIPLDFLAGS"].split()
    gpu, gemm = BUILD / "gpubench", BUILD / "gemm"
    _compile_hip(hipcc, "-O3", "gpubench.hip", gpu, libs)
    _compile_hip(hipcc, "-O2", "gemm.hip", gemm, [*libs, "-lhipblas"])
    return gpu, gemm


def build(rocm: str) -> tuple[Path, Path, Path | None]:
    BUILD.mkdir(parents=True, exist_ok=True)
    ram = BUILD / "rambench"
    gpu, gemm = _build_system() if rocm == "system" else _build_venv()
    cmd = ["g++", "-O3", "-march=znver5", *COMMON_FLAGS, str(MICRO / "rambench.cpp"), "-o", str(ram), "-lpthread"]
    subprocess.run(cmd, check=True)
    return gpu, ram, gemm


def run(cmd: list[str], env: dict | None = None) -> tuple[dict, list[dict]]:
    print("$", " ".join(cmd), file=sys.stderr)
    out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env).stdout
    lines = [json.loads(x) for x in out.splitlines() if x.strip()]
    return lines[0], lines[1:]


def _measure(args, gpu: Path, ram: Path, gemm: Path | None, mon: SysMon) -> tuple[dict, list[dict]]:
    parts = args.only.split(",")
    gpu_env = SYSTEM_ENV if args.rocm == "system" else None
    meta, samples = {}, []
    for p in [x for x in parts if x not in ("ram", "gemm")]:
        t0 = time.perf_counter()
        m, s = run([str(gpu), p, str(args.reps)], gpu_env)
        meta["gpu"] = m
        meta.setdefault("parts", {})[p] = mon.window(t0, time.perf_counter())
        samples += s
    if "gemm" in parts and gemm:
        t0 = time.perf_counter()
        m, s = run([str(gemm), str(args.reps)], gpu_env)
        meta["gpu"] = m
        meta.setdefault("parts", {})["gemm"] = mon.window(t0, time.perf_counter())
        samples += s
    if "ram" in parts:
        m, s = run([str(ram), "2048", str(args.reps)])
        meta["ram"] = m
        samples += s
    return meta, samples


def _rows(samples: list[dict]) -> list[dict]:
    groups: dict[tuple, list[float]] = {}
    for s in samples:
        groups.setdefault((s["test"], s["var"], s["param"], s["unit"]), []).append(s["value"])
    return [
        {"test": t, "var": v, "param": p, "unit": u, **stats.summarize(vals)} for (t, v, p, u), vals in groups.items()
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--only", default="vram,pcie,launch,persist,wmma,fma,gemm,ram")
    ap.add_argument("--label", default="")
    ap.add_argument(
        "--rocm",
        choices=["venv", "system"],
        default="venv",
        help="venv: ROCm from work/ (default); system: Fedora's /usr (comparison)",
    )
    ap.add_argument("--out-dir", default=str(REPO / "research" / "measurements"))
    args = ap.parse_args()

    gpu, ram, gemm = build(args.rocm)
    env = envinfo.collect(None)
    for w in env["warnings"]:
        print(f"Warning: {w}", file=sys.stderr)
    mon = SysMon(0.1).start()
    try:
        meta, samples = _measure(args, gpu, ram, gemm, mon)
    finally:
        mon.stop()

    sysw = mon.window(mon.samples[0][0], mon.samples[-1][0]) if mon.samples else {}
    data = {
        "tool": "micro",
        "rocm": args.rocm,
        "env": env,
        "meta": meta,
        "system": sysw,
        "rows": _rows(samples),
        "samples": samples,
    }

    stem = f"{date.today().isoformat()}-micro" + (f"-{args.label}" if args.label else "")
    base = report.unique_path(Path(args.out_dir), stem)
    base.parent.mkdir(parents=True, exist_ok=True)
    jp, mp = report.with_ext(base, ".json"), report.with_ext(base, ".md")
    jp.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    mp.write_text(markdown(data, jp.name))
    print(f"Report: {mp}", file=sys.stderr)
    return 0


PARAM = {
    "vram": "Working set MiB",
    "pcie": "Block size KiB",
    "launch": "",
    "persist": "Blocks",
    "peak": "",
    "gemm": "",
    "ram": "Threads",
}
TITLES = {
    "vram": "VRAM and caches",
    "pcie": "PCIe",
    "launch": "Kernel launch cost",
    "persist": "Synchronization inside a running kernel",
    "gemm": "GEMM via hipBLAS (A, B f16/bf16, accumulate f32)",
    "peak": "Compute peak",
    "ram": "RAM (CPU, AVX-512)",
}


def _env_table(data: dict) -> list[str]:
    env, g = data["env"], data["meta"].get("gpu", {})
    num = report.num
    out = [
        "| | |",
        "|---|---|",
        f"| GPU | {g.get('device')} ({g.get('arch')}), {g.get('cus')} CUs, "
        f"driver clock {g.get('clock_khz', 0) / 1000:.0f} MHz, "
        f"memory bus {g.get('bus_bits')} bit, L2 {g.get('l2_bytes', 0) >> 20} MiB |",
        f"| HIP | runtime {g.get('hip_runtime')}, headers {g.get('hip_headers')} |",
        f"| Repo | {env['git']['commit']}{' (modified)' if env['git']['dirty'] else ''} |",
        f"| Kernel / CPU | {env['kernel']} / {env['cpu']}, governor {env['cpu_governor']} |",
        f"| RAM | {env['mem_total_gib']} GiB |",
        f"| GPU before start | {env['gpu'].get('vram_used_mib')} MiB VRAM in use (desktop), "
        f"DPM {env['gpu'].get('power_dpm_force_performance_level')} |",
    ]
    for part, w in data["meta"].get("parts", {}).items():
        out.append(
            f"| GPU during `{part}` | clock mean {num(w.get('sclk_mean_mhz'))} MHz, "
            f"power mean {num(w.get('power_mean_w'))} W, "
            f"temp. max {num(w.get('temp_max_c'))} °C, load {num(w.get('gpu_busy_mean'))} % |"
        )
    s = data["system"]
    if s and not data["meta"].get("parts"):
        out.append(
            f"| GPU during run | clock mean {num(s.get('sclk_mean_mhz'))} MHz, "
            f"power mean {num(s.get('power_mean_w'))} W, temp. max {num(s.get('temp_max_c'))} °C |"
        )
    return out


def _test_table(test: str, rows: list[dict]) -> list[str]:
    num = report.num
    out = [
        "",
        f"## {TITLES[test]}",
        "",
        f"| Measurement | {PARAM[test]} | Median | Min-max | stdev | Unit |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in rows:
        param = num(r["param"]) if PARAM[test] else ""
        out.append(
            f"| {r['var']} | {param} | {num(r['median'])} | {num(r['min'])}-{num(r['max'])} | "
            f"{num(r['stdev'])} | {r['unit']} |"
        )
    return out


def markdown(data: dict, json_name: str) -> str:
    env = data["env"]
    out = [
        f"# {env['time'][:10]}: micro-benchmarks gfx1200",
        "",
        f"Generated by `bench/micro/run.py`. Raw data: [{json_name}]({json_name}).",
        "",
        "## Environment",
        "",
        *_env_table(data),
    ]
    if env["warnings"]:
        out += ["", "## Warnings", ""] + [f"- {w}" for w in env["warnings"]]
    out += ["", "Median (min-max) and stdev over the repetitions. Bandwidth in GB/s is decimal (10^9)."]
    for test in TITLES:
        rows = [r for r in data["rows"] if r["test"] == test]
        if rows:
            out += _test_table(test, rows)
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
