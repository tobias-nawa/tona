"""Coverage gates: measure host C++ and Python, enforce the floor and the patch threshold.

Usage:
  coverage_gate.py run [--out DIR] [--base REF] [--summary FILE]
      Builds the engine with gcov at -O0, runs the unit tests (ctest label `unit`) and pytest under coverage.py,
      writes cpp.xml and python.xml (Cobertura) and the HTML reports into DIR, then applies the gates.
  coverage_gate.py check --cpp XML --python XML [--base REF] [--summary FILE]
      Only the gates, on reports that exist already.

Gates (tools/gates/coverage-floor.json holds the floor):
  * total: line coverage of C++ and Python must not drop below the floor. At 1.5 points or more above, the job prints
    the new value to commit. A change that lowers the floor itself fails (compared with the file on --base).
  * patch (with --base): new and changed lines need >= 80 % coverage (diff-cover), C++ and Python together.

Scope: the host code under engine/ without engine/tests, engine/third_party and engine/gpu (device code and the
code that needs the GPU; AGENTS.md "Code gates"), and Python under tools/ and scripts/. Branch coverage is only
reported. Both CI (.github/workflows/tests.yml) and `just coverage` call this file.
"""

import argparse
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FLOOR_FILE = "tools/gates/coverage-floor.json"
PATCH_MIN = 80.0
# The floor sits this far below the measurement, so that another compiler patch level or package state in the CI
# container does not turn the job red by itself (a few lines of inline code more or less)
FLOOR_MARGIN = 0.5
RAISE_HINT_MARGIN = 1.5  # points above the floor at which the job asks to raise it (a full point above the margin)
CPP_FILTER = "engine/"
CPP_EXCLUDES = ("engine/tests/", "engine/third_party/", "engine/gpu/")
# The CPU target of the coverage build equals the CI one, so that local and CI numbers are the same
CPP_CMAKE_ARGS = (
    "-DCMAKE_BUILD_TYPE=Debug",
    "-DTONA_COVERAGE=ON",
    "-DTONA_MARCH=x86-64-v3",
    # Tests that need the model or a llama.cpp build are out of scope, whatever the machine has
    "-DTONA_MODEL=/nonexistent/model.gguf",
    "-DTONA_LLAMACPP_BUILD=/nonexistent",
)


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    print("+", " ".join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, cwd=kw.pop("cwd", ROOT), check=kw.pop("check", True), **kw)


def line_percent(xml_path: Path) -> float:
    """Line coverage of a Cobertura report, in percent (both gcovr and coverage.py write the two counters)."""
    root = ET.parse(xml_path).getroot()
    valid, covered = int(root.get("lines-valid", 0)), int(root.get("lines-covered", 0))
    return 100.0 * covered / valid if valid else 0.0


def branch_percent(xml_path: Path) -> float | None:
    root = ET.parse(xml_path).getroot()
    valid, covered = int(root.get("branches-valid", 0)), int(root.get("branches-covered", 0))
    return 100.0 * covered / valid if valid else None


def floor_value(measured: float) -> float:
    """What to commit as the floor: the measurement cut to one decimal, minus the margin."""
    return round(math.floor(measured * 10 + 1e-9) / 10 - FLOOR_MARGIN, 1)


def measure_cpp(out: Path, jobs: int) -> Path:
    build = ROOT / "work" / "build" / "coverage-build"
    run(["cmake", "-S", "engine", "-B", str(build), "-G", "Ninja", *CPP_CMAKE_ARGS], stdout=subprocess.DEVNULL)
    run(["cmake", "--build", str(build), "-j", str(jobs)])
    for stale in build.rglob("*.gcda"):  # counters of an earlier run would add up
        stale.unlink()
    run(["ctest", "--test-dir", str(build), "-L", "unit", "--output-on-failure", "-j", str(jobs)])
    xml = out / "cpp.xml"
    cmd = ["gcovr", "--root", ".", "--object-directory", str(build), "--filter", CPP_FILTER]
    for ex in CPP_EXCLUDES:
        cmd += ["--exclude", ex]
    cmd += ["--exclude-throw-branches", "--exclude-unreachable-branches", "--cobertura", str(xml)]
    cmd += ["--html-details", str(out / "cpp" / "index.html"), "--json-summary", str(out / "cpp-summary.json")]
    (out / "cpp").mkdir(parents=True, exist_ok=True)
    run(cmd)
    return xml


def measure_python(out: Path) -> Path:
    env = {**os.environ, "COVERAGE_FILE": str(out / "python.coverage")}
    run([sys.executable, "-m", "coverage", "run", "-m", "pytest", "tools/tests"], env=env)
    xml = out / "python.xml"
    run([sys.executable, "-m", "coverage", "xml", "-o", str(xml)], env=env)
    run([sys.executable, "-m", "coverage", "html", "-d", str(out / "python")], env=env, stdout=subprocess.DEVNULL)
    return xml


def read_floor(text: str) -> dict[str, float]:
    return {k: float(v["lines"]) for k, v in json.loads(text).items()}


def floor_on_base(base: str) -> dict[str, float] | None:
    show = run(["git", "show", f"{base}:{FLOOR_FILE}"], check=False, capture_output=True, text=True)
    return read_floor(show.stdout) if show.returncode == 0 else None


def judge_total(measured: dict[str, float], floor: dict[str, float], base: dict[str, float] | None):
    """Problems (fail the job) and notes (printed) of the total-coverage gate."""
    problems, notes = [], []
    for key in sorted(floor):
        if measured[key] + 1e-9 < floor[key]:
            problems.append(f"{key}: line coverage {measured[key]:.2f} % is below the floor {floor[key]:.1f} %")
        elif measured[key] >= floor[key] + RAISE_HINT_MARGIN:
            notes.append(f"{key}: {measured[key]:.2f} % is above the floor; set it to {floor_value(measured[key])}")
        if base is not None and key in base and floor[key] + 1e-9 < base[key]:
            problems.append(f"{key}: the floor drops from {base[key]:.1f} % to {floor[key]:.1f} % in {FLOOR_FILE}")
    return problems, notes


def patch_gate(reports: list[Path], base: str, out: Path) -> tuple[int, str]:
    cmd = ["diff-cover", *map(str, reports), "--compare-branch", base, "--fail-under", str(PATCH_MIN)]
    cmd += ["--format", f"markdown:{out / 'patch.md'}", "--show-uncovered"]
    done = run(cmd, check=False, capture_output=True, text=True)
    sys.stdout.write(done.stdout)
    sys.stderr.write(done.stderr)
    md = out / "patch.md"
    return done.returncode, md.read_text() if md.exists() else ""


def summary_text(measured, floor, extra_rows, patch_md) -> str:
    lines = ["## Coverage (host code, no GPU)", "", "| | Lines | Floor | Branches (info) |", "|---|---:|---:|---:|"]
    for key in sorted(measured):
        br = extra_rows.get(key)
        lines.append(
            f"| {key} | {measured[key]:.2f} % | {floor.get(key, 0):.1f} % | {'-' if br is None else f'{br:.1f} %'} |"
        )
    if patch_md:
        lines += ["", f"### Changed lines (threshold {PATCH_MIN:.0f} %)", "", patch_md]
    return "\n".join(lines) + "\n"


def gate(cpp: Path, py: Path, base: str | None, out: Path, summary: str | None) -> int:
    measured = {"cpp": line_percent(cpp), "python": line_percent(py)}
    branches = {"cpp": branch_percent(cpp), "python": branch_percent(py)}
    floor = read_floor((ROOT / FLOOR_FILE).read_text())
    problems, notes = judge_total(measured, floor, floor_on_base(base) if base else None)
    patch_md = ""
    if base:
        code, patch_md = patch_gate([cpp, py], base, out)
        if code:
            problems.append(f"changed lines are below {PATCH_MIN:.0f} % coverage (see the list above)")
    text = summary_text(measured, floor, branches, patch_md)
    print(text)
    if summary:
        with open(summary, "a", encoding="utf-8") as fh:
            fh.write(text)
    for n in notes:
        print(f"NOTE {n}")
        if summary:
            with open(summary, "a", encoding="utf-8") as fh:
                fh.write(f"\n> {n}\n")
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    return 1 if problems else 0


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "check"):
        p = sub.add_parser(name)
        p.add_argument("--base", help="git ref the patch gate and the floor comparison use (pull requests)")
        p.add_argument("--summary", help="file to append the Markdown summary to ($GITHUB_STEP_SUMMARY)")
        p.add_argument("--out", type=Path, default=ROOT / "work" / "build" / "coverage")
        if name == "run":
            p.add_argument("--jobs", type=int, default=4)
        else:
            p.add_argument("--cpp", type=Path, required=True)
            p.add_argument("--python", type=Path, required=True)
    return ap.parse_args(argv)


def main(argv: list[str]) -> int:
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.cmd == "run":
        cpp, py = measure_cpp(args.out, args.jobs), measure_python(args.out)
    else:
        cpp, py = args.cpp, args.python
    return gate(cpp, py, args.base, args.out, args.summary)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
