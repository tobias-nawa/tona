"""Function gate (lizard): per function at most 80 lines, cyclomatic complexity 15, 6 parameters.

Usage: functions.py [FILE...]  (no arguments: all tracked files)

Nesting depth (at most 5) is checked by clang-tidy (readability-function-size, .clang-tidy): lizard's nesting count
is wrong for C++ (consecutive brace-less loops add up; engine/cpu/ref.cpp hc_mix, depth 2, measured as 6).

An agent reads and checks a short, flat function in one go. There is no exception list: split the function, and pass
many related values as one struct (for GPU kernels this costs nothing, the kernel arguments are one buffer either
way).
"""

import csv
import io
import subprocess
import sys

from common import SIZE_SUFFIXES, fail_if_empty, gate_files, repo_root, report

LIMITS = {"length": 80, "ccn": 15, "params": 6}


def measure(files: list[str], language: str) -> list[list[str]]:
    # Files are passed explicitly so that .hip counts as C++. Without files lizard would scan the working directory.
    if not files:
        return []
    cmd = ["lizard", "-l", language, "--csv", *files]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root(), check=False)
    if out.returncode not in (0, 1):
        print(out.stderr, file=sys.stderr)
        sys.exit(2)
    return list(csv.reader(io.StringIO(out.stdout)))


def violations(row: list[str]) -> list[str]:
    # lizard CSV: nloc, ccn, tokens, params, length, location, file, name, long_name, start, end
    values = {"ccn": int(row[1]), "params": int(row[3]), "length": int(row[4])}
    return [f"{k} {values[k]} > {LIMITS[k]}" for k in LIMITS if values[k] > LIMITS[k]]


def main() -> int:
    argv = sys.argv[1:]
    cpp, explicit = gate_files(argv, SIZE_SUFFIXES)
    py, _ = gate_files(argv, (".py",))
    if not explicit:
        fail_if_empty(cpp + py, "functions")
    rows = measure(cpp, "cpp") + measure(py, "python")
    problems = []
    for row in rows:
        if bad := violations(row):
            problems.append(f"{row[6]}:{row[9]}: {row[7]}: {', '.join(bad)}")
    return report("functions", problems, len(rows), "Split the function; group parameters into a struct.")


if __name__ == "__main__":
    sys.exit(main())
