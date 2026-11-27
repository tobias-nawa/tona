"""OpenMP gate: every source with a `#pragma omp` is compiled with OpenMP.

Usage: openmp.py DB_DIR [DB_DIR...]
A source compiled as HIP language gets no -fopenmp from find_package(OpenMP); its pragmas then compile to nothing and
the loop runs on one thread, silently (research/measurements/2026-11-18-b1-host-parallel.md). The gate reads the
compile database of each build directory (configure only is enough) and fails for a source that contains a pragma
and whose compile command has no -fopenmp. A missing database is an error, never a silent pass. engine/third_party/
is vendored and exempt.
"""

import json
import re
import sys
from pathlib import Path

from common import EXCLUDED_PREFIXES, repo_root

PRAGMA = re.compile(r"^[ \t]*#[ \t]*pragma[ \t]+omp\b", re.MULTILINE)


def has_pragma(path: Path) -> bool:
    try:
        return bool(PRAGMA.search(path.read_text(errors="replace")))
    except OSError:
        return False


def missing_openmp(entries: list[dict], root: Path) -> list[str]:
    """Sources (relative to root) that have a pragma and a compile command without -fopenmp."""
    bad = set()
    for e in entries:
        src = Path(e["directory"], e["file"]).resolve() if not Path(e["file"]).is_absolute() else Path(e["file"])
        try:
            rel = src.relative_to(root).as_posix()
        except ValueError:
            continue
        if rel.startswith(EXCLUDED_PREFIXES) or not has_pragma(src):
            continue
        command = e.get("command") or " ".join(e.get("arguments", []))
        if "-fopenmp" not in command:
            bad.add(rel)
    return sorted(bad)


def main(argv: list[str]) -> int:
    if not argv:
        print("usage: openmp.py DB_DIR [DB_DIR...]", file=sys.stderr)
        return 2
    root = repo_root()
    failed = False
    for d in argv:
        db = Path(d) / "compile_commands.json"
        if not db.is_file():
            print(f"openmp: no compile database in {d} (configure first)", file=sys.stderr)
            return 2
        for rel in missing_openmp(json.loads(db.read_text()), root):
            print(f"openmp: {rel} has #pragma omp but is compiled without -fopenmp (its loops run on one thread)")
            failed = True
    if not failed:
        print("openmp: ok")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
