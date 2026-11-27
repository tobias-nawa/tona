#!/usr/bin/python3 -I
"""Release check: every shared library a program needs is shipped or belongs to the base system.

release_needed.py PROGRAM SHIPPED_NAME...

PROGRAM's DT_NEEDED entries (readelf -d) must each be one of SHIPPED_NAME (the file names under runtime/) or one of
the libraries of the Fedora base system the release does not ship (libc, libm, libstdc++, libgcc_s, the dynamic loader).
A library the toolchain provides and the release forgot (the OpenMP runtime was one: 2026-11-18) would otherwise be
taken from the user's system, or be missing, at the first start. Exit 1 and a list of the missing names on failure.

RUNPATH/RPATH of a shipped program may contain only $ORIGIN-relative entries: an absolute path (the toolchain's lib
directory, added by -fopenmp in the HIP link: 2026-11-18) makes the program differ between machines and lets it run
from a path that the user's system does not have. --runpath-only checks just that, for programs without the ROCm part.
"""

import re
import subprocess
import sys

# Not shipped: the base system of the target (docs/maintain/release.md, scripts/release-build.sh)
BASE_SYSTEM = {
    "libc.so.6",
    "libm.so.6",
    "libstdc++.so.6",
    "libgcc_s.so.1",
    "ld-linux-x86-64.so.2",
    "libpthread.so.0",
    "libdl.so.2",
    "librt.so.1",
}
NEEDED = re.compile(r"\(NEEDED\)\s+Shared library: \[([^\]]+)\]")


RUNPATH = re.compile(r"\((?:RUNPATH|RPATH)\)\s+Library (?:runpath|rpath): \[([^\]]*)\]")


def runpath_entries(readelf_dynamic: str) -> list[str]:
    return [e for value in RUNPATH.findall(readelf_dynamic) for e in value.split(":") if e]


def bad_runpath(entries: list[str]) -> list[str]:
    """Entries that are not relative to $ORIGIN (or ${ORIGIN})."""
    return [e for e in entries if not (e.startswith("$ORIGIN") or e.startswith("${ORIGIN}"))]


def needed_libraries(readelf_dynamic: str) -> list[str]:
    return NEEDED.findall(readelf_dynamic)


def missing(needed: list[str], shipped: set[str]) -> list[str]:
    """Needed libraries that are neither shipped nor of the base system, in order."""
    return [n for n in needed if n not in shipped and n not in BASE_SYSTEM]


def readelf_dynamic(program: str) -> str:
    return subprocess.run(["readelf", "-d", program], capture_output=True, text=True, check=True).stdout


def check_runpath(program: str) -> bool:
    bad = bad_runpath(runpath_entries(readelf_dynamic(program)))
    for entry in bad:
        print(f"release: {program} has the RUNPATH entry {entry}, which is not relative to $ORIGIN", file=sys.stderr)
    return not bad


def main(argv: list[str]) -> int:
    if argv[:1] == ["--runpath-only"] and len(argv) >= 2:
        return 0 if all([check_runpath(p) for p in argv[1:]]) else 1
    if len(argv) < 2:
        print("usage: release_needed.py PROGRAM SHIPPED_NAME... | --runpath-only PROGRAM...", file=sys.stderr)
        return 2
    absent = missing(needed_libraries(readelf_dynamic(argv[0])), set(argv[1:]))
    for name in absent:
        print(f"release: {argv[0]} needs {name}, which is neither shipped nor a base-system library", file=sys.stderr)
    return 1 if absent or not check_runpath(argv[0]) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
