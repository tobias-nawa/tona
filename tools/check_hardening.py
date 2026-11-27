#!/usr/bin/python3 -I
"""Proves on finished programs that the hardening of engine/cmake/hardening.cmake is in them (a checksec in 100 lines).

Usage: check_hardening.py PROGRAM...

Reads the ELF files with readelf (binutils) and fails, naming program and property, when one is missing:
- PIE: the file is a position-independent executable (ASLR for the image);
- full RELRO: a GNU_RELRO segment and immediate binding (BIND_NOW), so the GOT is read-only once the program runs;
- no executable stack: GNU_STACK without the X flag, and no segment that is writable and executable at once;
- no text relocations (they make code pages writable);
- stack protector: the program imports __stack_chk_fail (some function has a canary);
- FORTIFY_SOURCE: the program imports at least one checked libc function (__memcpy_chk, __snprintf_chk, ...);
- CET (x86-64): the functions start with endbr64 (the code was compiled with -fcf-protection); arm64: BTI landing
  pads. Also on x86-64: the GNU property note marks the whole program IBT and SHSTK capable. The linker writes it only
  when every input has it, so one object without it (the startup objects of compiler-rt, which the HIP link pulled in
  before engine/cmake/hardening.cmake switched it to libgcc) silently drops both marks.

The proof is "this property occurs in the program", the strongest statement a finished binary allows: that every
function has a canary is the compiler's promise, not something a symbol shows. A program whose source has no libc call
with a known buffer size would fail the FORTIFY line; the programs of this repository are far from that.
"""

import re
import subprocess
import sys
from pathlib import Path


def readelf(*args: str) -> str:
    out = subprocess.run(["readelf", "-W", *args], capture_output=True, text=True, check=False)
    if out.returncode != 0:
        raise SystemExit(f"check_hardening: readelf {' '.join(args)}: {out.stderr.strip()}")
    return out.stdout


def machine(header: str) -> str:
    m = re.search(r"Machine:\s+(.*)", header)
    return m.group(1).strip() if m else ""


def check(path: Path) -> list[str]:
    """What is missing in the program, as short phrases; empty when it is hardened."""
    header = readelf("-h", str(path))
    segments = readelf("-l", str(path))
    dynamic = readelf("-d", str(path))
    symbols = readelf("--dyn-syms", str(path))
    notes = readelf("-n", str(path))
    missing = []
    if "DYN (Position-Independent Executable file)" not in header and "PIE" not in dynamic:
        missing.append("not a PIE")
    if "GNU_RELRO" not in segments:
        missing.append("no RELRO segment")
    if not re.search(r"\((FLAGS|FLAGS_1)\)\s+.*\b(BIND_NOW|NOW)\b", dynamic):
        missing.append("lazy binding (no BIND_NOW), RELRO is partial")
    stack = re.search(r"GNU_STACK\s+(?:\S+\s+){5}(\S+)", segments)
    if not stack or "E" in stack.group(1):
        missing.append("executable stack (or no GNU_STACK)")
    if re.search(r"^\s*LOAD\s+(?:\S+\s+){5}RWE", segments, re.MULTILINE):
        missing.append("a segment that is writable and executable")
    if "(TEXTREL)" in dynamic:
        missing.append("text relocations")
    if "__stack_chk_fail" not in symbols:
        missing.append("no stack protector (__stack_chk_fail not imported)")
    if not re.search(r"\b__\w+_chk\b", symbols):
        missing.append("no FORTIFY_SOURCE (no __*_chk import)")
    missing += check_cpu_marks(path, machine(header), notes)
    return missing


def check_cpu_marks(path: Path, arch: str, notes: str) -> list[str]:
    """The CET (x86-64) or BTI (arm64) findings of one program."""
    missing = []
    if "X86-64" in arch and not has_endbr(path):
        missing.append("no CET instrumentation (no endbr64 in the code)")
    if "X86-64" in arch and not ("IBT" in notes and "SHSTK" in notes):
        missing.append("the GNU property note does not mark IBT and SHSTK (an input of the link lacks it)")
    if "AArch64" in arch and "BTI" not in notes:
        missing.append("no BTI marking")
    return missing


def has_endbr(path: Path) -> bool:
    """At least half as many endbr64 as defined functions: one landing pad per function that can be called indirectly.

    A bare "endbr64 occurs" proves nothing, the C runtime's _start has one in any program. The finished programs of the
    repository have more of them than functions (every entry point of a function with several), a program built
    without -fcf-protection a handful.
    """
    out = subprocess.run(
        ["objdump", "-d", "--no-show-raw-insn", str(path)], capture_output=True, text=True, check=False
    )
    pads = out.stdout.count("endbr64")
    functions = sum(
        1 for line in readelf("-s", str(path)).splitlines() if re.search(r"\sFUNC\s", line) and " UND " not in line
    )
    return pads * 2 >= functions > 0


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__, file=sys.stderr)
        return 2
    failed = 0
    for name in argv:
        path = Path(name)
        if not path.is_file():
            print(f"{name}: not found", file=sys.stderr)
            failed += 1
            continue
        missing = check(path)
        if missing:
            failed += 1
            print(f"{name}: NOT hardened: " + "; ".join(missing), file=sys.stderr)
        else:
            print(f"{name}: hardened (PIE, full RELRO, NX, canary, FORTIFY, CET instrumentation)")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
