#!/usr/bin/env python3
"""Print the register profile of the gfx1200 kernels of an engine build.

Spills decide how fast the persistent megakernel runs, so this lists VGPRs, SGPRs, spills and scratch per kernel from
the code object metadata of the HIP objects (same tools as scripts/isa-diff.py).

Usage: scripts/kernel-regs.py BUILD_DIR_OR_OBJECT [NAME_FILTER]   (e.g. scripts/kernel-regs.py work/build/w66 k_mega)
"""

import importlib.util
import re
import sys
import tempfile
from pathlib import Path

_spec = importlib.util.spec_from_file_location("isa_diff", Path(__file__).resolve().parent / "isa-diff.py")
isa = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(isa)

COLUMNS = ("vgpr_count", "sgpr_count", "vgpr_spill_count", "sgpr_spill_count", "private_segment_fixed_size")
WIDTHS = (6, 6, 8, 8, 9)


def demangle(name):
    if not name.startswith("_Z"):
        return name
    return re.sub(r"^void tona::gpu::\(anonymous namespace\)::", "", isa.tool("llvm-cxxfilt", name).strip())


def profile(build):
    out = {}
    objects = [Path(build)] if Path(build).is_file() else isa.built_objects(build)
    with tempfile.TemporaryDirectory() as t:
        for obj in objects:
            co = isa.code_object(obj, Path(t))
            if co is not None:
                for k, v in isa.metadata(co).items():
                    out.setdefault(k, v)
    return out


def main():
    if len(sys.argv) not in (2, 3):
        sys.exit(__doc__)
    pattern = re.compile(sys.argv[2] if len(sys.argv) == 3 else ".")
    rows = {demangle(k): v for k, v in profile(sys.argv[1]).items()}
    print(f"{'kernel':<48}{'VGPR':>6}{'SGPR':>6}{'vspill':>8}{'sspill':>8}{'scratch':>9}")
    for name in sorted(rows):
        if pattern.search(name):
            r = rows[name]
            short = name.split("(")[0]
            cells = "".join(f"{r.get(c, '-'):>{w}}" for c, w in zip(COLUMNS, WIDTHS, strict=True))
            print(f"{short[:47]:<48}{cells}")


if __name__ == "__main__":
    main()
