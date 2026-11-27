#!/usr/bin/env python3
"""Compare the gfx1200 machine code of two engine builds kernel by kernel.

A refactor that only moves or renames host code must leave every GPU kernel unchanged. This collects all kernels from
the HIP objects of each build (whatever file they ended up in), strips addresses, and reports kernels that were added,
removed or changed, including register counts and LDS size from the code object metadata.

Usage: scripts/isa-diff.py [--rename REGEX=REPLACEMENT ...] BUILD_A BUILD_B   (e.g. work/build/base work/build/r5)
Exit status 0 when all kernels are identical.

A change that renames a kernel (a template parameter replaced by another type) would show as removed plus added. With
--rename the names of BUILD_A are demangled (llvm-cxxfilt), rewritten by each rule in order (re.sub) and compared
with the demangled names of BUILD_B; the instruction lists must still be equal. The rules are part of what you claim:
state them next to the result.
"""

import re
import subprocess
import sys
import tempfile
from pathlib import Path

LLVM = (
    Path(__file__).resolve().parent.parent / "work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_core/lib/llvm/bin"
)
TARGET = "hipv4-amdgcn-amd-amdhsa--gfx1200"
META_KEYS = (
    "vgpr_count",
    "sgpr_count",
    "group_segment_fixed_size",
    "private_segment_fixed_size",
    "vgpr_spill_count",
    "sgpr_spill_count",
    "kernarg_segment_size",
)


def tool(name, *args):
    return subprocess.run([str(LLVM / name), *args], check=True, capture_output=True, text=True).stdout


def code_object(obj, tmp):
    """Extracts the gfx1200 code object from a HIP object file, or None if it has no device code."""
    if ".hip_fatbin" not in tool("llvm-readelf", "-S", str(obj)):
        return None
    fatbin, co = tmp / "fatbin.bin", tmp / f"{obj.name}.co"
    tool("llvm-objcopy", "-O", "binary", "--only-section=.hip_fatbin", str(obj), str(fatbin))
    tool(
        "clang-offload-bundler", "--unbundle", "--type=o", f"--input={fatbin}", f"--targets={TARGET}", f"--output={co}"
    )
    return co


def kernels(co):
    """Maps each function symbol to its normalized instruction list."""
    out, name = {}, None
    for line in tool("llvm-objdump", "-d", "--no-show-raw-insn", "--no-leading-addr", str(co)).splitlines():
        if m := re.match(r"^<(.+)>:$", line):
            name = m.group(1)
            out[name] = []
        elif name and line.startswith("\t") and not line.strip().startswith("..."):
            out[name].append(line.split("//")[0].strip())
    return out


def metadata(co):
    """Maps each kernel name to the resource fields that affect occupancy. The keys of a kernel record are sorted, so
    some (group_segment_fixed_size, kernarg_segment_size) come before its .name: collect the whole record first."""
    out, record = {}, None
    for line in [*tool("llvm-readelf", "--notes", str(co)).splitlines(), "  - .end:"]:
        if m := re.match(r"^  - \.(\w+):\s*(.*)$", line):  # a new kernel record
            if record and "name" in record:
                out[record.pop("name")] = record
            record = {}
        elif not (m := re.match(r"^    \.(\w+):\s*(.*)$", line)) or record is None:
            continue
        key, value = m.groups()
        if key == "name" or key in META_KEYS:
            record[key] = value.strip()
    return out


def built_objects(build):
    """Object files the build still produces; stale ones from removed sources would shadow renamed kernels."""
    out = subprocess.run(
        ["ninja", "-C", str(build), "-t", "targets", "all"], check=True, capture_output=True, text=True
    )
    objects = (Path(build) / line.split(":")[0] for line in out.stdout.splitlines())
    return sorted(o for o in objects if o.suffix == ".o" and o.is_file())


def collect(build):
    code, meta = {}, {}
    with tempfile.TemporaryDirectory() as t:
        for obj in built_objects(build):
            co = code_object(obj, Path(t))
            if co is None:
                continue
            for k, v in kernels(co).items():
                # A device function that every translation unit compiles (a noinline helper in a shared header) has
                # one body per unit, and they differ; all distinct bodies count, not the one of the object seen first
                bodies = code.setdefault(k, [])
                if v not in bodies:
                    bodies.append(v)
            for k, v in metadata(co).items():
                meta.setdefault(k, v)
    return code, meta


def demangled(table, rules):
    """The table with demangled names, rewritten by the rules (REGEX, REPLACEMENT)."""
    names = list(table)
    out = subprocess.run(
        [str(LLVM / "llvm-cxxfilt")], input="\n".join(names), check=True, capture_output=True, text=True
    ).stdout.splitlines()
    for regex, replacement in rules:
        out = [re.sub(regex, replacement, name) for name in out]
    return dict(zip(out, table.values(), strict=True))


def parse_args(argv):
    rules, builds = [], []
    i = 0
    while i < len(argv):
        if argv[i] == "--rename" and i + 1 < len(argv) and "=" in argv[i + 1]:
            rules.append(tuple(argv[i + 1].split("=", 1)))
            i += 2
        else:
            builds.append(argv[i])
            i += 1
    return rules, builds


def main():
    rules, builds = parse_args(sys.argv[1:])
    if len(builds) != 2:
        sys.exit(__doc__)
    (code_a, meta_a), (code_b, meta_b) = collect(builds[0]), collect(builds[1])
    if rules:
        code_a, meta_a = demangled(code_a, rules), demangled(meta_a, rules)
        code_b, meta_b = demangled(code_b, []), demangled(meta_b, [])
    problems = [f"removed: {k}" for k in sorted(code_a.keys() - code_b.keys())]
    problems += [f"added: {k}" for k in sorted(code_b.keys() - code_a.keys())]
    problems += [
        f"code differs: {k}" for k in sorted(code_a.keys() & code_b.keys()) if sorted(code_a[k]) != sorted(code_b[k])
    ]
    problems += [
        f"resources differ: {k}: {meta_a[k]} -> {meta_b[k]}"
        for k in sorted(meta_a.keys() & meta_b.keys())
        if meta_a[k] != meta_b[k]
    ]
    print(f"{len(code_a)} functions in A, {len(code_b)} in B")
    print("\n".join(problems) if problems else "all kernels identical")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
