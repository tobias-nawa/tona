"""Isolation gate: shared code stays free of per-architecture branches.

Usage: arch_isolation.py [FILE...]  (no arguments: all tracked files)

This is check C6 of docs/architecture/model-architectures.md. Three rules keep a model architecture from leaking into
code that every architecture shares:

1. Kernel headers (engine/gpu/mega_op_*.h, matvec_*.h, mega_kernel.h) do not name `ArchId` and do not include an
   architecture header (anything under arch/, the model layer's arch.h, model_base.h and the per-architecture model
   headers). A kernel decides by its profile (a template parameter), never by asking which model runs: the Coder's
   machine code must not change when another architecture is added.
2. Architecture directories (engine/gpu/arch/<name>/) do not include the headers of another architecture directory.
3. `switch (..arch..)` and `case ArchId::..` appear only where an architecture is mapped to its code: `model_registry.*`
   and `arch.*` in engine/model/, the backend registries, files named `*factory*`, and the two places of the HIP
   backend that pick the architecture's program: `make_arch_program` in gpu/mega_build.cpp and `prefill_sequence` in
   gpu/prefill_run.cpp. Everywhere else the branch belongs into the architecture's own module, behind an interface.

Comments are ignored (they may explain the rule). There is no exception list: move the branch.
"""

import re
import sys

from common import CPP_SUFFIXES, fail_if_empty, gate_files, repo_root, report

KERNEL_HEADER = re.compile(r"^engine/gpu/(mega_op_[^/]+|matvec_[^/]+|mega_kernel)\.h$")
ARCH_DIR = re.compile(r"^engine/gpu/arch/([^/]+)/")
# Where an architecture is mapped to its code: the one `switch` of the model layer, the registries, the factories, the
# HIP backend's choice of program builder and prefill sequence (host, cold path)
MAPPING_FILE = re.compile(
    r"^engine/model/(model_registry|arch)\.[^/]+$|^engine/backend/registry\.[^/]+$|/[^/]*factory[^/]*$"
    r"|^engine/gpu/(mega_build|prefill_run)\.cpp$"
)
# Headers of the model layer that belong to one architecture (or to the architecture mechanism itself)
ARCH_MODEL_HEADERS = {"arch.h", "model_base.h", "qwen4exp.h", "qwen35moe.h", "nemotron_h_moe.h"}

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
SWITCH = re.compile(r"\bswitch\s*\(([^)]*)\)")
SWITCH_OVER_ARCH = re.compile(r"\barch\b|ArchId")
CASE_ARCH = re.compile(r"\bcase\s+(?:\w+::)*ArchId::")
ARCH_ID = re.compile(r"\bArchId\b")
ARCH_OF_PATH = re.compile(r"(?:^|/)arch/([^/]+)/")


def strip_comments(text: str) -> str:
    """Blanks comments but keeps the line breaks, so the reported line numbers stay right."""
    return COMMENT.sub(lambda m: "\n" * m.group(0).count("\n"), text)


def line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def is_arch_header(path: str) -> bool:
    return "arch/" in path or path.rsplit("/", 1)[-1] in ARCH_MODEL_HEADERS


def kernel_header_problems(f: str, code: str) -> list[str]:
    problems = [
        f"{f}:{line_of(code, m.start())}: kernel header names ArchId (decide by the profile)"
        for m in ARCH_ID.finditer(code)
    ]
    problems += [
        f"{f}:{line_of(code, m.start())}: kernel header includes {m.group(1)} (an architecture header)"
        for m in INCLUDE.finditer(code)
        if is_arch_header(m.group(1))
    ]
    return problems


def arch_dir_problems(f: str, code: str, own: str) -> list[str]:
    problems = []
    for m in INCLUDE.finditer(code):
        other = ARCH_OF_PATH.search(m.group(1))
        if other and other.group(1) != own:
            problems.append(
                f"{f}:{line_of(code, m.start())}: includes {m.group(1)} of architecture {other.group(1)} (own: {own})"
            )
    return problems


def switch_problems(f: str, code: str) -> list[str]:
    problems = [
        f"{f}:{line_of(code, m.start())}: switch over the architecture outside the mapping files"
        for m in SWITCH.finditer(code)
        if SWITCH_OVER_ARCH.search(m.group(1))
    ]
    problems += [
        f"{f}:{line_of(code, m.start())}: case ArchId:: outside the mapping files" for m in CASE_ARCH.finditer(code)
    ]
    return problems


def file_problems(f: str, text: str) -> list[str]:
    code = strip_comments(text)
    problems = []
    if KERNEL_HEADER.match(f):
        problems += kernel_header_problems(f, code)
    own = ARCH_DIR.match(f)
    if own:
        problems += arch_dir_problems(f, code, own.group(1))
    if not MAPPING_FILE.search(f):
        problems += switch_problems(f, code)
    return problems


def main() -> int:
    root = repo_root()
    files, explicit = gate_files(sys.argv[1:], CPP_SUFFIXES)
    files = [f for f in files if f.startswith("engine/")]
    if not explicit:
        fail_if_empty(files, "arch-isolation")
    problems = []
    for f in files:
        with open(root / f, encoding="utf-8", errors="replace") as fh:
            problems += file_problems(f, fh.read())
    return report(
        "arch-isolation",
        problems,
        len(files),
        "Move the per-architecture branch into the architecture's module (docs/architecture/model-architectures.md).",
    )


if __name__ == "__main__":
    sys.exit(main())
