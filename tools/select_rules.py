"""Path rules of tools/select_checks.py: what the build graph cannot know. Files that steer the build or the golden
runs trigger everything, documents trigger only the gates, and every engine file gets a change class of
docs/contribute/measurement.md. An unknown case is never settled downwards: it triggers everything."""

import re
from dataclasses import dataclass, field

GOLDEN_BINARIES = {
    "tona-gpu-gen": ["gpu", "cpu plan", "prefill", "prefill dflash"],
    "tona-server": ["server"],
}
ALL_GOLDENS = ["gpu", "cpu plan", "prefill", "prefill dflash", "server"]

# Files outside the build graph that steer the build or feed the golden runs (the former ENGINE_PATHS of
# scripts/gpu-status.sh, which now has no list of its own)
TRIGGER_FILES = {
    "scripts/golden.sh",
    "scripts/golden-server.sh",
    "scripts/bench-long-code.txt",  # prompt of the prefill variants and of the server golden
    "scripts/build-mtp-vocab.py",  # generates engine/model/mtp_vocab.inc
    "tools/gen_tables.py",  # generates engine/model/quant_tables.h and unicode_tables.h
}
TRIGGER_PREFIXES = ("toolchain/",)  # CMake files are compared by tools/select_cmake.py
DOC_PREFIXES = ("docs/", "research/", "licenses/", ".github/")
DOC_FILES = {"LICENSE", ".gitignore"}
CONFIG_FILES = {
    ".pre-commit-config.yaml", "ruff.toml", "pyproject.toml", ".clang-format", ".clang-tidy", ".clangd", "justfile",
}  # fmt: skip
SCRIPT_PREFIXES = ("scripts/", "tools/", "bench/")
# Headers and expected tokens of the mini models: read only by tools/mini_token_check.py, never by an engine build
TEST_DATA_PREFIXES = ("data/mini-models/",)

# Engine files off the hot path (C2): everything else in the engine counts as hot path (C3), the higher class
OFF_HOT_PATH = re.compile(
    r"engine/(update/|tools/tona_(top|chat|gguf|model|tokenize|ref|ple|sfx|spec|wide|expert|gemm)"
    r"|server/(api|routes_|chat|http_util|request_options|system_status|server_main))"
)
COMPUTE_PATHS = re.compile(r"engine/(gpu/|cpu/|model/quant)")  # code whose numbers decide the output
# File name parts that mark long-context state and memory pressure paths; a name part, so that `mapping` is no `pin`
LONG_CONTEXT = {
    "rope", "yarn", "idx", "topk", "kv", "cache", "seq", "stash", "checkpoint", "ctx", "context", "budget", "plan",
    "attn", "dattn",
}  # fmt: skip
PRESSURE = {
    "pinned", "pin", "stall", "alloc", "upload", "budget", "guarded", "uring", "wait", "abort", "prefetch", "stash",
    "checkpoint", "omp",
}  # fmt: skip
STASH = {"stash", "cache", "checkpoint", "prompt"}

RECIPES = {
    "C0": ["R0"],
    "C1": ["R0", "R1"],
    "C2": ["R0", "R1", "R3"],
    "C3": ["R0", "R1", "R3", "R5"],
    "C5": ["R0", "R1", "R3", "R4", "R5"],
}
RECIPE_NAMES = {
    "R0": "gates (just check)",
    "R1": "unit tests (just test-cpu; just test-py when Python changed)",
    "R2": "kernel identity (scripts/isa-diff.py base build against yours)",
    "R3": "golden checks (those the build graph reaches)",
    "R4": "bit-exact (scripts/ab.py, TONA_DETERMINISTIC=1, workloads mid and long)",
    "R5": "speed (scripts/ab.py, workloads of the affected rates)",
    "R7": "long-context run (scripts/ab.py workload huge, bench/needle --hard)",
    "R8": "pressure series (20 requests, 10 under page-cache pressure, TONA_DIAG=1)",
}
CLASS_NAMES = {
    "C0": "docs, research, comments",
    "C1": "scripts, tools, bench, tests that feed no golden check",
    "C2": "host code off the hot path",
    "C3": "host code on the hot path",
    "C5": "device code or unknown: assume the machine code changes",
    "C7": "long context",
    "C8": "stability: stalls, memory pressure, recovery",
}


@dataclass
class Verdict:
    """What one changed file demands."""

    path: str
    why: str
    cls: str = "C1"
    everything: bool = False
    goldens: list[str] = field(default_factory=list)
    unit: list[str] = field(default_factory=list)  # ctest names
    unit_all: bool = False
    python: bool = False
    installer: bool = False
    compute: bool = False
    long_context: bool = False
    pressure: bool = False
    stash: bool = False
    via: list[str] = field(default_factory=list)  # the binaries it goes into


def parts(path: str) -> set[str]:
    return set(re.split(r"[_./-]+", path.rsplit("/", 1)[-1].lower()))


def everything(path: str, why: str) -> Verdict:
    return Verdict(path, why, cls="C5", everything=True, goldens=list(ALL_GOLDENS), unit_all=True, compute=True)


def outside_engine(path: str) -> Verdict | None:
    """Rules for files that are not engine sources; None when the path belongs to the engine."""
    if path in TRIGGER_FILES or path.startswith(TRIGGER_PREFIXES):
        return everything(path, "steers the build or feeds the golden runs")
    name = path.rsplit("/", 1)[-1]
    if (name == "CMakeLists.txt" or name.endswith(".cmake")) and not path.startswith(("installer/", "engine/")):
        return everything(path, "build definition outside the engine")
    if path.startswith(DOC_PREFIXES) or path in DOC_FILES or name.endswith(".md"):
        return Verdict(path, "documentation or repository metadata: gates only", cls="C0")
    if path in CONFIG_FILES or path.startswith(SCRIPT_PREFIXES + TEST_DATA_PREFIXES):
        return Verdict(path, "script or tool that feeds no golden check: gates and Python tests", cls="C1", python=True)
    if path.startswith("installer/"):
        return Verdict(path, "installer: its own build and tests, no engine output", cls="C2", installer=True)
    if not path.startswith("engine/"):
        return everything(path, "unknown kind of file")
    return None


def extras(verdict: Verdict, path: str) -> Verdict:
    names = parts(path)
    verdict.compute = verdict.compute or bool(COMPUTE_PATHS.match(path))
    verdict.long_context = bool(names & LONG_CONTEXT)
    verdict.pressure = bool(names & PRESSURE)
    verdict.stash = bool(names & STASH) and path.startswith("engine/server/")
    return verdict


def engine_class(path: str, device: bool) -> str:
    if path.startswith("engine/tests/"):
        return "C1"
    if device:
        return "C5"
    return "C2" if OFF_HOT_PATH.match(path) else "C3"
