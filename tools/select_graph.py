"""The build graph behind tools/select_checks.py: which source files go into which binary, and which ctest test runs
which binary. Ninja knows every input of a binary, headers included (from its dependency log), so the graph decides
what a change can reach; no hand-kept path list does."""

import json
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path


class GraphError(Exception):
    """The graph cannot be read or looks incomplete; the caller must then check everything."""


@dataclass
class Graph:
    root: Path  # the source tree the build was configured from
    targets: dict[str, frozenset[str]]  # executable -> source files (repository-relative) that go into it
    device: frozenset[str]  # files that go into a device-code translation unit (.hip and what it includes)
    tests: dict[str, tuple[str, bool]] = field(default_factory=dict)  # ctest name -> (executable, is unit test)

    def reaching(self, path: str) -> list[str]:
        return sorted(name for name, files in self.targets.items() if path in files)

    def known(self, path: str) -> bool:
        return any(path in files for files in self.targets.values())


def run(cmd: list[str]) -> str:
    try:
        done = subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise GraphError(f"{' '.join(cmd)} failed: {error}") from error
    return done.stdout


def source_root(build: Path) -> Path:
    cache = build / "CMakeCache.txt"
    try:
        text = cache.read_text()
    except OSError as error:
        raise GraphError(f"{cache} is missing: no configured build") from error
    match = re.search(r"^CMAKE_HOME_DIRECTORY:INTERNAL=(.+)$", text, re.MULTILINE)
    if not match:
        raise GraphError(f"{cache} names no source directory")
    # The engine is configured from <repository>/engine
    return Path(match.group(1)).parent


def relative_sources(lines: str, root: Path) -> frozenset[str]:
    """The inputs inside the repository, as repository-relative paths; build outputs and system files drop out."""
    found = set()
    prefix = str(root) + "/"
    for line in lines.splitlines():
        # work/ holds the toolchain and builds of the machine, not the repository
        if line.startswith(prefix) and not line.startswith(prefix + "work/"):
            found.add(line[len(prefix) :])
    return frozenset(found)


def executables(build: Path) -> list[str]:
    out = run(["ninja", "-C", str(build), "-t", "targets", "all"])
    return [line.split(":")[0] for line in out.splitlines() if "EXECUTABLE_LINKER" in line.split(":", 1)[-1]]


def object_deps(deps_text: str, root: Path) -> dict[str, frozenset[str]]:
    """Per object file, the repository files it was compiled from, from `ninja -t deps` (the dependency log; blocks:
    `object: #deps N ...`, then the indented files). `ninja -t inputs` does not show these headers."""
    deps: dict[str, set[str]] = {}
    current: set[str] = set()
    for line in deps_text.splitlines():
        if line and not line[0].isspace():
            current = deps.setdefault(line.split(":", 1)[0], set())
        else:
            current |= relative_sources(line.strip(), root)
    return {obj: frozenset(files) for obj, files in deps.items()}


def target_files(inputs: str, root: Path, deps: dict[str, frozenset[str]]) -> frozenset[str]:
    files = set(relative_sources(inputs, root))
    for line in inputs.splitlines():
        files |= deps.get(line, frozenset())
    return frozenset(files)


def ctest_tests(build: Path) -> dict[str, tuple[str, bool]]:
    data = json.loads(run(["ctest", "--test-dir", str(build), "--show-only=json-v1"]))
    tests = {}
    for test in data.get("tests", []):
        command = test.get("command") or [""]
        labels = [p["value"] for p in test.get("properties", []) if p["name"] == "LABELS"]
        tests[test["name"]] = (Path(command[0]).name, bool(labels and "unit" in labels[0]))
    return tests


def check_complete(targets: dict[str, frozenset[str]]) -> None:
    """Without the dependency log the headers are missing, and a header change would reach nothing."""
    if not targets:
        raise GraphError("the build has no executable targets")
    if not any(f.endswith((".h", ".hpp", ".inc")) for files in targets.values() for f in files):
        raise GraphError("no header in the build graph: the build has no dependency log, build it first")


def load(build: Path) -> Graph:
    if not build.is_dir() or not shutil.which("ninja"):
        raise GraphError(f"no build directory {build} or no ninja")
    root = source_root(build)
    deps = object_deps(run(["ninja", "-C", str(build), "-t", "deps"]), root)
    targets = {}
    for name in executables(build):
        targets[name] = target_files(run(["ninja", "-C", str(build), "-t", "inputs", name]), root, deps)
    check_complete(targets)
    device = frozenset().union(*(files for obj, files in deps.items() if obj.endswith(".hip.o")))
    return Graph(root, targets, device, ctest_tests(build))
