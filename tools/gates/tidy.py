"""Lint gate: clang-tidy (.clang-tidy) on the C++/HIP/Objective-C++ files changed against a base.

Usage: tidy.py [--all] [--base REF] [--engine-db DIR] [--installer-db DIR] [-j N] [FILE...]
Without FILE: the files changed against the merge base with REF (default origin/main, else main), including
uncommitted and untracked ones. Findings in files that were not changed (headers a changed file includes) do not count.
A changed header is linted through the sources that include it (it needs their context, e.g. the HIP macros).

Needs a clang-tidy and a compile database. Linux: the ROCm toolchain's (it parses HIP). macOS: the wheel of
tools/gates/requirements-tidy-macos.txt in work/venv-tidy (`just lint` sets it up), with the SDK from xcrun. The
database comes from configuring only, no build: `source toolchain/env.sh && cmake -S engine -B work/build/engine -G
Ninja` (macOS: without env.sh, with -DTONA_BACKEND_METAL=ON). A missing toolchain or database is an error, never a
silent pass.

--system (CI without ROCm): the clang-tidy and run-clang-tidy of the PATH, and only the translation units of the given
compile database, which then holds the host-only sources (the engine configured without the toolchain, the installer).
Sources and headers that only the GPU build compiles are not linted there; `just lint` lints them with the toolchain.
The system clang-tidy has another major version than the toolchain's, so a finding that only one of them reports is
looked at in `just lint`, which decides.
"""

import argparse
import json
import platform
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import NamedTuple, NoReturn

from common import CPP_SUFFIXES, EXCLUDED_PREFIXES, repo_root

# Glob, not a fixed Python version: the venv takes the python3 of whoever ran scripts/setup.sh (3.12 on Ubuntu 24.04)
LLVM_BIN_GLOB = "work/rocm-10.1/lib/python*/site-packages/_rocm_sdk_core/lib/llvm/bin"
# macOS: the wheel's own binaries (its bin/ entries are Python wrappers); the Python version is the venv's
MACOS_LLVM_GLOB = "work/venv-tidy/lib/python*/site-packages/clang_tidy/data/bin"
SYSTEM = platform.system()
TIDY_SUFFIXES = (*CPP_SUFFIXES, ".mm")
MAX_JOBS = 4
# Source tree -> directory with its compile_commands.json (configure only: cmake -S <tree> -B <dir> -G Ninja).
DATABASES = {"engine/": "work/build/engine", "installer/": "work/build/installer"}
INCLUDE = re.compile(r'^\s*#\s*include\s+"([^"]+)"', re.MULTILINE)
HEADER_SUFFIXES = (".h", ".hpp")
FINDING = re.compile(r"^(/\S+?):\d+:\d+: (?:warning|error): ")


def die(message: str) -> NoReturn:
    # Exit 2 = the gate could not run; 1 = it ran and found something.
    print(message, file=sys.stderr)
    sys.exit(2)


def git(*args: str) -> list[str]:
    out = subprocess.run(["git", *args], capture_output=True, text=True, check=True, cwd=repo_root())
    return out.stdout.splitlines()


def pick_base(requested: str | None) -> str:
    candidates = [requested] if requested else ["origin/main", "main"]
    for ref in candidates:
        probe = subprocess.run(
            ["git", "rev-parse", "--verify", "-q", ref], capture_output=True, cwd=repo_root(), check=False
        )
        if probe.returncode == 0:
            return ref
    die(f"tidy: base {' / '.join(candidates)} not found; pass --base REF")


def changed_files(base: str) -> list[str]:
    merge_base = git("merge-base", base, "HEAD")[0]
    names = git("diff", "--name-only", "--diff-filter=ACMR", merge_base)
    names += git("ls-files", "--others", "--exclude-standard")
    return sorted(set(names))


# Sources that only build on one platform: their compile database exists only there (the Metal backend needs
# Apple's toolchain; tona-bench and tona-width-nll link it), so this platform can neither configure nor lint them.
# The reference backend and the Linux-only sources (io_uring, AVX-512 experts, the llama.cpp comparison) in turn
# build only on Linux.
PLATFORM_ONLY = {
    "Linux": (
        "engine/gpu/",
        "engine/io/uring.cpp",
        "engine/cpu/experts.cpp",
        "engine/cpu/experts_decode.h",
        "engine/tests/attn_wmma_ref.h",
        "engine/tools/tona_hip_check.h",
        "engine/tests/test_dequant_ggml.cpp",
        "engine/tests/test_gpu_",
        "engine/tools/tona_gpu_",
        "engine/tools/tona_expert_check",
        "engine/tools/tona_gemm_",
        "engine/tools/tona_wide_probe",
    ),
    "Darwin": (
        "engine/backend/metal/",
        "engine/cpu/experts_unsupported.cpp",
        "engine/io/uring_pread.cpp",
        "engine/tools/tona_bench.cpp",
        "engine/tools/tona_width_nll.cpp",
    ),
}


def other_platform(system: str) -> tuple[str, ...]:
    """Prefixes of the sources this platform does not build: what is built only elsewhere."""
    return PLATFORM_ONLY["Linux" if system == "Darwin" else "Darwin"]


OTHER_PLATFORM = other_platform(SYSTEM)


class Toolchain(NamedTuple):
    clang_tidy: Path
    run_tidy: list[str]  # command that starts run-clang-tidy
    extra: list[str]  # arguments for both (the macOS SDK)


def system_toolchain() -> Toolchain:
    tidy, runner = shutil.which("clang-tidy"), shutil.which("run-clang-tidy")
    if not tidy or not runner:
        die("tidy: --system needs clang-tidy and run-clang-tidy on the PATH (Fedora: clang-tools-extra)")
    return Toolchain(Path(tidy), [runner], [])


def toolchain(system: str, root: Path) -> Toolchain:
    """clang-tidy of this platform; exits 2 when it is missing (lint is never skipped silently)."""
    if system != "Darwin":
        found_llvm = sorted(root.glob(LLVM_BIN_GLOB))
        llvm = found_llvm[-1] if found_llvm else root / "work/rocm-10.1"
        if not (llvm / "run-clang-tidy").is_file():
            die(
                "tidy: ROCm toolchain missing (work/rocm-10.1). Run scripts/setup.sh (--no-gpu without an AMD GPU); "
                "lint is not skipped silently."
            )
        return Toolchain(llvm / "clang-tidy", [str(llvm / "run-clang-tidy")], [])
    found = sorted(root.glob(MACOS_LLVM_GLOB))
    if not found or not (found[-1] / "clang-tidy").is_file():
        die(
            "tidy: clang-tidy missing (work/venv-tidy). Run just lint, which sets it up from "
            "tools/gates/requirements-tidy-macos.txt; lint is not skipped silently."
        )
    llvm = found[-1]
    # The compile database of Apple clang names no SDK; the wheel's clang-tidy needs it for the system headers
    sdk = subprocess.run(["xcrun", "--show-sdk-path"], capture_output=True, text=True, check=True).stdout.strip()
    return Toolchain(
        llvm / "clang-tidy", [sys.executable, str(llvm / "run-clang-tidy.py")], [f"-extra-arg=-isysroot{sdk}"]
    )


def lintable(names: list[str]) -> list[str]:
    root = repo_root()
    files = [
        n
        for n in names
        if n.endswith(TIDY_SUFFIXES)
        and n.startswith(tuple(DATABASES))
        and not n.startswith(EXCLUDED_PREFIXES)
        and (root / n).is_file()
    ]
    other = [n for n in files if n.startswith(OTHER_PLATFORM)]
    if other:
        print(f"tidy: {len(other)} file(s) of another platform not linted here: {', '.join(other)}", file=sys.stderr)
    return [n for n in files if n not in other]


def database_for(name: str, overrides: dict[str, str | None]) -> str:
    prefix = next(p for p in DATABASES if name.startswith(p))
    return overrides.get(prefix) or DATABASES[prefix]


def database_files(db: str) -> set[str]:
    path = repo_root() / db / "compile_commands.json"
    if not path.is_file():
        die(f"tidy: {path} missing. Configure only: source toolchain/env.sh && cmake -S <tree> -B {db} -G Ninja")
    return {entry["file"] for entry in json.loads(path.read_text())}


def check_database(db: str, sources: list[str]) -> None:
    path = repo_root() / db / "compile_commands.json"
    known = database_files(db)
    if not any(k.startswith(str(repo_root()) + "/") for k in known):
        die(f"tidy: {path} belongs to another checkout; configure it here (cmake -S ... -B {db})")
    for s in sources:
        # Exact path: a database from another checkout (a worktree shares work/) would lint nothing and pass.
        if str(repo_root() / s) not in known:
            die(f"tidy: {s} is not in {path}; configure again (cmake -S ... -B {db})")


def tidy_header(tc: Toolchain, db: str, header: str, header_filter: str) -> str:
    cmd = [str(tc.clang_tidy), "-p", db, "-quiet", *tc.extra, f"-header-filter={header_filter}", header]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=repo_root(), check=False)
    return out.stdout + out.stderr


def tidy_sources(tc: Toolchain, db: str, sources: list[str], header_filter: str, jobs: int) -> str:
    root = repo_root()
    regex = [f"^{re.escape(str(root / s))}$" for s in sources]
    cmd = [*tc.run_tidy, "-clang-tidy-binary", str(tc.clang_tidy), "-p", db, "-quiet", *tc.extra]
    cmd += [f"-j{jobs}", f"-header-filter={header_filter}", *regex]
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=root, check=False)
    return out.stdout + out.stderr


def is_header(name: str) -> bool:
    return name.endswith(HEADER_SUFFIXES)


def resolve_include(includer: str, name: str, tree: str) -> str | None:
    root = repo_root()
    for candidate in (Path(includer).parent / name, Path(tree) / name):
        if (root / candidate).is_file():
            return candidate.as_posix()
    return None


def reverse_includes(tree: str) -> dict[str, set[str]]:
    """Header -> files of the tree that include it directly (quoted includes only)."""
    root = repo_root()
    users: dict[str, set[str]] = {}
    for name in git("ls-files", "--", tree):
        if not name.endswith(CPP_SUFFIXES):
            continue
        for inc in INCLUDE.findall((root / name).read_text(errors="replace")):
            target = resolve_include(name, inc, tree)
            if target:
                users.setdefault(target, set()).add(name)
    return users


def including_sources(header: str, tree: str, users: dict[str, set[str]]) -> set[str]:
    """Sources of the tree that include the header, directly or through other headers."""
    found: set[str] = set()
    seen = {header}
    todo = [header]
    while todo:
        for user in users.get(todo.pop(), ()):
            if user in seen:
                continue
            seen.add(user)
            if is_header(user):
                todo.append(user)
            else:
                found.add(user)
    return found


def config_dir(name: str) -> Path:
    """Directory of the .clang-tidy that governs the file: the nearest one above it."""
    root = repo_root()
    d = (root / name).parent
    while d != root and not (d / ".clang-tidy").is_file():
        d = d.parent
    return d


def file_filter(files: list[str]) -> str:
    return "^(" + "|".join(re.escape(str(repo_root() / f)) for f in files) + ")$"


def tree_of(name: str) -> str:
    return next(p for p in DATABASES if name.startswith(p))


def plan_sources(files: list[str], known: set[str]) -> tuple[list[str], list[str]]:
    """Sources to run clang-tidy on, and headers no source in the database includes (linted on their own).

    A header is linted through the sources that include it: on its own it lacks the context (HIP macros and
    types) its includer provides, and clang-tidy would report parse errors instead of findings. Only includers
    under the same .clang-tidy count, because clang-tidy applies the configuration of the source it runs on.
    """
    root = repo_root()
    sources = {f for f in files if not is_header(f)}
    orphans = []
    users_by_tree: dict[str, dict[str, set[str]]] = {}
    for h in (f for f in files if is_header(f)):
        tree = tree_of(h)
        users = users_by_tree.setdefault(tree, reverse_includes(tree))
        includers = {
            s for s in including_sources(h, tree, users) if str(root / s) in known and config_dir(s) == config_dir(h)
        }
        if includers:
            sources |= includers
        else:
            orphans.append(h)
    return sorted(sources), orphans


def host_only(db: str, files: list[str]) -> list[str]:
    """--system: the files the database compiles (headers go through their includers); the rest is the GPU build's."""
    known = database_files(db)
    root = repo_root()
    kept = [f for f in files if is_header(f) or str(root / f) in known]
    skipped = len(files) - len(kept)
    if skipped:
        print(
            f"tidy: {skipped} source(s) not in {db} (built only with the GPU toolchain) not linted here",
            file=sys.stderr,
        )
    return kept


def run_group(tc: Toolchain, db: str, files: list[str], jobs: int, system: bool = False) -> str:
    if system:
        files = host_only(db, files)
    check_database(db, [f for f in files if not is_header(f)])
    sources, orphans = plan_sources(files, database_files(db))
    if system:
        orphans = []  # a header no host source includes belongs to the GPU build
    by_config: dict[Path, list[str]] = {}
    for src in sources:
        by_config.setdefault(config_dir(src), []).append(src)
    parts = []
    for cfg, group in by_config.items():
        # Header findings only for headers under the same configuration as the sources of this run
        own = [f for f in files if config_dir(f) == cfg]
        parts.append(tidy_sources(tc, db, group, file_filter(own), jobs))
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        parts += pool.map(lambda h: tidy_header(tc, db, h, file_filter([h])), orphans)
    return "\n".join(parts)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--base", help="ref to compare against (default origin/main, else main)")
    ap.add_argument(
        "--engine-db", help=f"directory with the engine's compile_commands.json (default {DATABASES['engine/']})"
    )
    ap.add_argument("--installer-db", help=f"same for installer/ (default {DATABASES['installer/']})")
    ap.add_argument("-j", type=int, default=2, help=f"parallel clang-tidy jobs, at most {MAX_JOBS} (default 2)")
    ap.add_argument(
        "--system", action="store_true", help="clang-tidy from the PATH, host-only sources (CI without ROCm)"
    )
    ap.add_argument("--all", action="store_true", help="every tracked C++/HIP file instead of the changed ones")
    ap.add_argument("files", nargs="*")
    return ap.parse_args()


def own_findings(output: str, changed: set[str]) -> set[str]:
    # Findings in headers that were not changed are somebody else's backlog.
    if "Error while processing" in output:
        print(output, file=sys.stderr)
        die("tidy: clang-tidy could not process a file (see above); not reporting green")
    matches = ((FINDING.match(line), line) for line in output.splitlines())
    return {line for m, line in matches if m and m.group(1) in changed}


def main() -> int:
    args = parse_args()
    root = repo_root()
    tc = system_toolchain() if args.system else toolchain(SYSTEM, root)
    names = args.files or (git("ls-files") if args.all else changed_files(pick_base(args.base)))
    files = lintable([Path(n).resolve().relative_to(root).as_posix() if Path(n).is_absolute() else n for n in names])
    if not files:
        print("tidy: ok (no changed C++/HIP files in engine/ or installer/)")
        return 0
    jobs = max(1, min(args.j, MAX_JOBS))
    overrides = {"engine/": args.engine_db, "installer/": args.installer_db}
    groups: dict[str, list[str]] = {}
    for f in files:
        groups.setdefault(database_for(f, overrides), []).append(f)
    changed = {str(root / f) for f in files}
    findings: set[str] = set()
    for db, group in groups.items():
        findings |= own_findings(run_group(tc, db, group, jobs, args.system), changed)
    for line in sorted(findings):
        print(line, file=sys.stderr)
    if findings:
        print(f"tidy: {len(findings)} finding(s) in {len(files)} changed file(s).", file=sys.stderr)
        return 1
    print(f"tidy: ok ({len(files)} checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
