"""Shared helpers for the code gates: which files count as source, and how a gate fails."""

import os
import subprocess
import sys
from pathlib import Path

# Vendored code is not ours to reshape; everything else is.
EXCLUDED_PREFIXES = ("engine/third_party/",)

CPP_SUFFIXES = (".cpp", ".h", ".hpp", ".hip", ".cu")
# Size gates also cover the Metal backend; clang-tidy does not (its compile database is Linux only)
SIZE_SUFFIXES = (*CPP_SUFFIXES, ".mm", ".metal")
SCRIPT_SUFFIXES = (".py", ".sh")


def repo_root() -> Path:
    out = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True, check=True)
    return Path(out.stdout.strip())


def tracked_files(suffixes: tuple[str, ...]) -> list[str]:
    """Tracked and staged-new files with one of the suffixes, relative to the repo root."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root(),
    )
    files = [f for f in out.stdout.splitlines() if f.endswith(suffixes)]
    return sorted(f for f in files if not f.startswith(EXCLUDED_PREFIXES))


def gate_files(argv: list[str], suffixes: tuple[str, ...]) -> tuple[list[str], bool]:
    """Files to check and whether they were given explicitly.

    Without arguments: every tracked source file. With arguments (pre-commit passes the changed files): only those,
    reduced to the suffixes and exclusions of the gate, as paths relative to the repo root.
    """
    if not argv:
        return tracked_files(suffixes), False
    root = repo_root().resolve()
    files = []
    for arg in argv:
        path = Path(arg).resolve()
        rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else arg
        if rel.endswith(suffixes) and not rel.startswith(EXCLUDED_PREFIXES) and path.is_file():
            files.append(rel)
    return sorted(set(files)), True


def fail_if_empty(files: list[str], gate: str) -> None:
    # A gate that found nothing to check must not report success.
    if not files:
        print(f"{gate}: no files found to check", file=sys.stderr)
        sys.exit(2)


def report(gate: str, problems: list[str], checked: int, hint: str) -> int:
    if not problems:
        print(f"{gate}: ok ({checked} checked)")
        return 0
    for p in problems:
        print(p, file=sys.stderr)
    print(f"{gate}: {len(problems)} violation(s). {hint}", file=sys.stderr)
    return 1


# Pull requests of these accounts carry no template, a capitalised title and a bot sign-off; their commits are
# squashed away and their text is generated, so the gates that read the text or the commits skip them.
BOT_AUTHORS = ("dependabot[bot]",)


def is_bot_pull_request() -> bool:
    return os.environ.get("PR_AUTHOR", "") in BOT_AUTHORS
