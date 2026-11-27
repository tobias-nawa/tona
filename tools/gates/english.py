"""English-only gate: no German text outside the allow list, on the changed files.

Usage: english.py [FILE...]  (no arguments: every tracked file)

Everything in the repository is English (AGENTS.md rule 5). A line counts as German when it has an umlaut or two
German function words. Text files only; binary files are skipped.
"""

import re
import sys
from collections.abc import Iterator
from pathlib import Path

from common import repo_root, report, tracked_files

# Vendored and quoted third-party text: not ours, so not scanned.
THIRD_PARTY = ("engine/third_party/", "licenses/", "bench/corpus/")
# Inputs that tests make up (German strings): never real data.
TEST_PATHS = ("engine/tests/", "installer/tests/", "tools/tests/")

URL = re.compile(r"https?://\S+")
# Words that are German and not English (no "die", "der", "das", "mit": they are English words or labels too).
GERMAN_WORDS = re.compile(r"\b(und|nicht|ist|wird|für|eine|oder|auch|nur|kein|keine|sind|werden|wenn|dass)\b")
UMLAUT = re.compile(r"[äöüÄÖÜß]")
# German on purpose. (1) The pinned prompt that every golden is keyed to. (2) German workloads for speed and
# quality runs, and tests of German text handling. (3) One note that quotes the German licence line of the
# installer. Everything else is English, the screens included.
GERMAN_OK_PATHS = (
    "scripts/bench-long-code.txt",
    *THIRD_PARTY,
    *TEST_PATHS,
    "research/2026-11-25-ornith-license.md",
    "bench/chat/prompts.txt",
    "scripts/bench-set.py",
    "scripts/effort_tasks.py",
    "scripts/make_prose_corpus.py",
    "tools/coactivation_drive.py",
)
# This file holds the patterns it searches for.
SELF_FILES = {"tools/gates/english.py"}


def read_text(root: Path, rel: str) -> str | None:
    """The text of a tracked file, or None for binary files (a NUL byte in the first 8 KiB)."""
    path = root / rel
    if path.is_symlink() or not path.is_file():
        return None
    data = path.read_bytes()
    if b"\0" in data[:8192]:
        return None
    return data.decode("utf-8", errors="replace")


def is_german(line: str) -> bool:
    return bool(UMLAUT.search(line)) or len(GERMAN_WORDS.findall(line)) >= 2


def german_lines(rel: str, text: str) -> Iterator[tuple[int, str]]:
    if rel.startswith(GERMAN_OK_PATHS):
        return
    for n, line in enumerate(text.splitlines(), 1):
        if is_german(URL.sub("", line)):
            yield n, line.strip()


def main() -> int:
    root = repo_root()
    if sys.argv[1:]:
        resolved = root.resolve()
        files = []
        for arg in sys.argv[1:]:
            path = Path(arg).resolve()
            if path.is_relative_to(resolved) and path.is_file():
                files.append(path.relative_to(resolved).as_posix())
    else:
        files = tracked_files(("",))
    problems = []
    checked = 0
    for rel in sorted(set(files)):
        if rel in SELF_FILES:
            continue
        text = read_text(root, rel)
        if text is None:
            continue
        checked += 1
        problems += [f"{rel}:{n}: German text: {what[:100]}" for n, what in german_lines(rel, text)]
    return report(
        "english", problems, checked, "Write it in English, or (for the user interface) in the files of its allow list."
    )


if __name__ == "__main__":
    sys.exit(main())
