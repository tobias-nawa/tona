"""File gate: no source file over 500 lines.

Usage: file_length.py [FILE...]  (no arguments: all tracked files)

An agent reads a file of up to ~500 lines completely in one tool call; beyond that it samples and misses wiring.
There is no exception list: split the file by topic.
"""

import sys

from common import SCRIPT_SUFFIXES, SIZE_SUFFIXES, fail_if_empty, gate_files, repo_root, report

LIMIT = 500


def main() -> int:
    root = repo_root()
    files, explicit = gate_files(sys.argv[1:], SIZE_SUFFIXES + SCRIPT_SUFFIXES)
    if not explicit:
        fail_if_empty(files, "file-length")
    problems = []
    for f in files:
        with open(root / f, encoding="utf-8", errors="replace") as fh:
            n = sum(1 for _ in fh)
        if n > LIMIT:
            problems.append(f"{f}: {n} lines (limit {LIMIT})")
    return report("file-length", problems, len(files), "Split the file by topic.")


if __name__ == "__main__":
    sys.exit(main())
