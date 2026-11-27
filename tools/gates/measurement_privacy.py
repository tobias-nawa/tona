"""Privacy gate: no process lists in measurement logs, no private home paths in research and docs.

Usage: measurement_privacy.py [FILE...]  (no arguments: all tracked files)

Result files under research/measurements/ hold sums (available RAM and VRAM, swap, load, CPU and GPU share), never the
names of programs that ran on the machine. The known leak is a "top" field (process name: count) in a jsonl log line.
A name check over free text would flag the software we compare on purpose (llama.cpp, vkcube), so only the field is
checked.

Logs and notes under research/ and docs/ are public: an absolute home path (/home/<name>/, /Users/<name>/) names the
person and the folder layout of the machine that produced them. Write <repo> for the checkout and ~ for the home.
Agent scratchpad paths (/tmp/claude-<uid>/<encoded home>/<session>/) are refused the same way.
Placeholder names of examples and tests (u, user, me, x, ci, runner, you) stay allowed.
"""

import json
import re
import sys

from common import fail_if_empty, gate_files, repo_root, report

PREFIX = "research/measurements/"
FORBIDDEN_KEYS = ("top",)
PATH_PREFIXES = ("research/", "docs/")
TEXT_SUFFIXES = (".md", ".txt", ".log", ".out", ".json", ".jsonl", ".toml", ".tsv", ".csv", ".yml", ".yaml", ".sh")
HOME_PATH = re.compile(r"/(?:home|Users)/([A-Za-z0-9][A-Za-z0-9._-]*)/")
SCRATCH_PATH = re.compile(r"/tmp/claude-[0-9]+/")
PLACEHOLDERS = {"u", "user", "me", "x", "ci", "runner", "you", "name", "username"}


def bad_keys(line: str) -> list[str]:
    try:
        data = json.loads(line)
    except ValueError:
        return []
    return [k for k in FORBIDDEN_KEYS if isinstance(data, dict) and k in data]


def home_paths(line: str) -> list[str]:
    found = [m.group(0) for m in HOME_PATH.finditer(line) if m.group(1) not in PLACEHOLDERS]
    # an agent scratchpad path carries the home path and a session id in its folder names
    return found + [m.group(0) for m in SCRATCH_PATH.finditer(line)]


def main() -> int:
    root = repo_root()
    files, explicit = gate_files(sys.argv[1:], TEXT_SUFFIXES)
    files = [f for f in files if f.startswith(PATH_PREFIXES)]
    if not explicit:
        fail_if_empty(files, "measurement-privacy")
    problems = []
    for f in files:
        with open(root / f, encoding="utf-8", errors="replace") as fh:
            for n, line in enumerate(fh, 1):
                if f.startswith(PREFIX) and f.endswith(".jsonl"):
                    problems += [f'{f}:{n}: field "{k}" (process names)' for k in bad_keys(line)]
                problems += [f"{f}:{n}: private home path {p}" for p in home_paths(line)]
    return report(
        "measurement-privacy",
        problems,
        len(files),
        "Replace process lists by sums (see docs/contribute/measurement.md); write <repo> and ~ instead of home paths.",
    )


if __name__ == "__main__":
    sys.exit(main())
