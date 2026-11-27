#!/usr/bin/python3 -I
"""Builds the code corpus that scripts/ab.py cuts its long prompts from, out of one fixed git commit.

Usage: scripts/make_corpus.py [--commit SHA] [--out FILE]

Reading the files with `git show <commit>:<path>` instead of the working tree makes the corpus, and with it every
prompt and every deterministic output SHA, independent of what is checked out. The previous corpus
(a 6.3 MB file of mixed sources of an older state, partly other projects) could not be rebuilt; the reference SHAs
of the new corpus are in research/measurements/2026-11-02-ab-tool.md.

Format: per file a line `// ===== inference-engine/<path> =====`, the file, a blank line. Two parts, each in sorted
path order, without the vendored `engine/third_party/`:

1. the sources (C++, HIP, Python, shell), BASE_CHARS characters; every prompt up to about 700K tokens is cut from it
2. appended for the `huge` workload (1M tokens do not fit into part 1): documents and data (md, txt, json, inc, yml).
   Appending keeps part 1 as the exact prefix of the corpus, so the prompts and the SHAs of 2K to 130K do not change.
"""

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path

# The commit the reference SHAs belong to. Change it only together with the reference SHAs.
DEFAULT_COMMIT = "e3660c0"
DEFAULT_SHA256 = "e13a800340859f7d39c63712d90383faae7e5a47c3865f0811e40d912612d38e"  # of the corpus built from it
BASE_CHARS = 2361816  # length of part 1 at DEFAULT_COMMIT (sha256 3c3c2a1f...c048c): prompts are bisected inside it
SUFFIXES = (".cpp", ".h", ".hip", ".py", ".sh")
EXTENSION_SUFFIXES = (".md", ".txt", ".json", ".inc", ".yml")
EXCLUDED = ("engine/third_party/",)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def corpus_paths(commit: str, suffixes: tuple = SUFFIXES) -> list:
    names = git("ls-tree", "-r", "--name-only", "-z", commit).decode().split("\0")
    return sorted(n for n in names if n.endswith(suffixes) and not n.startswith(EXCLUDED))


def build(commit: str) -> str:
    parts = []
    for path in corpus_paths(commit) + corpus_paths(commit, EXTENSION_SUFFIXES):
        text = git("show", f"{commit}:{path}").decode("utf-8", errors="replace")
        parts.append(f"// ===== inference-engine/{path} =====\n{text.rstrip(chr(10))}\n\n")
    return "".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--commit", default=DEFAULT_COMMIT)
    ap.add_argument("--out", default="work/corpus/code.txt")
    args = ap.parse_args()
    text = build(args.commit)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    sha = hashlib.sha256(text.encode()).hexdigest()
    print(f"{out}: {len(text)} bytes, {text.count(chr(10))} lines, sha256 {sha}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
