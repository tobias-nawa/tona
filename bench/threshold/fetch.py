#!/usr/bin/python3 -I
"""Download the foreign repositories at their pinned commits into WORK/src (never into this repository).

Usage: fetch.py [NAME ...]    (default: all)
Only git runs here; no code of the repositories is executed or imported.
"""

import subprocess
import sys
from pathlib import Path

# -I (isolated mode) keeps the script directory off sys.path, so add it explicitly.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def git(d, *args):
    subprocess.run(["git", "-C", str(d), *args], check=True, capture_output=True)


def fetch(repo):
    d = common.repo_dir(repo)
    if not (d / ".git").exists():
        d.mkdir(parents=True, exist_ok=True)
        git(d, "init", "-q")
        git(d, "remote", "add", "origin", repo.url)
    git(d, "fetch", "-q", "--depth", "1", "origin", repo.commit)
    git(d, "checkout", "-q", "--detach", repo.commit)
    print(f"{repo.name}: {repo.commit} ({repo.license}) in {d}")


def main():
    names = sys.argv[1:] or list(common.REPOS)
    for n in names:
        fetch(common.REPOS[n])


if __name__ == "__main__":
    main()
