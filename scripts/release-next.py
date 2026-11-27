#!/usr/bin/env python3
"""Compute the next version from the commit subjects, draft its CHANGELOG.md section (docs/maintain/release.md, step 1).

    release-next.py [--repo DIR] [--ref HEAD] [--since TAG] [--date YYYY-MM-DD] [--first-version 0.1.0] [--write]

Main has one squash commit per pull request and its subject is the pull request title, a Conventional Commit
(tools/gates/pr_title.py). This script reads `git log --no-merges <last tag>..<ref>`, the last tag being the newest
`vX.Y.Z` reachable from the ref (or --since), and applies SemVer:

  major version 0 (the interface is not frozen, CHANGELOG.md "Versioning"):
      breaking change -> minor, feat -> minor, fix or perf -> patch
  from 1.0.0:
      breaking change -> major, feat -> minor, fix or perf -> patch
  docs, test, refactor, build, ci, chore alone change nothing a user gets: no release is proposed (exit 3), unless
  they are marked breaking.

A breaking change is `!` in the title or a `BREAKING CHANGE:` paragraph in the commit message (the pull request text
is the message). A subject that does not conform is an error and nothing is computed: fix it with a follow-up pull
request that makes the intended change explicit, or pass --since past it. Without any tag this is the first release:
the version is --first-version (0.1.0) and the history is not read, since the first release starts the log.

The draft goes to stdout; --write inserts it below `## [Unreleased]` in CHANGELOG.md. Edit it for users (short lines,
what they get and what breaks), then tag as printed. The script never tags, signs or pushes.
"""

import argparse
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tools" / "gates"))
import pr_title  # noqa: E402

TAG = re.compile(r"v([0-9]+)\.([0-9]+)\.([0-9]+)")
SECURITY = re.compile(r"security|\bCVE-|vulnerab", re.IGNORECASE)
# Types a user notices; the others (docs, test, refactor, build, ci, chore) stay out of the notes unless breaking.
SECTION_OF = {"feat": "Added", "fix": "Fixed", "perf": "Changed", "revert": "Changed"}
ORDER = ("Added", "Changed", "Fixed", "Security")
FIELD, RECORD = "\x1f", "\x1e"
NOTHING = 3  # exit code: no release worth proposing


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, check=check, capture_output=True, text=True)


def last_tag(repo: Path, ref: str) -> str | None:
    found = git(repo, "describe", "--tags", "--abbrev=0", "--match", "v[0-9]*.[0-9]*.[0-9]*", ref, check=False)
    return found.stdout.strip() if found.returncode == 0 else None


def commits(repo: Path, since: str, ref: str) -> list[tuple[str, str]]:
    """(subject, body) of every non-merge commit in since..ref, newest first."""
    out = git(repo, "log", "--no-merges", f"--format=%s{FIELD}%b{RECORD}", f"{since}..{ref}").stdout
    pairs = []
    for record in out.split(RECORD):
        if record.strip():
            subject, _, body = record.strip("\n").partition(FIELD)
            pairs.append((subject, body))
    return pairs


def bump(version: tuple[int, int, int], parsed: list[pr_title.Title]) -> tuple[int, int, int] | None:
    major, minor, patch = version
    breaking = any(t.breaking for t in parsed)
    if breaking and major > 0:
        return major + 1, 0, 0
    if breaking or any(t.type == "feat" for t in parsed):
        return major, minor + 1, 0
    if any(t.type in ("fix", "perf", "revert") for t in parsed):
        return major, minor, patch + 1
    return None


def section_of(title: pr_title.Title) -> str | None:
    if title.type == "fix" and SECURITY.search(title.subject):
        return "Security"
    return "Changed" if title.breaking else SECTION_OF.get(title.type)


def entry(title: pr_title.Title) -> str:
    refs = ", ".join(f"#{n}" for n in title.refs)
    scope = f"{title.scope}: " if title.scope else ""
    return f"- {'**Breaking:** ' if title.breaking else ''}{scope}{title.subject}" + (f" ({refs})" if refs else "")


def draft(version: str, day: str, parsed: list[pr_title.Title]) -> str:
    groups: dict[str, list[str]] = {name: [] for name in ORDER}
    for title in parsed:
        name = section_of(title)
        if name:
            groups[name].append(entry(title))
    lines = [f"## [{version}] - {day}", ""]
    for name in ORDER:
        if groups[name]:
            lines += [f"### {name}", "", *groups[name], ""]
    return "\n".join(lines)


def write_into(changelog: Path, text: str) -> None:
    body = changelog.read_text(encoding="utf-8")
    marker = re.search(r"^## \[Unreleased\][ \t]*\n", body, re.MULTILINE)
    if not marker:
        sys.exit("release-next: CHANGELOG.md has no `## [Unreleased]` heading")
    rest = body[marker.end() :].lstrip("\n")
    changelog.write_text(f"{body[: marker.end()]}\n{text}\n{rest}", encoding="utf-8")


def parse_all(pairs: list[tuple[str, str]]) -> tuple[list[pr_title.Title], list[str]]:
    parsed, errors = [], []
    for subject, body in pairs:
        title, problems = pr_title.parse(subject, body)
        errors += [f"{subject!r}: {p}" for p in problems]
        if title and not problems:
            parsed.append(title)
    return parsed, errors


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--repo", type=Path, default=HERE.parent)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--since", help="tag to start after (default: the last vX.Y.Z tag)")
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--first-version", default="0.1.0")
    parser.add_argument("--write", action="store_true", help="insert the draft into CHANGELOG.md")
    args = parser.parse_args(argv)
    since = args.since or last_tag(args.repo, args.ref)
    if since is None:
        print(f"no tag: first release {args.first_version}; write its notes by hand (docs/maintain/release.md)")
        print(f"git tag -s v{args.first_version}")
        return 0
    base = TAG.fullmatch(since)
    if not base:
        sys.exit(f"release-next: --since {since!r} is not a vX.Y.Z tag; the base version needs one")
    parsed, errors = parse_all(commits(args.repo, since, args.ref))
    if errors:
        print(f"release-next: {len(errors)} subject problem(s) since {since}:", file=sys.stderr)
        print("\n".join(errors), file=sys.stderr)
        return 1
    major, minor, patch = (int(n) for n in base.groups())
    new = bump((major, minor, patch), parsed)
    if new is None:
        print(f"nothing a user would notice since {since} ({len(parsed)} commits); no release proposed")
        return NOTHING
    version = ".".join(map(str, new))
    text = draft(version, args.date, parsed)
    if args.write:
        write_into(args.repo / "CHANGELOG.md", text)
    print(text)
    print(f"# {since} -> v{version}, {len(parsed)} commits. Edit the section, commit it, then:")
    print(f"git tag -s v{version}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
