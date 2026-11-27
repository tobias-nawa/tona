"""Pull request title gate: Conventional Commits, because the title is the commit and the commit decides the version.

Usage: pr_title.py [--title TEXT] [--body-file FILE]
       The title comes from --title, else from the environment variable PR_TITLE; the text from --body-file, else from
       PR_BODY (the CI job passes both through `env:`, never through the shell command line).

Main gets one squash commit per pull request and its subject is the pull request title (scripts/github-setup.sh), so
the title is what scripts/release-next.py reads to compute the version. The commits inside a pull request are free-form
(they keep the trailer rules of commit_trailers.py) and are not seen by the release.

Format: `type(scope)!: subject (#n)`
- type: feat fix perf docs test refactor build ci chore revert
- scope: optional, one word of SCOPES (the top-level parts of the tree, see the list below)
- `!` after the type or scope, or a paragraph starting `BREAKING CHANGE:` in the text, marks a breaking change
- subject: English, starts lower case, no trailing full stop, at most MAX_SUBJECT characters
- the issue reference `(#n)` closes the title. Required for feat, fix, perf and every breaking change (they need an
  accepted issue, CONTRIBUTING.md); the other types may omit it (a typo, a broken link, a test, a CI tweak). GitHub
  appends the pull request number itself when it squashes; a title that already ends in `(#n)` therefore gets two.
Dependabot titles pass: `.github/dependabot.yml` gives them the prefix `build(deps)` and `ci(deps)`; their capital
"Bump" is accepted when PR_AUTHOR is the bot.
"""

import argparse
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from common import is_bot_pull_request, report

TYPES = ("feat", "fix", "perf", "docs", "test", "refactor", "build", "ci", "chore", "revert")
# Types whose title needs `(#n)`: they change what users get and have an accepted issue behind them.
ISSUE_REQUIRED = ("feat", "fix", "perf")
# The parts of the tree (engine/<dir>, installer, tools, ...) plus `deps` for dependency bumps and `release`.
SCOPES = (
    "engine", "backend", "gpu", "cpu", "io", "model", "models", "offer", "server", "client", "spec", "update",
    "installer", "tona", "tools", "scripts", "bench", "docs", "research", "ci", "release", "deps", "toolchain",
)  # fmt: skip
MAX_SUBJECT = 100
TITLE = re.compile(
    r"^(?P<type>[a-z]+)(?:\((?P<scope>[^()\s]*)\))?(?P<bang>!)?: (?P<subject>.+?)(?P<refs>(?: \(#[0-9]+\))*)$"
)
REF = re.compile(r"\(#([0-9]+)\)")
BREAKING_PARAGRAPH = re.compile(r"^BREAKING[ -]CHANGE:[ \t]*\S", re.MULTILINE)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
NON_ASCII = re.compile(r"[^\x20-\x7e]")


@dataclass(frozen=True)
class Title:
    type: str
    scope: str
    subject: str
    breaking: bool
    refs: tuple[int, ...]


def parse(title: str, body: str = "") -> tuple[Title | None, list[str]]:
    """The parsed title and the reasons it does not conform (empty when it does)."""
    match = TITLE.match(title.strip())
    if not match:
        return None, [f"title `{title}` is not `type(scope): subject (#n)` (types: {', '.join(TYPES)})"]
    problems = []
    kind, scope, subject = match["type"], match["scope"] or "", match["subject"]
    refs = tuple(int(n) for n in REF.findall(match["refs"]))
    breaking = bool(match["bang"]) or bool(BREAKING_PARAGRAPH.search(COMMENT.sub("", body)))
    if kind not in TYPES:
        problems.append(f"type `{kind}` is not one of: {', '.join(TYPES)}")
    if match["scope"] is not None and scope not in SCOPES:
        problems.append(f"scope `{scope}` is not one of: {', '.join(SCOPES)} (or leave the scope out)")
    problems += subject_problems(subject)
    if not refs and (kind in ISSUE_REQUIRED or breaking):
        problems.append(f"a `{kind}` title that changes behaviour ends with the issue reference `(#n)`")
    return Title(kind, scope, subject, breaking, refs), problems


def subject_problems(subject: str) -> list[str]:
    problems = []
    if NON_ASCII.search(subject):
        problems.append("the subject is English plain text: no non-ASCII characters")
    if len(subject) > MAX_SUBJECT:
        problems.append(f"the subject has {len(subject)} characters, at most {MAX_SUBJECT}")
    if subject[0].isupper() and not subject.split(maxsplit=1)[0].isupper():
        problems.append("the subject starts lower case (`feat: add x`, not `feat: Add x`); acronyms may stay")
    if subject.endswith("."):
        problems.append("the subject has no trailing full stop")
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--title")
    parser.add_argument("--body-file")
    args = parser.parse_args(argv)
    title = args.title if args.title is not None else os.environ.get("PR_TITLE", "")
    body = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else os.environ.get("PR_BODY", "")
    problems = parse(title, body)[1]
    if is_bot_pull_request():
        # Dependabot writes "Bump ..." and offers no setting for the case; the type and scope still count.
        problems = [p for p in problems if "starts lower case" not in p]
    return report("pr_title", problems, 1, "See the header of tools/gates/pr_title.py and CONTRIBUTING.md.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
