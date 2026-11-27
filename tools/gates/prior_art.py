"""Prior-art gate: a pull request that changes engine code must show that its author looked for existing work.

Usage: prior_art.py [--base REF] [--body-file FILE] [--files FILE...] [--online]
       The pull request text comes from --body-file, else from the environment variable PR_BODY. The changed files come
       from --files, else from `git diff --name-only REF...HEAD` (REF defaults to the environment variable PR_BASE).

The text carries `Change type: bugfix | technique | backend | refactor | none` and a section "Prior art"
(.github/PULL_REQUEST_TEMPLATE.md, skill tools/agent_files/research-before-building.md). Rules:
- `none` is refused when a changed file is engine code (the paths of tools/select_rules.py: engine sources, build
  definitions, anything that triggers the golden checks; docs, tests, tools and the installer are not engine code).
- bugfix and technique and backend: at least one link or the words "nothing found", plus a `Search:` line that names
  scan.py with its topics version (v<N>), the topics or terms, and a window. Windows: 3 months for bugfix, 12 for
  technique and backend, as `N months` or as a date range.
- backend also names docs/backends/.
- refactor: no prior art, but the text must carry the proof keyword of bit-identical output (bit-identical, bit-exact,
  TONA_DETERMINISTIC or isa-diff). The gate checks only that the hint is there.
Links of every kind are checked for a possible form (tools/gates/prior_art_links.py: arXiv identifier, GitHub pull
request and commit); `--online` (CI only) also asks arxiv.org and github.com whether they exist, and never fails on a
network error. Without engine code in the change the gate asks for nothing about the section itself. The text is read
as data from a file or the environment; never put it into a shell command line (workflows.py refuses that in CI).
"""

import argparse
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path

import prior_art_links
from common import is_bot_pull_request, repo_root, report

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import select_rules

TYPES = ("bugfix", "technique", "backend", "refactor", "none")
PROOF = re.compile(r"bit[- ]?(?:identical|exact)|TONA_DETERMINISTIC|isa-diff", re.IGNORECASE)
MIN_MONTHS = {"bugfix": 3, "technique": 12, "backend": 12}
DAYS_PER_MONTH = 30
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
TYPE_LINE = re.compile(r"^[ \t>*_-]*change[ \t]+type[ \t]*(?:\*\*|__)?[ \t]*:[ \t]*(.*)$", re.IGNORECASE | re.MULTILINE)
HEADING = re.compile(r"^(#{1,6})[ \t]+(.*?)[ \t#]*$", re.MULTILINE)
SEARCH_LINE = re.compile(r"^[ \t>*_-]*search[ \t]*:.*(?:\n(?![ \t]*(?:$|[-*#]))[^\n]*)*", re.IGNORECASE | re.MULTILINE)
LINK = re.compile(r"https?://[^\s)>\]]+")
RANGE = re.compile(r"(\d{4}-\d{2}-\d{2})\s*(?:\.\.|to|-|–|until)\s*(\d{4}-\d{2}-\d{2})")
MONTHS = re.compile(r"(\d+)\s*(?:months?|mo)\b", re.IGNORECASE)
TOOL_VERSION = re.compile(r"scan\.py.{0,80}?\bv\d+\b", re.IGNORECASE | re.DOTALL)
TOPICS = re.compile(r"\b(?:topics?|terms?)\s*:\s*\S", re.IGNORECASE)


def change_type(text: str) -> tuple[str | None, str]:
    """The declared type, or None and why it is unusable."""
    found = TYPE_LINE.findall(text)
    if not found:
        return None, "no line `Change type: bugfix | technique | backend | refactor | none`"
    word = re.sub(r"[`*_\s]", "", found[0]).lower()
    if word not in TYPES:
        return None, f"`Change type:` must be exactly one of {', '.join(TYPES)}, found `{found[0].strip()}`"
    return word, ""


def prior_art_section(text: str) -> str | None:
    """Text under the heading "Prior art" up to the next heading of the same or a higher level."""
    for match in HEADING.finditer(text):
        if match.group(2).strip().lower() != "prior art":
            continue
        rest = text[match.end() :]
        end = next((m.start() for m in HEADING.finditer(rest) if len(m.group(1)) <= len(match.group(1))), len(rest))
        return rest[:end]
    return None


def engine_files(paths: list[str]) -> list[str]:
    """Changed files that are engine code: what decides the output, not docs, tests, tools or the installer."""
    found = []
    for path in paths:
        verdict = select_rules.outside_engine(path)
        if verdict is None:
            if not path.startswith("engine/tests/"):
                found.append(path)
        elif verdict.everything or (verdict.cls not in ("C0", "C1") and not verdict.installer):
            found.append(path)
    return found


def window_months(search: str) -> float | None:
    """Length of the stated window in 30-day months; the longest of the statements, None if there is none."""
    spans = []
    for first, last in RANGE.findall(search):
        try:
            spans.append((date.fromisoformat(last) - date.fromisoformat(first)).days / DAYS_PER_MONTH)
        except ValueError:
            continue
    spans += [float(m) for m in MONTHS.findall(search)]
    return max(spans) if spans else None


def search_problems(kind: str, search: str | None) -> list[str]:
    if not search:
        return ["no `Search:` line (topics or terms, window, scan.py with its topics version vN)"]
    problems = []
    if not TOOL_VERSION.search(search):
        problems.append("`Search:` does not name scan.py with its topics version (for example `scan.py topics v1`)")
    if not TOPICS.search(search):
        problems.append("`Search:` does not list the topics or terms (`topics: ...` or `terms: ...`)")
    months = window_months(search)
    if months is None:
        problems.append("`Search:` states no window (a date range `2025-11-11..2026-11-11` or `12 months`)")
    elif months < MIN_MONTHS[kind]:
        problems.append(f"window is shorter than {MIN_MONTHS[kind]} months, required for a {kind}")
    return problems


def check(text: str, paths: list[str]) -> list[str]:
    text = COMMENT.sub("", text)
    return type_problems(text, paths) + prior_art_links.form_problems(text)


def type_problems(text: str, paths: list[str]) -> list[str]:
    kind, why = change_type(text)
    if kind is None:
        return [why]
    engine = engine_files(paths)
    if kind == "none":
        if engine:
            shown = ", ".join(engine[:3]) + (" ..." if len(engine) > 3 else "")
            return [f"`Change type: none` but the change touches engine code ({shown}); pick bugfix or technique"]
        return []
    if not engine:
        return []
    if kind == "refactor":
        return [] if PROOF.search(text) else ["`Change type: refactor` needs the proof of bit-identical output"]
    return section_problems(kind, text, engine)


def section_problems(kind: str, text: str, engine: list[str]) -> list[str]:
    section = prior_art_section(text)
    if section is None or not section.strip() or section.strip().lower().rstrip(".") in ("n/a", "na"):
        return [f"engine code changed ({engine[0]}...) and the section `Prior art` is missing or empty"]
    searches = SEARCH_LINE.findall(section)
    search = "\n".join(searches) if searches else None
    body = SEARCH_LINE.sub("", section)
    problems = []
    if not LINK.search(body) and "nothing found" not in section.lower():
        problems.append("`Prior art` has no link and does not say `nothing found`")
    problems += search_problems(kind, search)
    if kind == "backend" and "docs/backends/" not in section:
        problems.append("a backend names its page under docs/backends/ in `Prior art`")
    return problems


def changed_files(base: str) -> list[str]:
    out = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", f"{base}...HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root(),
    )
    return out.stdout.splitlines()


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default=os.environ.get("PR_BASE", ""))
    parser.add_argument("--body-file")
    parser.add_argument("--files", nargs="*")
    parser.add_argument("--online", action="store_true", help="ask arxiv.org and github.com whether the links exist")
    args = parser.parse_args(argv)
    if is_bot_pull_request():
        print("prior_art: ok (bot pull request, no template)")
        return 0
    text = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else os.environ.get("PR_BODY", "")
    if args.files is None and not args.base:
        print("prior_art: give --files or --base (or PR_BASE)", file=sys.stderr)
        return 2
    paths = args.files if args.files is not None else changed_files(args.base)
    problems = check(text, paths)
    if args.online:
        found, warnings = prior_art_links.online_problems(COMMENT.sub("", text))
        problems += found
        for warning in warnings:
            print(f"prior_art: warning: {warning}", file=sys.stderr)
    return report(
        "prior_art",
        problems,
        len(paths),
        "See the header of tools/gates/prior_art.py and the skill research-before-building.",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
