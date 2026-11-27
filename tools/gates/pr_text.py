"""Pull request text gate: the `AI use:` line of the template and the trailer rules.

Usage: pr_text.py [--body-file FILE] [--title TEXT] [--online]
       The text comes from --body-file, else from the environment variable PR_BODY, the title from --title, else
       PR_TITLE (the CI job passes them through `env:`, never through the shell command line). PR_AUTHOR names the
       author; Dependabot pull requests have no template and are skipped.

The project has no DCO and no CLA. The template carries one plain statement instead of boxes: by opening the pull
request the author confirms the rights (GitHub Terms of Service, section D.6, inbound = outbound). The repository
setting of the squash merge takes the pull request title and text into the commit message (scripts/github-setup.sh), so
the statement and the `Assisted-by:` lines stay in the history. This gate makes sure the AI disclosure is made:
- exactly one line `AI use: ...` outside comments;
- `none` (any case), or a text of at least two words naming the agent and what it did;
- with a tool named, at least one `Assisted-by: <agent>:<model>` line in the text (the squash merge keeps the text,
  not the trailers); with `none`, no `Assisted-by:` line;
- `feat`, `fix`, `perf` and breaking changes close their issue: `Closes|Fixes|Resolves #n` in the text, and the
  `(#n)` of the title is among the numbers (GitHub closes an issue only through such a keyword);
- `--online` (CI only): for `feat` and `perf` from an author without write access, the issue carries the label
  `accepted` (read-only GitHub API call; PR_AUTHOR_ASSOCIATION, GITHUB_REPOSITORY and GH_TOKEN from the workflow).
  A network error is a warning, never a finding;
- a section `## Models` with a model row or `n/a` (the template's empty row does not count);
- the text itself obeys the trailer rules of commit_trailers.py (it becomes the squash commit message).
HTML comments are removed first: the template's guidance sits in them, and a line written inside a comment counts
for nothing.

What it cannot check: that the line is true. It makes the disclosure explicit and permanent; the maintainer's review
does the rest.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

from commit_trailers import message_problems
from common import BOT_AUTHORS, report
from pr_title import parse

COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
AI_LINE = re.compile(r"^[ \t]*AI use:[ \t]*(.*?)[ \t]*$", re.MULTILINE | re.IGNORECASE)
ASSISTED_LINE = re.compile(r"^Assisted-by:[ \t]*\S", re.MULTILINE | re.IGNORECASE)
SKIPPED_AUTHORS = BOT_AUTHORS
MIN_TOOL_WORDS = 2
CLOSES = re.compile(r"\b(?:closes|fixes|resolves):?[ \t]+#([0-9]+)", re.IGNORECASE)
CLOSING_TYPES = ("feat", "fix", "perf")
ACCEPTED_TYPES = ("feat", "perf")
WRITE_ROLES = ("OWNER", "MEMBER", "COLLABORATOR")
ACCEPTED_LABEL = "accepted"
TIMEOUT_S = 10
MODELS_SECTION = re.compile(r"^##[ \t]+Models[ \t]*$(.*?)(?=^##[ \t]|\Z)", re.MULTILINE | re.DOTALL)


def models_problems(visible: str) -> list[str]:
    """The `## Models` section holds a table row with content (header and separator skipped) or `n/a`."""
    match = MODELS_SECTION.search(visible)
    if match is None:
        return ["the section `## Models` is missing: name the measured model(s) or write `n/a`"]
    lines = [line.strip() for line in match.group(1).splitlines() if line.strip()]
    if any(line.lower().rstrip(".") == "n/a" for line in lines):
        return []
    rows = [line for line in lines if line.startswith("|")][2:]
    if any(cell.strip() for row in rows for cell in row.strip("|").split("|")):
        return []
    return ["`## Models` needs a row per measured model or `n/a`"]


def issue_problems(title: str, visible: str) -> list[str]:
    """The text closes the issue named in the title (needed for feat, fix, perf and breaking changes)."""
    parsed, _ = parse(title, visible)
    if parsed is None or not (parsed.type in CLOSING_TYPES or parsed.breaking) or not parsed.refs:
        return []  # a missing `(#n)` is pr_title.py's finding
    closed = {int(n) for n in CLOSES.findall(visible)}
    if not closed:
        return [f"a `{parsed.type}` pull request needs `Closes #{parsed.refs[0]}` (or Fixes, Resolves) in its text"]
    return [f"the text does not close #{n} of the title (`Closes #{n}`)" for n in parsed.refs if n not in closed]


def accepted_problems(title: str, association: str, repo: str, token: str) -> tuple[list[str], list[str]]:
    """Findings and warnings: the issue of a `feat` or `perf` from an outside author carries `accepted`."""
    parsed, _ = parse(title)
    if parsed is None or parsed.type not in ACCEPTED_TYPES or not parsed.refs or association in WRITE_ROLES:
        return [], []
    number = parsed.refs[0]
    request = urllib.request.Request(f"https://api.github.com/repos/{repo}/issues/{number}")
    request.add_header("Accept", "application/vnd.github+json")
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
            labels = {label["name"] for label in json.load(response).get("labels", [])}
    except urllib.error.HTTPError as error:
        if error.code in (404, 410):
            return [f"issue #{number} of the title does not exist"], []
        return [], [f"could not read issue #{number}: HTTP {error.code}"]
    except (urllib.error.URLError, TimeoutError, ValueError) as error:
        return [], [f"could not read issue #{number}: {error}"]
    if ACCEPTED_LABEL not in labels:
        return [f"issue #{number} has no label `{ACCEPTED_LABEL}`: discuss it first (CONTRIBUTING.md)"], []
    return [], []


def check(text: str, title: str = "") -> list[str]:
    visible = COMMENT.sub("", text)
    lines = AI_LINE.findall(visible)
    if len(lines) != 1:
        return [f"exactly one line `AI use: none` or `AI use: <agent, model, what it did>` needed, found {len(lines)}"]
    problems = []
    declared = lines[0].rstrip(".").strip().lower()
    assisted = bool(ASSISTED_LINE.search(visible))
    if declared == "none":
        if assisted:
            problems.append("`AI use: none` contradicts the `Assisted-by:` line in the text")
    elif len(declared.split()) < MIN_TOOL_WORDS:
        problems.append("`AI use:` must be `none` or name the agent, the model and what it did")
    elif not assisted:
        problems.append(
            "`AI use:` names a tool, so the text needs a line `Assisted-by: <agent>:<model>` per agent: the squash"
            " merge keeps the pull request text, not the trailers of its commits"
        )
    problems += models_problems(visible)
    problems += message_problems(visible, "pull request text")
    if title:
        problems += issue_problems(title, visible)
    return problems


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--body-file")
    parser.add_argument("--title")
    parser.add_argument("--online", action="store_true", help="check the label `accepted` on the issue (CI only)")
    args = parser.parse_args(argv)
    if os.environ.get("PR_AUTHOR", "") in SKIPPED_AUTHORS:
        print("pr_text: skipped (Dependabot)")
        return 0
    text = Path(args.body_file).read_text(encoding="utf-8") if args.body_file else os.environ.get("PR_BODY", "")
    title = args.title if args.title is not None else os.environ.get("PR_TITLE", "")
    problems = check(text, title)
    if args.online and title:
        found, warnings = accepted_problems(
            title,
            os.environ.get("PR_AUTHOR_ASSOCIATION", ""),
            os.environ.get("GITHUB_REPOSITORY", ""),
            os.environ.get("GH_TOKEN", ""),
        )
        problems += found
        for warning in warnings:
            print(f"pr_text: warning: {warning}", file=sys.stderr)
    return report("pr_text", problems, 1, "See the header of tools/gates/pr_text.py and the template's `AI use`.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
