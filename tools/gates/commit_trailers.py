"""Commit trailer gate: the trailer rules of AGENTS.md and CONTRIBUTING.md, checked instead of read.

Usage: commit_trailers.py MSGFILE...        (the commit-msg hook: pre-commit passes the message file)
       commit_trailers.py --range BASE..HEAD (CI: every commit of a pull request, merges skipped)

Rules:
- `Assisted-by:` is provenance (agent and model), so when it is there it has the form `<agent>:<model>`, optionally
  followed by tools, as in the Linux kernel (`Assisted-by: Claude Code:claude-sonnet-5-5`). A line like
  `Assisted-by: AI` says nothing a reviewer can check.
- A tool is not an author and does not certify anything: no `Co-Authored-By` that names a tool, model or bot, no
  `Signed-off-by` with such a name (the project has no DCO; the kernel and QEMU also forbid tool sign-offs), no
  `Claude-Session:` line or session link, no "Generated with ..." line or `Generated-by:` trailer. The project rule
  beats the default trailer of a tool.

What it cannot check: whether a human used a tool at all (a missing `Assisted-by:` passes: the gate never guesses),
whether the named model is the one that ran, and a tool that writes a human-looking name into `Signed-off-by`. Comment
lines (`#`) are what git strips, so they are ignored too.
"""

import re
import subprocess
import sys
from pathlib import Path

from common import is_bot_pull_request, repo_root, report

# Names that identify a tool, model or bot. Word boundaries: "Aider" matches, "Raider" does not.
TOOL_NAME = re.compile(
    r"\b(?:claude|anthropic|copilot|chatgpt|gpt|openai|codex|gemini|cursor|aider|devin|opencode|junie|qwen|llama|"
    r"mistral|deepseek|llm)\b|\[bot\]|noreply@anthropic\.com",
    re.IGNORECASE,
)
TRAILER = re.compile(r"^([A-Za-z][A-Za-z-]*):[ \t]*(.*)$")
ASSISTED_VALUE = re.compile(r"^[A-Za-z][A-Za-z0-9 ._+-]*:[A-Za-z0-9][A-Za-z0-9._/\[\]-]*(?: \S.*)?$")
GENERATED_WITH = re.compile(r"^[^\w\n]*generated\s+with\b", re.IGNORECASE)
SESSION_LINK = re.compile(r"claude\.ai/code/session", re.IGNORECASE)


def line_problem(line: str) -> str | None:
    if GENERATED_WITH.match(line):
        return 'a "Generated with ..." line; a tool adds none of these here'
    if SESSION_LINK.search(line):
        return "a tool session link; the commit carries no session reference"
    match = TRAILER.match(line)
    if not match:
        return None
    key, value = match.group(1).lower(), match.group(2).strip()
    if key == "claude-session":
        return "`Claude-Session:`; the commit carries no session reference"
    if key == "generated-by":
        return "`Generated-by:`; use `Assisted-by: <agent>:<model>`"
    if key == "co-authored-by" and TOOL_NAME.search(value):
        return f"`Co-Authored-By` names a tool or bot ({value}); a tool is not an author"
    if key == "signed-off-by" and TOOL_NAME.search(value):
        return f"`Signed-off-by` names a tool or bot ({value}); only a human certifies, and the project has no DCO"
    if key == "assisted-by" and not ASSISTED_VALUE.match(value):
        return f"`Assisted-by: {value}` is not `<agent>:<model>` (for example `Claude Code:claude-sonnet-5-5`)"
    return None


def message_problems(message: str, source: str) -> list[str]:
    """Problems of one commit message (or of a pull request text that becomes the squash message)."""
    problems = []
    for number, line in enumerate(message.splitlines(), start=1):
        if line.startswith("#"):
            continue
        why = line_problem(line.strip())
        if why:
            problems.append(f"{source}:{number}: {why}")
    return problems


def range_messages(spec: str) -> list[tuple[str, str]]:
    """(short hash, message) of every non-merge commit in the range."""
    out = subprocess.run(
        ["git", "log", "--no-merges", "--format=%h%x1f%B%x1e", spec],
        capture_output=True,
        text=True,
        check=True,
        cwd=repo_root(),
    )
    pairs = []
    for record in out.stdout.split("\x1e"):
        if record.strip():
            short, _, message = record.strip("\n").partition("\x1f")
            pairs.append((short, message))
    return pairs


def main(argv: list[str]) -> int:
    if len(argv) == 2 and argv[0] == "--range" and is_bot_pull_request():
        print("commit_trailers: ok (bot pull request: its sign-off is squashed away)")
        return 0
    if len(argv) == 2 and argv[0] == "--range":
        pairs = range_messages(argv[1])
    elif argv and not argv[0].startswith("-"):
        pairs = [(path, Path(path).read_text(encoding="utf-8", errors="replace")) for path in argv]
    else:
        print("usage: commit_trailers.py MSGFILE... | --range BASE..HEAD", file=sys.stderr)
        return 2
    problems = [p for source, message in pairs for p in message_problems(message, source)]
    return report("commit_trailers", problems, len(pairs), "See the header of tools/gates/commit_trailers.py.")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
