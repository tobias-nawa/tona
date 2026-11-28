"""Workflow gate: the patterns in .github/workflows/ that zizmor does not reject.

Usage: workflows.py [FILE...]  (no arguments: all tracked workflow files)

Rules, with the reason (GitHub Docs, "Secure use": https://docs.github.com/en/actions/reference/security/secure-use):
- `statuses: write` (also `permissions: write-all`): a workflow that can set commit statuses could post `gpu-golden`,
  the required check that only a maintainer sets after a local GPU run (docs/contribute/verification.md).
- `self-hosted` runners: they are not ephemeral, so code from a pull request can persist on the maintainer's machine.
- `${{ github.event.* }}` inside `run:`: the value is pasted into the shell script before it runs, so a title or branch
  name of a pull request becomes code. Pass it through `env:` instead. zizmor only flags the fields it knows to be
  attacker-controlled; this gate refuses the whole class so nobody has to judge.

- an action, API host, API key variable or command line tool that sends repository or pull request content to an LLM
  service: `anthropics/claude-code-action`, `run-gemini-cli`, Copilot and Codex actions, `api.anthropic.com`,
  `OPENAI_API_KEY`, `npm i -g @anthropic-ai/claude-code`. The text of issues and pull requests comes from outsiders;
  a model that reads it with a token in reach can be told to leak the token ("Comment and Control", 2026-04; see
  docs/maintain/maintainer.md: no bot reads outsider text). Reviews are done by a human, locally. The list is a floor,
  not a proof: an unknown vendor needs a new line here, and a script that calls a service from a file the workflow
  runs is not seen.

- `pull_request_target` together with the head of the pull request (`pull_request.head`, `github.head_ref`,
  `refs/pull/`): that trigger runs with a write token and secrets, so it may only look at the base branch
  (GitHub Security Lab, "Preventing pwn requests"). zizmor reports the trigger itself (`dangerous-triggers`); the one
  workflow that needs it, .github/workflows/pr-bot.yml, carries a documented `zizmor: ignore`; this rule keeps the
  head out of it.

Covered by zizmor already (v1.31.1, .pre-commit-config.yaml) and therefore not repeated here: `pull_request_target`,
`workflow_run`, unpinned actions, missing `persist-credentials: false`.
"""

import re
import sys

from common import gate_files, repo_root, report

WORKFLOW_DIR = ".github/workflows/"
EVENT_IN_EXPRESSION = re.compile(r"\$\{\{[^}]*\bgithub\.event\b")
STATUSES_WRITE = re.compile(r"^\s*statuses\s*:\s*write\b")
WRITE_ALL = re.compile(r"^\s*permissions\s*:\s*write-all\b")
SELF_HOSTED = re.compile(r"\bself-hosted\b")
LLM_ACTION = re.compile(
    r"\buses\s*:\s*['\"]?(?:anthropics/|google-github-actions/run-gemini-cli|google-gemini/|openai/|github/copilot|"
    r"github/gh-aw|githubnext/|actions/ai-inference|coderabbitai/|qodo-ai/|codium-ai/|sourcegraph/cody|continuedev/|"
    r"aider-ai/|paul-gauthier/|cline/|all-hands-ai/|openhands|sweepai/|mistralai/|ollama/)",
    re.IGNORECASE,
)
LLM_TEXT = re.compile(
    r"api\.anthropic\.com|api\.openai\.com|generativelanguage\.googleapis\.com|aiplatform\.googleapis\.com|"
    r"models\.github\.ai|models\.inference\.ai\.azure\.com|api\.mistral\.ai|api\.groq\.com|openrouter\.ai|"
    r"api\.deepseek\.com|api\.x\.ai|bedrock-runtime|"
    r"\b(?:ANTHROPIC|OPENAI|GEMINI|MISTRAL|GROQ|OPENROUTER|DEEPSEEK)_API_KEY\b|"
    r"@anthropic-ai/|@openai/codex|@google/gemini-cli|@github/copilot|pip3?\s+install\s+[^\n]*\b(?:anthropic|openai)\b",
    re.IGNORECASE,
)
TARGET_TRIGGER = re.compile(r"^\s*pull_request_target\s*:")
PULL_REQUEST_HEAD = re.compile(r"\bpull_request\.head\b|\bgithub\.head_ref\b|refs/pull/|\bworkflow_run\.head")
RUN_KEY = re.compile(r"^(\s*)(?:-\s+)?run\s*:\s*(.*)$")


def strip_comment(line: str) -> str:
    return re.sub(r"(^|\s)#.*$", "", line)


def run_lines(lines: list[str]) -> set[int]:
    """Indices of the lines that belong to a `run:` value, inline or as a block."""
    found: set[int] = set()
    i = 0
    while i < len(lines):
        match = RUN_KEY.match(lines[i])
        i += 1
        if not match:
            continue
        found.add(i - 1)
        if not re.match(r"^[|>][+-]?\d*\s*(#.*)?$", match.group(2)):
            continue
        key_indent = len(match.group(1))
        while i < len(lines) and (not lines[i].strip() or len(lines[i]) - len(lines[i].lstrip()) > key_indent):
            found.add(i)
            i += 1
    return found


def check_file(name: str, text: str) -> list[str]:
    lines = text.splitlines()
    in_run = run_lines(lines)
    problems = []
    target = any(TARGET_TRIGGER.match(strip_comment(raw)) for raw in lines)
    for n, raw in enumerate(lines):
        line = raw if n in in_run else strip_comment(raw)
        where = f"{name}:{n + 1}"
        if STATUSES_WRITE.match(line) or WRITE_ALL.match(line):
            problems.append(f"{where}: `statuses: write` (or write-all); only a maintainer posts commit statuses")
        if SELF_HOSTED.search(line):
            problems.append(f"{where}: self-hosted runner; use GitHub-hosted runners for code from pull requests")
        if LLM_ACTION.search(line) or LLM_TEXT.search(line):
            problems.append(f"{where}: sends repository or pull request content to an LLM service; no model runs in CI")
        if target and PULL_REQUEST_HEAD.search(strip_comment(raw)):
            problems.append(f"{where}: a `pull_request_target` workflow must not touch the head of the pull request")
        if n in in_run and EVENT_IN_EXPRESSION.search(raw):
            problems.append(f"{where}: github.event.* inside `run:`; pass it through `env:` and use the variable")
    return problems


def main() -> int:
    root = repo_root()
    files, _ = gate_files(sys.argv[1:], (".yml", ".yaml"))
    files = [f for f in files if f.startswith(WORKFLOW_DIR)]
    problems = []
    for f in files:
        problems += check_file(f, (root / f).read_text(encoding="utf-8"))
    return report("workflows", problems, len(files), "See the header of tools/gates/workflows.py.")


if __name__ == "__main__":
    sys.exit(main())
