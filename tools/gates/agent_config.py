"""Agent-configuration gate: a pull request from outside the maintainer team does not touch files that run code or
instruct an agent on the machine of whoever checks the change out.

Usage: agent_config.py [--base REF] [--files FILE...]
       The changed files come from --files, else from `git diff --name-only REF...HEAD` (REF defaults to the
       environment variable PR_BASE). The author's relation to the repository comes from the environment variable
       PR_AUTHOR_ASSOCIATION (the `author_association` field GitHub sets on the pull request: OWNER, MEMBER,
       COLLABORATOR, CONTRIBUTOR, FIRST_TIME_CONTRIBUTOR, NONE) and PR_AUTHOR (the login). Allowed without a finding:
       the associations named in AGENT_CONFIG_ALLOWED (default OWNER), and `dependabot[bot]` for the pins in
       workflows and .pre-commit-config.yaml. An empty or unknown value counts as an outsider.

Why: a repository file such as `.mcp.json`, `.claude/settings.json` (hooks), `.cursor/`, `.vscode/tasks.json` or
`.devcontainer/` is started by the tool, with the rights of the user, as soon as the folder is opened (Snyk ToxicSkills,
2026-02). Files that only instruct agents (AGENTS.md, CLAUDE.md,
skills) are read with the authority of a system prompt. Review of the text cannot see every trick, so the change is
refused for outsiders and the maintainer writes it (or reimplements it) from the described intent.

Mechanism, in two layers: this gate is the CI job `agent-config` (a required check of the `main` ruleset), and
`.github/CODEOWNERS` assigns the same paths to the maintainer, so that the review is required once a second maintainer
exists (docs/contribute/github.md). The association is GitHub's, not the pull request's text: nobody can set it.
What it cannot catch: code that runs through a path not on the list (a new build script someone then runs), content
inside an allowed file, and a maintainer account that is itself compromised.
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import PurePosixPath

from common import repo_root, report

# (pattern on the repo-relative path, what it is). The first match names the finding.
EXECUTED = [
    (r"(^|/)\.?mcp\.json$", "MCP server configuration"),
    (r"^\.claude/(settings[^/]*\.json|hooks/|commands/|agents/)", "Claude Code settings, hooks, commands or agents"),
    (r"^\.cursor/", "Cursor configuration"),
    (r"^\.windsurf/", "Windsurf configuration"),
    (r"^\.(codex|gemini|continue|roo|clinerules|junie|opencode|zed)/", "agent configuration folder"),
    (r"^\.vscode/", "editor tasks and settings (tasks run on folder open)"),
    (r"^\.idea/", "editor configuration"),
    (r"^\.devcontainer(/|\.json$)", "dev container (runs commands on create)"),
    (r"(^|/)\.envrc$", "direnv script"),
    (r"^\.(husky|githooks)/", "git hooks folder"),
    (r"^\.pre-commit-config\.yaml$", "pre-commit hooks (run on every commit of everyone)"),
    (r"^\.github/(actions|workflows)/", "CI workflow or action"),
]
INSTRUCTIONS = [
    (r"(^|/)(AGENTS|CLAUDE|GEMINI)\.md$", "agent instruction file"),
    (r"(^|/)SKILL\.md$", "skill"),
    (r"^(\.claude|\.agents)/", "agent folder"),
    (r"^(tools/agent_files|\.github/skills)/", "skill source or copy"),
    (r"^\.github/(copilot-instructions\.md|instructions/|agents/|prompts/)", "Copilot instructions"),
    (r"^\.(cursorrules|windsurfrules|clinerules|aider[^/]*)$", "rules file of an agent"),
]
RULES = [(re.compile(p), what) for p, what in EXECUTED + INSTRUCTIONS]


def classify(path: str) -> str | None:
    """What kind of agent-run or agent-read file this path is, or None."""
    normalized = PurePosixPath(path).as_posix()
    for pattern, what in RULES:
        if pattern.search(normalized):
            return what
    return None


# Dependabot bumps the pins in these (docs/contribute/github.md); a person cannot register a login ending in [bot].
BOT = "dependabot[bot]"
BOT_PATHS = (".pre-commit-config.yaml", ".github/workflows/")


def check(paths: list[str], association: str, allowed: set[str], author: str = "") -> list[str]:
    if association.upper() in allowed:
        return []
    if author == BOT:
        paths = [p for p in paths if not p.startswith(BOT_PATHS)]
    return [
        f"{path}: {what}; only the maintainer changes it (author association: {association or 'unknown'})"
        for path in paths
        if (what := classify(path))
    ]


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
    parser.add_argument("--files", nargs="*")
    args = parser.parse_args(argv)
    if args.files is None and not args.base:
        print("agent_config: give --files or --base (or PR_BASE)", file=sys.stderr)
        return 2
    paths = args.files if args.files is not None else changed_files(args.base)
    allowed = {a.strip().upper() for a in os.environ.get("AGENT_CONFIG_ALLOWED", "OWNER").split(",") if a.strip()}
    association = os.environ.get("PR_AUTHOR_ASSOCIATION", "")
    author = os.environ.get("PR_AUTHOR", "")
    return report(
        "agent_config",
        check(paths, association, allowed, author),
        len(paths),
        "See the header of tools/gates/agent_config.py; describe what you need in the issue instead.",
    )


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
