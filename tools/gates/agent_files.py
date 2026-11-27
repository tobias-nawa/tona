"""Agent files: generates the copies of the shared skill and the per-directory CLAUDE.md, or checks that they match.

Usage: agent_files.py [--check]

Contributors use different coding agents, and each looks for skills in its own directory: `.agents/skills/` (Codex,
Cursor, Windsurf, Amp and others), `.claude/skills/` (Claude Code) and `.github/skills/` (Copilot). Each skill text
lives once in `tools/agent_files/<skill>.md`; this script writes it to the three directories.
`--check` reports every copy that is missing or differs and changes nothing (pre-commit hook, CI).

No symlinks: git checks a committed symlink out as a text file on Windows without developer mode, and some agents
refuse to write through one (research/43-agent-harnesses.md).

Nested instruction files: next to every `AGENTS.md` below the repository root (`engine/backend/<name>/AGENTS.md`) the
script also keeps a `CLAUDE.md` that holds only `@AGENTS.md`. Claude Code reads `AGENTS.md` only where no `CLAUDE.md`
exists, and its documentation does not say whether a nested `AGENTS.md` loads when the root has a `CLAUDE.md`. The
one-line file makes it load either way and costs nothing in the harnesses that read `AGENTS.md` directly. Delete this
part once a test with `/context all` shows that the nested file loads without it.

Gemini CLI reads `GEMINI.md` and `AGENTS.md` only when its setting `context.fileName` lists it (geminicli.com/docs/cli/
gemini-md, checked 2026-11-25), so the root `GEMINI.md` holds the same one-line import (Gemini expands `@file`). Every
other agent in docs/contribute/optimize-for-your-hardware.md reads `AGENTS.md` itself; Aider through `.aider.conf.yml`.
"""

import subprocess
import sys
from pathlib import Path

from common import repo_root

SKILLS = ("optimize-for-hardware", "research-before-building")
TEMPLATE_DIR = Path("tools/agent_files")
SKILL_DIRS = (".agents/skills", ".claude/skills", ".github/skills")
IMPORT_LINE = "@AGENTS.md\n"
SKIP_PREFIXES = ("engine/third_party/", "work/")


def nested_instruction_files(root: Path) -> list[str]:
    """Tracked or new AGENTS.md files below the root, relative to it."""
    out = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
        check=True,
        cwd=root,
    )
    return sorted(
        f
        for f in out.stdout.splitlines()
        if f.endswith("/AGENTS.md") and not f.startswith(SKIP_PREFIXES) and (root / f).is_file()
    )


def expected_files(root: Path) -> dict[str, str]:
    """Path (relative to the root) -> content every generated file must have."""
    files = {}
    for skill in SKILLS:
        text = (root / TEMPLATE_DIR / f"{skill}.md").read_text(encoding="utf-8")
        files.update({f"{d}/{skill}/SKILL.md": text for d in SKILL_DIRS})
    if (root / "AGENTS.md").is_file():
        files["GEMINI.md"] = IMPORT_LINE
    for agents in nested_instruction_files(root):
        files[str(Path(agents).with_name("CLAUDE.md"))] = IMPORT_LINE
    return files


def stale(root: Path, files: dict[str, str]) -> list[str]:
    bad = []
    for rel, text in files.items():
        path = root / rel
        if not path.is_file() or path.read_text(encoding="utf-8") != text:
            bad.append(rel)
    return bad


def main(argv: list[str]) -> int:
    check = "--check" in argv
    root = repo_root()
    files = expected_files(root)
    bad = stale(root, files)
    if check:
        for rel in bad:
            print(f"agent_files: {rel} is missing or differs from its source", file=sys.stderr)
        if bad:
            print("Run: python3 tools/gates/agent_files.py", file=sys.stderr)
            return 1
        print(f"agent_files: ok ({len(files)} checked)")
        return 0
    for rel in bad:
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(files[rel], encoding="utf-8")
        print(f"agent_files: wrote {rel}")
    print(f"agent_files: {len(files)} files up to date")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
