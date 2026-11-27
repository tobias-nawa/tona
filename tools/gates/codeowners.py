"""CODEOWNERS gate: the paths that decide what agents read and what runs on a machine stay with the maintainer.

Usage: codeowners.py   (reads .github/CODEOWNERS and the tracked files; no arguments)

GitHub applies the last matching line of `.github/CODEOWNERS`. A backend owner who is given a folder
(`/engine/backend/foo/ @someone`) would therefore also own the `AGENTS.md` inside it unless a later line takes it back.
The gate resolves owners the way GitHub does and fails when
- the file is missing or a line has a pattern but no `@owner`;
- one of the PROTECTED paths (instruction files, CI and gates, updater and release scripts, the reference backend with
  the megakernel, the CODEOWNERS file itself) is not owned by the maintainer;
- a tracked `AGENTS.md`, `CLAUDE.md` or `SKILL.md` anywhere, or a file under `.claude/`, `.agents/`, `.github/`,
  `tools/gates/` is not owned by the maintainer (a new backend folder with an owner line and no AGENTS.md line below it
  is caught here).

What it cannot check: that GitHub sees the same owner (a typo in the handle that still matches MAINTAINER passes; teams
and e-mail owners are not resolved), and that review is required (that is `REQUIRED_APPROVALS` in the ruleset).
"""

import re
import subprocess
import sys

from common import repo_root, report

MAINTAINER = "@tobias-nawa"
CODEOWNERS = ".github/CODEOWNERS"
PROTECTED = [
    "AGENTS.md",
    "CLAUDE.md",
    ".claude/skills/x/SKILL.md",
    ".agents/skills/x/SKILL.md",
    "tools/agent_files/x.md",
    "docs/contribute/optimize-for-your-hardware.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    ".github/CODEOWNERS",
    ".github/workflows/check.yml",
    ".github/PULL_REQUEST_TEMPLATE.md",
    "tools/gates/common.py",
    ".pre-commit-config.yaml",
    "justfile",
    "engine/update/updater.cpp",
    "installer/CMakeLists.txt",
    "toolchain/models.toml",
    "scripts/release-build.sh",
    "scripts/github-setup.sh",
    "docs/maintain/release.md",
    "engine/gpu/mega_abort.cpp",
    "engine/backend/backend.h",
    "scripts/gpu_queue.py",
]
GUARDED_NAMES = ("AGENTS.md", "CLAUDE.md", "SKILL.md")
GUARDED_PREFIXES = (".claude/", ".agents/", ".github/", "tools/gates/", "tools/agent_files/")


def pattern_regex(pattern: str) -> re.Pattern[str]:
    """The gitignore-style pattern of CODEOWNERS as a regular expression on a repo-relative path."""
    anchored = pattern.startswith("/") or "/" in pattern.rstrip("/")
    directory = pattern.endswith("/")
    body = pattern.strip("/")
    out = ""
    i = 0
    while i < len(body):
        if body.startswith("**/", i):
            out += "(?:.*/)?"
            i += 3
        elif body.startswith("**", i):
            out += ".*"
            i += 2
        elif body[i] == "*":
            out += "[^/]*"
            i += 1
        elif body[i] == "?":
            out += "[^/]"
            i += 1
        else:
            out += re.escape(body[i])
            i += 1
    prefix = "^" if anchored else "^(?:.*/)?"
    suffix = "/.*$" if directory else "(?:/.*)?$"
    return re.compile(prefix + out + suffix)


def parse(text: str) -> tuple[list[tuple[re.Pattern[str], list[str]]], list[str]]:
    rules, problems = [], []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.split(" #")[0].strip()
        if not line or line.startswith("#"):
            continue
        pattern, *owners = line.split()
        if not owners or not all(o.startswith("@") or "@" in o for o in owners):
            problems.append(
                f"{CODEOWNERS}:{number}: `{pattern}` has no owner (`@user`, `@org/team` or an e-mail address)"
            )
            continue
        rules.append((pattern_regex(pattern), owners))
    return rules, problems


def owners_of(path: str, rules: list[tuple[re.Pattern[str], list[str]]]) -> list[str]:
    owners: list[str] = []
    for regex, found in rules:
        if regex.match(path):
            owners = found  # the last match wins
    return owners


def tracked(root: str) -> list[str]:
    out = subprocess.run(["git", "ls-files"], capture_output=True, text=True, check=True, cwd=root)
    return out.stdout.splitlines()


def guarded(paths: list[str]) -> list[str]:
    return [p for p in paths if p.endswith(GUARDED_NAMES) or p.startswith(GUARDED_PREFIXES)]


def check(text: str, paths: list[str]) -> list[str]:
    rules, problems = parse(text)
    for path in PROTECTED + guarded(paths):
        owners = owners_of(path, rules)
        if owners != [MAINTAINER]:
            shown = " ".join(owners) if owners else "nobody"
            problems.append(
                f"{path}: owned by {shown}, expected only {MAINTAINER} ({CODEOWNERS}: the last matching line wins)"
            )
    return problems


def main() -> int:
    root = repo_root()
    path = root / CODEOWNERS
    if not path.is_file():
        print(f"codeowners: {CODEOWNERS} is missing", file=sys.stderr)
        return 1
    paths = tracked(str(root))
    problems = check(path.read_text(encoding="utf-8"), paths)
    return report(
        "codeowners", sorted(set(problems)), len(PROTECTED) + len(guarded(paths)), "See tools/gates/codeowners.py."
    )


if __name__ == "__main__":
    sys.exit(main())
