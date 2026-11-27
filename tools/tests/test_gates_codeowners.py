from pathlib import Path

import codeowners
import pytest
from conftest import commit_all, write

REAL = (Path(__file__).resolve().parents[2] / ".github/CODEOWNERS").read_text(encoding="utf-8")
M = codeowners.MAINTAINER


def owners(text: str, path: str) -> list[str]:
    rules, problems = codeowners.parse(text)
    assert problems == []
    return codeowners.owners_of(path, rules)


@pytest.mark.parametrize(
    ("pattern", "path", "hit"),
    [
        ("AGENTS.md", "AGENTS.md", True),
        ("AGENTS.md", "engine/backend/x/AGENTS.md", True),
        ("/AGENTS.md", "engine/backend/x/AGENTS.md", False),
        ("/tools/gates/", "tools/gates/common.py", True),
        ("/tools/gates/", "tools/gates_other/x.py", False),
        ("/scripts/release*", "scripts/release-build.sh", True),
        ("/scripts/release*", "scripts/sub/release-x.sh", False),
        ("/engine/backend/*.h", "engine/backend/backend.h", True),
        ("/engine/backend/*.h", "engine/backend/metal/x.h", False),
        ("docs/**/x.md", "docs/a/b/x.md", True),
        ("/tools/tests/test_gates_*.py", "tools/tests/test_gates_a.py", True),
    ],
)
def test_pattern_matching(pattern, path, hit):
    assert bool(codeowners.pattern_regex(pattern).match(path)) is hit


def test_the_last_match_wins():
    text = f"/engine/backend/foo/ @owner\n/engine/backend/foo/AGENTS.md {M}\n"
    assert owners(text, "engine/backend/foo/x.cpp") == ["@owner"]
    assert owners(text, "engine/backend/foo/AGENTS.md") == [M]


def test_the_real_file_keeps_the_protected_paths_with_the_maintainer():
    paths = ["engine/backend/metal/AGENTS.md", "engine/backend/metal/CLAUDE.md", "engine/gpu/forward.hip"]
    assert codeowners.check(REAL, paths) == []


def test_a_backend_owner_without_a_line_for_agents_md_is_caught():
    text = REAL + "/engine/backend/foo/ @someone\n"
    problems = codeowners.check(text, ["engine/backend/foo/AGENTS.md", "engine/backend/foo/x.cpp"])
    assert len(problems) == 1
    assert "engine/backend/foo/AGENTS.md: owned by @someone" in problems[0]
    fixed = text + f"/engine/backend/foo/AGENTS.md {M}\n"
    assert codeowners.check(fixed, ["engine/backend/foo/AGENTS.md"]) == []


def test_an_extra_owner_or_a_missing_rule_is_caught():
    shared = REAL.replace(f"/tools/gates/                              {M}", f"/tools/gates/ {M} @other")
    assert any("tools/gates/common.py" in p for p in codeowners.check(shared, []))
    assert any("nobody" in p for p in codeowners.check("# empty\n", []))


def test_a_line_without_an_owner_is_a_problem():
    _, problems = codeowners.parse("/tools/gates/\n")
    assert len(problems) == 1 and "no owner" in problems[0]


def test_main_needs_the_file(repo, capsys):
    write(repo, "README.md", "x")
    commit_all(repo)
    assert codeowners.main() == 1
    write(repo, ".github/CODEOWNERS", REAL)
    commit_all(repo)
    assert codeowners.main() == 0
