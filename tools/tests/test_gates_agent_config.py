import agent_config
import pytest
from conftest import commit_all, git, write


@pytest.mark.parametrize(
    "path",
    [
        ".mcp.json",
        "sub/.mcp.json",
        ".vscode/mcp.json",
        ".vscode/tasks.json",
        ".claude/settings.json",
        ".claude/settings.local.json",
        ".claude/hooks/pre.sh",
        ".cursor/rules/x.mdc",
        ".codex/config.toml",
        ".gemini/settings.json",
        ".devcontainer/devcontainer.json",
        ".envrc",
        ".husky/pre-commit",
        ".pre-commit-config.yaml",
        ".github/workflows/check.yml",
        "AGENTS.md",
        "engine/backend/foo/AGENTS.md",
        "CLAUDE.md",
        ".claude/skills/x/SKILL.md",
        ".agents/skills/x/SKILL.md",
        "tools/agent_files/research-before-building.md",
        ".github/copilot-instructions.md",
        ".cursorrules",
    ],
)
def test_agent_files_are_classified(path):
    assert agent_config.classify(path)


@pytest.mark.parametrize(
    "path",
    [
        "README.md",
        "engine/gpu/forward.hip",
        "docs/agents.md",
        "tools/gates/common.py",
        "scripts/mcp_probe.py",
        "vscode/x",
    ],
)
def test_ordinary_files_are_not(path):
    assert agent_config.classify(path) is None


def test_outsider_is_refused_with_each_path():
    found = agent_config.check([".mcp.json", "README.md", "AGENTS.md"], "CONTRIBUTOR", {"OWNER"})
    assert [f.split(":")[0] for f in found] == [".mcp.json", "AGENTS.md"]
    assert "CONTRIBUTOR" in found[0]


@pytest.mark.parametrize("association", ["", "NONE", "FIRST_TIME_CONTRIBUTOR", "MEMBER", "garbage"])
def test_unknown_and_lower_associations_count_as_outsiders(association):
    assert agent_config.check([".mcp.json"], association, {"OWNER"})


def test_owner_passes_and_the_allowed_set_can_grow():
    assert agent_config.check([".mcp.json"], "OWNER", {"OWNER"}) == []
    assert agent_config.check([".mcp.json"], "owner", {"OWNER"}) == []
    assert agent_config.check([".mcp.json"], "MEMBER", {"OWNER", "MEMBER"}) == []


def test_main_uses_environment_and_diff(repo, monkeypatch, capsys):
    write(repo, "a.txt", "1")
    commit_all(repo, "base")
    git(repo, "checkout", "-q", "-b", "pr")
    write(repo, ".claude/settings.json", "{}")
    write(repo, "docs/x.md", "x")
    commit_all(repo, "pr")
    monkeypatch.setenv("PR_AUTHOR_ASSOCIATION", "CONTRIBUTOR")
    assert agent_config.main(["--base", "main"]) == 1
    assert ".claude/settings.json" in capsys.readouterr().err
    monkeypatch.setenv("PR_AUTHOR_ASSOCIATION", "OWNER")
    assert agent_config.main(["--base", "main"]) == 0
    monkeypatch.setenv("PR_AUTHOR_ASSOCIATION", "")
    assert agent_config.main(["--files", "docs/x.md"]) == 0
    assert agent_config.main([]) == 2


def test_dependabot_may_bump_pins_but_nothing_else():
    paths = [".github/workflows/check.yml", ".pre-commit-config.yaml", ".mcp.json", "AGENTS.md"]
    found = agent_config.check(paths, "NONE", {"OWNER"}, "dependabot[bot]")
    assert [f.split(":")[0] for f in found] == [".mcp.json", "AGENTS.md"]
    assert len(agent_config.check(paths, "NONE", {"OWNER"}, "someone")) == 4
