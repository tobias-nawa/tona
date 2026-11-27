import commit_trailers
import pytest
from conftest import commit_all, write

GOOD = "Fix the thing (#12)\n\nBody text.\n\nAssisted-by: Claude Code:claude-sonnet-5-5\n"


def problems(message: str) -> list[str]:
    return commit_trailers.message_problems(message, "msg")


def test_clean_messages_pass():
    assert problems(GOOD) == []
    assert problems("No tool involved (#1)\n") == []
    assert problems("Co-authored-by: Ada Lovelace <ada@example.org>\n") == []
    assert problems("Signed-off-by: Ada Lovelace <ada@example.org>\n") == []


@pytest.mark.parametrize(
    "value",
    [
        "Claude Code:claude-opus-5-5",
        "Copilot:gpt-5",
        "Aider:gemini-3-pro [ruff]",
        "Codex:gpt-5.1-codex-max",
        "opencode:qwen3-coder-480b",
    ],
)
def test_assisted_by_accepts_agent_and_model(value):
    assert problems(f"x\n\nAssisted-by: {value}\n") == []


@pytest.mark.parametrize("value", ["AI", "Claude", "Claude Code", ":gpt-5", "Claude Code:", "a b c"])
def test_assisted_by_needs_agent_and_model(value):
    found = problems(f"x\n\nAssisted-by: {value}\n")
    assert len(found) == 1
    assert "not `<agent>:<model>`" in found[0]


@pytest.mark.parametrize(
    "line",
    [
        "Co-Authored-By: Claude <noreply@anthropic.com>",
        "co-authored-by: Copilot <copilot@users.noreply.github.com>",
        "Co-authored-by: dependabot[bot] <49699333+dependabot[bot]@users.noreply.github.com>",
        "Co-authored-by: ChatGPT <x@example.org>",
        "Signed-off-by: Claude <noreply@anthropic.com>",
        "Signed-off-by: Codex CLI <codex@example.org>",
        "Claude-Session: https://claude.ai/code/session_01Abc",
        "https://claude.ai/code/session_01Abc",
        "Generated-by: Claude Code",
        "Generated with Claude Code",
        "\U0001f916 Generated with [Claude Code](https://claude.com/claude-code)",
    ],
)
def test_forbidden_trailers_are_reported(line):
    found = problems(f"x\n\n{line}\n")
    assert len(found) == 1
    assert found[0].startswith("msg:3: ")


def test_prose_and_comments_are_not_trailers():
    text = "Rule: never a Co-Authored-By line, no Claude-Session reference (#1)\n# Co-Authored-By: Claude\n"
    assert problems(text) == []


def test_word_boundary_does_not_hit_other_names():
    assert problems("Co-authored-by: Rolf Raider <r@example.org>\n") == []


def test_main_reads_message_files(tmp_path, capsys):
    good = tmp_path / "good"
    bad = tmp_path / "bad"
    good.write_text(GOOD)
    bad.write_text("x\n\nClaude-Session: https://claude.ai/code/session_1\n")
    assert commit_trailers.main([str(good)]) == 0
    assert commit_trailers.main([str(bad)]) == 1
    assert "session" in capsys.readouterr().err
    assert commit_trailers.main([]) == 2


def test_range_checks_every_commit_and_skips_merges(repo, capsys):
    write(repo, "a", "1")
    commit_all(repo, "base (#1)")
    base = commit_trailers.range_messages("HEAD")[0][0]
    write(repo, "a", "2")
    commit_all(repo, "ok (#1)\n\nAssisted-by: Claude Code:claude-sonnet-5-5")
    write(repo, "a", "3")
    commit_all(repo, "bad (#1)\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    assert commit_trailers.main(["--range", f"{base}..HEAD"]) == 1
    err = capsys.readouterr().err
    assert err.count("Co-Authored-By") == 1
    assert commit_trailers.main(["--range", f"{base}..HEAD~1"]) == 0


def test_the_range_of_a_dependabot_pull_request_is_skipped(monkeypatch):
    monkeypatch.setenv("PR_AUTHOR", "dependabot[bot]")
    assert commit_trailers.main(["--range", "does-not-exist..HEAD"]) == 0
