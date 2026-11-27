"""tools/gates/pr_title.py: the pull request title is a Conventional Commit, the commit that main gets."""

import pr_title
import pytest


@pytest.mark.parametrize(
    "title",
    [
        "feat(server): stream the tool calls (#12)",
        "fix: stop the stall on a full cache (#7)",
        "perf(gpu): fuse the norm into the matvec (#99)",
        "feat(installer)!: rename the setting port (#5)",
        "docs: fix a broken link",
        "ci(deps): bump the actions group with 3 updates",
        "build(deps): bump ruff from 0.14.0 to 0.14.1",
        "refactor(model): split the loader (#31) (#40)",
        "revert: feat(server): stream the tool calls (#12)",
    ],
)
def test_conforming_titles_pass(title):
    assert pr_title.parse(title)[1] == []


@pytest.mark.parametrize(
    ("title", "fragment"),
    [
        ("Fix the cache stall (#123)", "is not `type(scope)"),
        ("fixes: the cache (#1)", "type `fixes`"),
        ("fix(gates): the cache (#1)", "scope `gates`"),
        ("fix(): the cache (#1)", "scope ``"),
        ("fix: Stop the stall (#1)", "lower case"),
        ("fix: stop the stall. (#1)", "full stop"),
        ("fix: stop the stall", "issue reference"),
        ("feat: add x", "issue reference"),
        ("docs: café menu", "non-ASCII"),
        ("fix: " + "x" * 101 + " (#1)", "at most"),
        ("fix:no space (#1)", "is not `type(scope)"),
    ],
)
def test_bad_titles_fail(title, fragment):
    assert any(fragment in p for p in pr_title.parse(title)[1])


def test_acronyms_may_start_the_subject():
    assert pr_title.parse("docs: GGUF notes")[1] == []


def test_a_bang_marks_a_breaking_change_and_needs_the_issue():
    parsed, problems = pr_title.parse("refactor!: drop the flag")
    assert parsed.breaking
    assert any("issue reference" in p for p in problems)


def test_a_breaking_change_paragraph_in_the_text_counts():
    body = "What.\n\nBREAKING CHANGE: the port moved to 8662.\n"
    parsed, problems = pr_title.parse("fix(server): move the port (#3)", body)
    assert parsed.breaking and problems == []


def test_a_breaking_change_inside_a_comment_does_not_count():
    parsed, _ = pr_title.parse("fix: x (#3)", "<!-- BREAKING CHANGE: describe -->\n")
    assert not parsed.breaking


def test_the_fields_are_parsed_and_both_numbers_kept():
    parsed, _ = pr_title.parse("feat(client): add a tab (#12) (#40)")
    assert (parsed.type, parsed.scope, parsed.subject, parsed.refs) == ("feat", "client", "add a tab", (12, 40))


def test_main_reads_the_environment(monkeypatch):
    monkeypatch.setenv("PR_TITLE", "nope")
    assert pr_title.main([]) == 1
    monkeypatch.setenv("PR_TITLE", "docs: fine")
    assert pr_title.main([]) == 0


def test_a_dependabot_title_may_start_upper_case(monkeypatch):
    monkeypatch.setenv("PR_TITLE", "build(deps): Bump the python group with 4 updates")
    monkeypatch.delenv("PR_AUTHOR", raising=False)
    assert pr_title.main([]) == 1
    monkeypatch.setenv("PR_AUTHOR", "dependabot[bot]")
    assert pr_title.main([]) == 0
    monkeypatch.setenv("PR_TITLE", "Bump the python group")
    assert pr_title.main([]) == 1
