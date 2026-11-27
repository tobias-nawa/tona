import io
import json
import urllib.error
from pathlib import Path

import pr_text
import pytest

TEMPLATE = Path(__file__).resolve().parents[2] / ".github/PULL_REQUEST_TEMPLATE.md"
ASSISTED = "\nAssisted-by: Claude Code:claude-sonnet-5-5\n"


def text(line: str, tail: str = "") -> str:
    head = "## What and why\n\nSomething.\n\n## Models\n\nn/a\n\n"
    return f"{head}## AI use\n\n{line}\n\nBy opening this pull request I confirm.\n{tail}"


def test_none_passes_in_any_case():
    assert pr_text.check(text("AI use: none")) == []
    assert pr_text.check(text("ai use: None.")) == []


def test_a_tool_with_an_assisted_by_line_passes():
    assert pr_text.check(text("AI use: Claude Code, first draft of the tests", ASSISTED)) == []


def test_a_tool_without_an_assisted_by_line_fails():
    found = pr_text.check(text("AI use: Claude Code, first draft of the tests"))
    assert any("Assisted-by: <agent>:<model>" in p for p in found)


def test_an_assisted_by_line_inside_a_comment_does_not_count():
    tail = "\n<!-- Assisted-by: Claude Code:claude-sonnet-5-5 -->\n"
    found = pr_text.check(text("AI use: Claude Code, first draft", tail))
    assert any("Assisted-by: <agent>:<model>" in p for p in found)


def test_none_with_an_assisted_by_line_fails():
    assert any("contradicts" in p for p in pr_text.check(text("AI use: none", ASSISTED)))


@pytest.mark.parametrize("line", ["AI use: Claude", "AI use: x."])
def test_a_too_short_line_fails(line):
    assert any("must be `none`" in p for p in pr_text.check(text(line, ASSISTED)))


def test_an_empty_line_fails():
    assert any("must be `none`" in p for p in pr_text.check(text("AI use:", ASSISTED)))


def test_no_line_and_two_lines_fail():
    assert any("found 0" in p for p in pr_text.check("## AI use\n\nnothing\n"))
    both = text("AI use: none", "\nAI use: none\n")
    assert any("found 2" in p for p in pr_text.check(both))


def test_a_line_inside_a_comment_does_not_count():
    assert any("found 0" in p for p in pr_text.check("<!--\nAI use: none\n-->\n"))


def test_the_unedited_template_fails():
    assert pr_text.check(TEMPLATE.read_text(encoding="utf-8"))


def test_the_template_filled_in_passes():
    filled = TEMPLATE.read_text(encoding="utf-8").replace("\nAI use:\n", "\nAI use: none\n")
    filled = filled.replace("\n| | | |\n", "\n| qwen3-8b | Q4_K_M | decode |\n")
    assert pr_text.check(filled) == []


def test_trailer_rules_apply_to_the_text():
    body = text("AI use: Claude Code, first draft", ASSISTED) + "\nCo-Authored-By: Claude <noreply@anthropic.com>\n"
    found = pr_text.check(body)
    assert len(found) == 1
    assert "pull request text" in found[0]


def test_main_reads_env_and_skips_dependabot(monkeypatch, capsys):
    monkeypatch.setenv("PR_BODY", "bump")
    monkeypatch.delenv("PR_AUTHOR", raising=False)
    assert pr_text.main([]) == 1
    monkeypatch.setenv("PR_AUTHOR", "dependabot[bot]")
    assert pr_text.main([]) == 0
    monkeypatch.setenv("PR_AUTHOR", "someone")
    monkeypatch.setenv("PR_BODY", text("AI use: none"))
    assert pr_text.main([]) == 0
    assert "ok" in capsys.readouterr().out


def closing(title: str, tail: str, line: str = "AI use: none") -> list[str]:
    return pr_text.check(text(line, tail), title)


@pytest.mark.parametrize("keyword", ["Closes", "fixes", "Resolves:", "CLOSES"])
def test_a_closing_keyword_with_the_number_of_the_title_passes(keyword):
    assert closing("fix(server): stop the stall (#12)", f"\n{keyword} #12\n") == []


def test_feat_fix_perf_and_breaking_need_a_closing_keyword():
    for title in ("feat: add x (#3)", "fix: y (#3)", "perf(gpu): z (#3)", "refactor!: w (#3)"):
        assert any("Closes #3" in p for p in closing(title, "")), title


def test_other_types_need_no_closing_keyword():
    for title in ("docs: fix a typo", "ci: tweak (#3)", "test: add (#3)", "chore: x", "build(deps): y", "revert: z"):
        assert closing(title, "") == [], title


def test_the_number_must_match_the_title():
    assert any("does not close #3" in p for p in closing("fix: y (#3)", "\nCloses #4\n"))


def test_the_template_line_without_a_number_does_not_count():
    assert closing("fix: y (#3)", "\nCloses #\n")


def test_a_keyword_inside_a_comment_does_not_count():
    assert closing("fix: y (#3)", "\n<!-- Closes #3 -->\n")


def test_main_reads_the_title(monkeypatch):
    monkeypatch.delenv("PR_AUTHOR", raising=False)
    monkeypatch.setenv("PR_TITLE", "feat: add x (#3)")
    monkeypatch.setenv("PR_BODY", text("AI use: none"))
    assert pr_text.main([]) == 1
    monkeypatch.setenv("PR_BODY", text("AI use: none", "\nCloses #3\n"))
    assert pr_text.main([]) == 0


class Response:
    def __init__(self, labels):
        self.payload = json.dumps({"labels": [{"name": n} for n in labels]}).encode()

    def __enter__(self):
        return io.BytesIO(self.payload)

    def __exit__(self, *exc):
        return False


def fake_api(monkeypatch, outcome):
    def urlopen(request, timeout):
        if isinstance(outcome, Exception):
            raise outcome
        return Response(outcome)

    monkeypatch.setattr(pr_text.urllib.request, "urlopen", urlopen)


def accepted(title="feat: add x (#3)", association="NONE"):
    return pr_text.accepted_problems(title, association, "o/r", "token")


def test_an_outside_feat_needs_the_accepted_label(monkeypatch):
    fake_api(monkeypatch, ["bug"])
    found, warnings = accepted()
    assert warnings == [] and "no label `accepted`" in found[0]
    fake_api(monkeypatch, ["bug", "accepted"])
    assert accepted() == ([], [])
    assert accepted("perf: faster (#3)") == ([], [])


def test_maintainers_and_other_types_skip_the_label_check(monkeypatch):
    fake_api(monkeypatch, [])
    for association in ("OWNER", "MEMBER", "COLLABORATOR"):
        assert accepted(association=association) == ([], [])
    assert accepted("fix: y (#3)") == ([], [])
    assert accepted("docs: z") == ([], [])


def test_a_missing_issue_is_a_finding_and_a_network_error_a_warning(monkeypatch):
    fake_api(monkeypatch, urllib.error.HTTPError("u", 404, "nf", {}, None))
    assert "does not exist" in accepted()[0][0]
    fake_api(monkeypatch, urllib.error.HTTPError("u", 403, "rate", {}, None))
    found, warnings = accepted()
    assert found == [] and "HTTP 403" in warnings[0]
    fake_api(monkeypatch, urllib.error.URLError("offline"))
    assert accepted()[0] == []


def test_main_online_reports_a_missing_label(monkeypatch):
    fake_api(monkeypatch, [])
    monkeypatch.delenv("PR_AUTHOR", raising=False)
    monkeypatch.setenv("PR_AUTHOR_ASSOCIATION", "NONE")
    monkeypatch.setenv("PR_TITLE", "feat: add x (#3)")
    monkeypatch.setenv("PR_BODY", text("AI use: none", "\nCloses #3\n"))
    assert pr_text.main(["--online"]) == 1
    assert pr_text.main([]) == 0


@pytest.mark.parametrize("section", ["n/a", "N/A.", "| Model | File | Measured |\n|---|---|---|\n| m | q | decode |"])
def test_a_models_section_with_content_passes(section):
    body = text("AI use: none").replace("n/a\n\n## AI use", f"{section}\n\n## AI use")
    assert pr_text.check(body) == []


@pytest.mark.parametrize("section", ["", "| Model | File | Measured |\n|---|---|---|\n| | | |", "<!-- n/a -->"])
def test_an_empty_models_section_fails(section):
    body = text("AI use: none").replace("n/a\n\n## AI use", f"{section}\n\n## AI use")
    assert any("Models" in p for p in pr_text.check(body))


def test_a_missing_models_section_fails():
    body = text("AI use: none").replace("## Models\n\nn/a\n\n", "")
    assert any("`## Models` is missing" in p for p in pr_text.check(body))
