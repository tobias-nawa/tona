"""tools/pr_bot.py: the text of the pull request comment is fixed, complete and safe against outsider text."""

import pr_bot
import pytest

BODY = """Closes #14

## What and why

x

Change type: none

## Prior art

n/a

## Models

n/a

AI use: none
"""
TITLE = "ci: pull request bot (#14)"


def comment(title=TITLE, body=BODY, paths=("docs/a.md",), association="MEMBER"):
    return pr_bot.build_comment(title, body, list(paths), association)


def test_first_line_is_fixed_and_a_clean_pull_request_says_so():
    text = comment()
    assert text.startswith(pr_bot.FIRST_LINE + "\n")
    assert "All text checks pass" in text
    assert "Welcome" not in text


@pytest.mark.parametrize(
    ("title", "body", "gate", "needle"),
    [
        ("Fix things", BODY, "#### Title", "Example: `fix(server)"),
        (TITLE, BODY.replace("AI use: none", ""), "#### Pull request text", "AI use: none"),
        (TITLE, BODY.replace("Change type: none", ""), "#### Prior art", "Change type: technique"),
    ],
)
def test_each_failed_gate_gets_its_findings_and_an_example(title, body, gate, needle):
    text = comment(title=title, body=body)
    assert gate in text
    assert needle in text
    assert "All text checks pass" not in text


def test_engine_code_with_change_type_none_fails_prior_art():
    text = comment(paths=["engine/gpu/k_mega.hip"])
    assert "#### Prior art" in text


@pytest.mark.parametrize("association", ["FIRST_TIME_CONTRIBUTOR", "FIRST_TIMER", "NONE"])
def test_first_contribution_gets_the_welcome_with_three_links(association):
    text = comment(association=association)
    assert text.count("Welcome") == 1
    assert "CONTRIBUTING.md" in text and "scripts/setup.sh --no-gpu" in text and "`just check`" in text
    assert text.count("](https://github.com/") == 3


@pytest.mark.parametrize("association", ["MEMBER", "OWNER", "CONTRIBUTOR", "COLLABORATOR", ""])
def test_everyone_else_gets_no_welcome(association):
    assert "Welcome" not in comment(association=association)


@pytest.mark.parametrize(
    ("path", "needle"),
    [
        ("engine/gpu/k_mega.hip", "golden checks, bit-exactness and speed"),
        ("engine/cpu/gemm.cpp", "golden checks, bit-exactness and speed"),
        ("engine/server/generator.cpp", "Engine host code"),
        ("engine/backend/metal/x.mm", "Another backend"),
        ("engine/server/kv_cache.cpp", "long-context run"),
        ("engine/server/pinned_pool.cpp", "pressure series"),
        ("installer/main.cpp", "no GPU run"),
        ("scripts/ab.py", "just test-py"),
        ("docs/a.md", "gates only"),
        ("toolchain/rocm.lock", "all checks"),
        (".github/workflows/pr-bot.yml", "reads the diff"),
        ("tools/gates/pr_text.py", "reads the diff"),
    ],
)
def test_areas_name_what_a_maintainer_runs(path, needle):
    assert needle in comment(paths=[path])


def test_the_exact_list_comes_from_select_checks():
    assert "python3 tools/select_checks.py" in comment()


def test_outsider_text_stays_inside_code_spans():
    evil = "@maintainer [click](http://x.example) <img src=x> `` \n\n## injected"
    text = comment(title=evil)
    for line in text.splitlines():
        if "@maintainer" in line:
            assert line.startswith("- `") and line.endswith("`")
            assert line.count("`") == 2
    assert "\n## injected" not in text


def test_a_long_finding_is_cut():
    assert len(pr_bot.code("x" * 5000)) <= pr_bot.MAX_FINDING + 2


def test_main_reads_the_files_and_skips_bots(tmp_path, capsys):
    (tmp_path / "title.txt").write_text(TITLE)
    (tmp_path / "body.md").write_text(BODY)
    (tmp_path / "files.txt").write_text("docs/a.md\nscripts/ab.py\n")
    (tmp_path / "association.txt").write_text("FIRST_TIMER")
    (tmp_path / "author.txt").write_text("someone")
    out = tmp_path / "comment.md"
    assert pr_bot.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    text = out.read_text()
    assert "Welcome" in text and "just test-py" in text
    out.unlink()
    (tmp_path / "author.txt").write_text("dependabot[bot]")
    assert pr_bot.main(["--dir", str(tmp_path), "--out", str(out)]) == 0
    assert not out.exists()
