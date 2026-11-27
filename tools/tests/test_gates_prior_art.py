import prior_art
import pytest

SEARCH12 = (
    "Search: tools/papers/scan.py topics v1; topics: moe-offload, kv-cache; terms: expert cache; "
    "window: 2025-11-11..2026-11-11 (12 months); sources: arXiv, GitHub PRs and issues."
)
SEARCH3 = "Search: tools/papers/scan.py topics v1; terms: hang on resume; window: 3 months; sources: GitHub issues."
LINK = "- [Paper](https://arxiv.org/abs/2608.07911), status: read. Adopted: nothing. Not adopted: needs training.\n"
ENGINE = ["engine/gpu/kernel.hip"]
HOT = ["engine/server/generator.cpp"]


def pr(kind: str, section: str | None, extra: str = "") -> str:
    text = f"## What and why\n\nx\n\nChange type: {kind}\n\n{extra}"
    if section is not None:
        text += f"### Prior art\n\n{section}\n\n## Test conditions\n\nstuff\n"
    return text


def problems(text: str, files: list[str]) -> list[str]:
    return prior_art.check(text, files)


def test_technique_with_link_and_search_passes():
    assert problems(pr("technique", LINK + SEARCH12), ENGINE) == []


def test_technique_with_nothing_found_and_search_passes():
    assert problems(pr("technique", "nothing found.\n" + SEARCH12), ENGINE) == []


def test_bugfix_three_months_passes():
    assert problems(pr("bugfix", "- https://github.com/ggml-org/llama.cpp/issues/1 open\n" + SEARCH3), HOT) == []


def test_bugfix_nothing_found_with_terms_passes():
    assert problems(pr("bugfix", "nothing found\n" + SEARCH3), HOT) == []


def test_technique_with_three_month_window_fails():
    found = problems(pr("technique", LINK + SEARCH3), ENGINE)
    assert any("shorter than 12 months" in p for p in found)


def test_date_range_shorter_than_required_fails():
    short = SEARCH12.replace("2025-11-11..2026-11-11 (12 months)", "2026-08-11..2026-11-10")
    assert any("shorter" in p for p in problems(pr("technique", LINK + short), ENGINE))


def test_missing_tool_version_fails():
    search = SEARCH12.replace("scan.py topics v1", "arxiv search")
    assert any("topics version" in p for p in problems(pr("technique", LINK + search), ENGINE))


def test_missing_topics_fails():
    search = "Search: tools/papers/scan.py topics v1; window: 12 months"
    assert any("topics or terms" in p for p in problems(pr("technique", LINK + search), ENGINE))


def test_missing_window_fails():
    search = "Search: tools/papers/scan.py topics v1; topics: kv-cache"
    assert any("no window" in p for p in problems(pr("technique", LINK + search), ENGINE))


def test_no_link_and_no_nothing_found_fails():
    found = problems(pr("technique", "I looked around.\n" + SEARCH12), ENGINE)
    assert any("no link" in p for p in found)


def test_link_only_inside_search_line_does_not_count():
    search = SEARCH12 + " See https://example.com"
    assert any("no link" in p for p in problems(pr("technique", search), ENGINE))


def test_missing_section_or_na_fails_for_engine_code():
    assert any("missing or empty" in p for p in problems(pr("technique", None), ENGINE))
    assert any("missing or empty" in p for p in problems(pr("bugfix", "n/a"), ENGINE))


def test_none_with_engine_code_fails():
    found = problems(pr("none", "n/a"), ["docs/a.md", *ENGINE])
    assert any("touches engine code" in p for p in found)


@pytest.mark.parametrize(
    "path", ["engine/model/quant.cpp", "engine/server/api.cpp", "scripts/golden.sh", "toolchain/rocm.lock"]
)
def test_none_refused_for_engine_class_paths(path):
    assert problems(pr("none", "n/a"), [path])


@pytest.mark.parametrize(
    "path",
    [
        "docs/a.md",
        "README.md",
        "research/papers.md",
        "tools/gates/x.py",
        "engine/tests/test_a.cpp",
        "installer/main.cpp",
        ".github/workflows/check.yml",
    ],
)
def test_none_allowed_without_engine_code(path):
    assert problems(pr("none", "n/a"), [path]) == []


def test_any_type_without_engine_code_needs_no_section():
    assert problems(pr("technique", None), ["docs/a.md", "tools/x.py"]) == []


def test_backend_needs_docs_backends():
    assert any("docs/backends/" in p for p in problems(pr("backend", LINK + SEARCH12), ["engine/backend/vk/a.cpp"]))
    ok = LINK + "See docs/backends/vulkan.md.\n" + SEARCH12
    assert problems(pr("backend", ok), ["engine/backend/vk/a.cpp"]) == []


def test_missing_or_invalid_type_fails():
    assert problems("## Prior art\n" + LINK + SEARCH12, ENGINE)
    template_line = pr("bugfix | technique | backend | none", None)
    assert any("exactly one" in p for p in problems(template_line, ENGINE))
    assert any("exactly one" in p for p in problems(pr("rewrite", None), ["docs/a.md"]))


def test_type_in_comments_is_ignored_and_markup_is_tolerated():
    text = "<!-- Change type: none -->\n- **Change type**: `technique`\n### Prior art\n" + LINK + SEARCH12 + "\n"
    assert problems(text, ENGINE) == []


def test_section_ends_at_next_heading():
    text = pr("technique", LINK + SEARCH12 + "\n") + "\n## Later\n\nnothing found\n"
    assert problems(text, ENGINE) == []
    only_later = "Change type: technique\n### Prior art\n\n## Later\n" + LINK + SEARCH12 + "\n"
    assert any("missing or empty" in p for p in problems(only_later, ENGINE))


def test_main_reads_env_and_files(monkeypatch, capsys):
    monkeypatch.setenv("PR_BODY", pr("technique", LINK + SEARCH12))
    assert prior_art.main(["--files", *ENGINE]) == 0
    monkeypatch.setenv("PR_BODY", "")
    assert prior_art.main(["--files", *ENGINE]) == 1
    assert "Change type" in capsys.readouterr().err


def test_main_body_file_and_git_diff(repo, monkeypatch, tmp_path):
    from conftest import commit_all, git, write

    write(repo, "README.md", "a\n")
    commit_all(repo)
    git(repo, "checkout", "-q", "-b", "topic")
    write(repo, "engine/gpu/k.hip", "x\n")
    commit_all(repo)
    body = tmp_path.parent / "body.txt"
    body.write_text(pr("technique", LINK + SEARCH12))
    assert prior_art.main(["--base", "main", "--body-file", str(body)]) == 0
    body.write_text(pr("none", "n/a"))
    assert prior_art.main(["--base", "main", "--body-file", str(body)]) == 1
    assert prior_art.main(["--body-file", str(body)]) == 2


def test_unfilled_pull_request_template_fails_for_engine_code():
    template = (prior_art.Path(__file__).resolve().parents[2] / ".github/PULL_REQUEST_TEMPLATE.md").read_text()
    assert any("exactly one" in p for p in problems(template, ENGINE))
    assert problems(template.replace("Change type: <!--", "Change type: none <!--", 1), ["docs/a.md"]) == []


def test_refactor_with_engine_code_needs_proof_hint():
    assert any("bit-identical" in p for p in problems(pr("refactor", None), ENGINE))
    proof = "Identical tokens with TONA_DETERMINISTIC=1: SHAs equal.\n"
    assert problems(pr("refactor", None, proof), ENGINE) == []
    assert problems(pr("refactor", None, "isa-diff: identical\n"), ENGINE) == []


def test_refactor_proof_in_a_comment_does_not_count():
    assert problems(pr("refactor", None, "<!-- bit-identical -->\n"), ENGINE)


def test_refactor_without_engine_code_needs_nothing():
    assert problems(pr("refactor", None), ["docs/a.md"]) == []


def test_a_dependabot_pull_request_is_skipped(monkeypatch):
    monkeypatch.setenv("PR_AUTHOR", "dependabot[bot]")
    monkeypatch.setenv("PR_BODY", "Bumps the python group.")
    assert prior_art.main(["--files", "engine/server/x.cpp"]) == 0
