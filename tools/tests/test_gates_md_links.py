import md_links


def check(tmp_path, files: dict[str, str], name: str) -> list[str]:
    for rel, text in files.items():
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return md_links.check_file(tmp_path, name, {})


def test_relative_link_and_anchor_resolve(tmp_path):
    files = {
        "a.md": "[b](docs/b.md#speed-comparison-scriptsabpy)\n",
        "docs/b.md": "## Speed comparison: `scripts/ab.py`\n",
    }
    assert check(tmp_path, files, "a.md") == []


def test_missing_file_is_reported(tmp_path):
    assert check(tmp_path, {"a.md": "[x](nope.md)\n"}, "a.md") == ["a.md:1: broken link nope.md"]


def test_missing_anchor_is_reported(tmp_path):
    problems = check(tmp_path, {"a.md": "[x](#gone)\n# Here\n"}, "a.md")
    assert problems == ["a.md:1: no heading for anchor #gone"]


def test_own_anchor_and_duplicate_headings(tmp_path):
    assert check(tmp_path, {"a.md": "[x](#here) [y](#here-1)\n# Here\n# Here\n"}, "a.md") == []


def test_code_and_external_are_skipped(tmp_path):
    text = "`[x](nope.md)` [e](https://example.org/x)\n```\n[y](nope.md)\n```\n"
    assert check(tmp_path, {"a.md": text}, "a.md") == []


def test_parent_directory_link(tmp_path):
    files = {"docs/a.md": "[r](../README.md)\n", "README.md": "# R\n"}
    assert check(tmp_path, files, "docs/a.md") == []
