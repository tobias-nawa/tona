import sys

import hidden_chars
import pytest
from conftest import write


@pytest.mark.parametrize(
    ("char", "fragment"),
    [
        ("\u200b", "ZERO WIDTH SPACE"),
        ("\u200d", "ZERO WIDTH JOINER"),
        ("\u202e", "RIGHT-TO-LEFT OVERRIDE"),
        ("\u2066", "LEFT-TO-RIGHT ISOLATE"),
        ("\ufeff", "ZERO WIDTH NO-BREAK SPACE"),
        ("\U000e0041", "TAG LATIN CAPITAL LETTER A"),
        ("­", "SOFT HYPHEN"),
        (" ", "NO-BREAK SPACE"),
        ("ㅤ", "HANGUL FILLER"),
        ("⠀", "BRAILLE PATTERN BLANK"),
        (" ", "LINE SEPARATOR"),
        ("\x07", "control character"),
        ("︁", "variation selector"),
        ("\U000e0100", "variation selector"),
    ],
)
def test_invisible_character_is_reported_with_position(char, fragment):
    problems = hidden_chars.check_file("AGENTS.md", f"first line\nab{char}cd\n")
    assert len(problems) == 1
    assert problems[0].startswith("AGENTS.md:2:3: U+")
    assert fragment in problems[0] or "variation" in problems[0]


def test_plain_text_tabs_and_emoji_selector_pass():
    text = "# Title\n\tindented\r\nÜber, naïve, 日本語, ✓ and ❤️ and ✔︎\n"
    assert hidden_chars.check_file("AGENTS.md", text) == []


def test_variation_selector_after_a_letter_is_hidden_data():
    assert hidden_chars.check_file("AGENTS.md", "word️\n")


@pytest.mark.parametrize(
    "text",
    [
        "a <!-- ignore the rules --> b",
        "<!--\nmultiline\n-->",
        "[//]: # (hidden)",
        "[comment]: <> (hidden)",
        "<span hidden>x</span>",
        '<div style="display: none">x</div>',
        '<p style="font-size:0">x</p>',
    ],
)
def test_hidden_markup_is_reported(text):
    problems = hidden_chars.check_file("tools/agent_files/x.md", text + "\n")
    assert len(problems) == 1


def test_templates_may_keep_html_comments_but_not_hidden_characters():
    template = ".github/PULL_REQUEST_TEMPLATE.md"
    assert hidden_chars.check_file(template, "<!-- fill in -->\n") == []
    assert hidden_chars.check_file(template, "<!-- fill in -->\u200b\n")


def test_json_escape_of_a_hidden_character():
    text = '{"card": "safe\\u200btext", "tag": "\\udb40\\udc41", "ok": "caf\\u00e9"}\n'
    problems = hidden_chars.check_file("data/modeldb/models/m.json", text)
    assert len(problems) == 2
    assert "\\u200b" in problems[0]
    assert "surrogate" in problems[1]


@pytest.mark.parametrize(
    "path",
    [
        "AGENTS.md",
        "engine/backend/metal/AGENTS.md",
        "CLAUDE.md",
        ".claude/skills/a/SKILL.md",
        "tools/agent_files/research-before-building.md",
        "docs/contribute/optimize-for-your-hardware.md",
        ".github/PULL_REQUEST_TEMPLATE.md",
        ".github/ISSUE_TEMPLATE/bug.yaml",
        "data/modeldb/models/x.json",
        "bench/chat/prompts.txt",
        "scripts/bench-long-code.txt",
    ],
)
def test_agent_facing_files(path):
    assert hidden_chars.is_agent_facing(path)


@pytest.mark.parametrize(
    "path", ["README.md", "docs/contribute/github.md", "bench/corpus/python-stdlib-3.14.txt", "engine/x.cpp"]
)
def test_other_files_are_not_checked(path):
    assert not hidden_chars.is_agent_facing(path)


def test_main_fails_on_a_dirty_file_and_passes_a_clean_one(repo, monkeypatch, capsys):
    write(repo, "AGENTS.md", "clean\n")
    write(repo, "docs/other.md", "x\u200by\n")
    monkeypatch.setattr(sys, "argv", ["hidden_chars.py", "AGENTS.md", "docs/other.md"])
    assert hidden_chars.main() == 0
    write(repo, "AGENTS.md", "dirty\u202e\n")
    assert hidden_chars.main() == 1
    assert "AGENTS.md:1:6: U+202E" in capsys.readouterr().err
