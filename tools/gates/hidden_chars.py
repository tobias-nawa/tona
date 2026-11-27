"""Hidden-content gate: text a reviewer cannot see but an agent reads, in the files that agents load as instructions.

Usage: hidden_chars.py [FILE...]  (no arguments: every tracked agent-facing file)

Agent-facing files: every AGENTS.md and CLAUDE.md, skills (`SKILL.md`, `.claude/`, `.agents/`, `.cursor/`,
`.github/skills/`, `tools/agent_files/`), `docs/contribute/optimize-for-your-hardware.md`, the pull request and issue
templates, `data/modeldb/` and the benchmark prompts (`bench/` and `scripts/` text, Markdown and JSON, not the vendored
corpus).
An agent reads these with the authority of its own instructions, so a character that renders as nothing is an attack
until shown otherwise (Pillar Security "Rules File Backdoor", 2025-03).

Fails, with file:line:column, on
- zero-width, bidirectional, tag-block, soft-hyphen, BOM and other format characters (Unicode category Cf), line and
  paragraph separators, control characters other than tab, newline and carriage return, private-use and unassigned
  code points;
- invisible letters and spaces: non-ASCII spaces (NBSP, em space, ideographic space), the Hangul fillers, the braille
  blank, the combining grapheme joiner;
- variation selectors that do not follow an emoji or symbol (they can carry data in the text of a plain word);
- the same characters written as a JSON escape (`\\u200b`) in a `.json` file, which a JSON parser turns into the real
  character in front of the agent;
- HTML comments, Markdown reference comments (`[//]: #`) and elements that hide their text (`hidden`, `display:none`,
  zero font size).

Exception, with its reason: the pull request and issue templates keep fill-in guidance in HTML comments (GitHub shows it
only in the editor). `prior_art.py` and `pr_text.py` strip those comments before they read a pull request text, and
`.github/` is owned by the maintainer (CODEOWNERS), who reads the raw bytes of every change there.
"""

import re
import sys
import unicodedata
from pathlib import PurePosixPath

from common import gate_files, repo_root, report

SUFFIXES = (".md", ".txt", ".json", ".toml", ".yaml", ".yml", ".tsv")
AGENT_NAMES = ("AGENTS.md", "CLAUDE.md", "SKILL.md")
AGENT_DIRS = (".claude/", ".agents/", ".cursor/", ".github/skills/", "tools/agent_files/", "data/modeldb/")
AGENT_FILES = ("docs/contribute/optimize-for-your-hardware.md", ".github/copilot-instructions.md")
TEMPLATE_PREFIXES = (".github/PULL_REQUEST_TEMPLATE", ".github/ISSUE_TEMPLATE/")
PROMPT_DIRS = ("bench/", "scripts/")
# The corpus is CPython's own source text, vendored for token statistics; not an instruction channel.
PROMPT_EXCLUDED = ("bench/corpus/",)

# Letters, marks and symbols that draw nothing (the categories alone do not say so)
INVISIBLE_OTHER = {
    0x034F: "COMBINING GRAPHEME JOINER",
    0x115F: "HANGUL CHOSEONG FILLER",
    0x1160: "HANGUL JUNGSEONG FILLER",
    0x17B4: "KHMER VOWEL INHERENT AQ",
    0x17B5: "KHMER VOWEL INHERENT AA",
    0x180E: "MONGOLIAN VOWEL SEPARATOR",
    0x2800: "BRAILLE PATTERN BLANK",
    0x3164: "HANGUL FILLER",
    0xFFA0: "HALFWIDTH HANGUL FILLER",
}
VARIATION = [(0xFE00, 0xFE0F), (0xE0100, 0xE01EF)]
EMOJI_PRESENTATION = (0xFE0E, 0xFE0F)
JSON_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
# Tag characters U+E0000..E007F and the variation selectors above U+FFFF are written as surrogate pairs starting here
HIGH_SURROGATE_HIDDEN = (0xDB40, 0xDB43)

HIDDEN_MARKUP = [
    (re.compile(r"<!--"), "HTML comment"),
    (re.compile(r"^\s*\[(?://|comment)\]\s*:\s*(?:#|<>)", re.MULTILINE), "Markdown reference comment"),
    (re.compile(r"<[a-zA-Z][^>]*\bhidden\b[^>]*>"), "element with the hidden attribute"),
    (
        re.compile(
            r"<[a-zA-Z][^>]*\bstyle\s*=\s*[\"'][^\"']*(?:display\s*:\s*none|font-size\s*:\s*0|visibility\s*:\s*hidden)"
        ),
        "element styled to be invisible",
    ),
]


def is_agent_facing(path: str) -> bool:
    name = PurePosixPath(path).name
    if name in AGENT_NAMES or path in AGENT_FILES or path.startswith(AGENT_DIRS + TEMPLATE_PREFIXES):
        return True
    return path.startswith(PROMPT_DIRS) and not path.startswith(PROMPT_EXCLUDED) and path.endswith(SUFFIXES)


def comments_allowed(path: str) -> bool:
    return path.startswith(TEMPLATE_PREFIXES)


def describe(cp: int) -> str:
    return unicodedata.name(chr(cp), f"code point U+{cp:04X}")


def is_variation(cp: int) -> bool:
    return any(lo <= cp <= hi for lo, hi in VARIATION)


def hidden_reason(cp: int, previous: str) -> str | None:
    """Why this code point is hidden, or None when it is an ordinary character."""
    if cp in EMOJI_PRESENTATION or is_variation(cp):
        after_symbol = bool(previous) and unicodedata.category(previous) in ("So", "Sk", "Sm", "No", "Nd")
        return None if cp in EMOJI_PRESENTATION and after_symbol else "variation selector outside an emoji"
    if cp in INVISIBLE_OTHER:
        return "invisible character"
    category = unicodedata.category(chr(cp))
    if category == "Cc":
        return None if chr(cp) in "\t\n\r" else "control character"
    if category in ("Cf", "Zl", "Zp", "Co", "Cn", "Cs"):
        return {
            "Cf": "format character (zero-width, bidi, tag, BOM)",
            "Zl": "line separator",
            "Zp": "paragraph separator",
        }.get(category, "private-use, unassigned or surrogate code point")
    if category == "Zs" and cp != 0x20:
        return "non-ASCII space"
    return None


def position(text: str, offset: int) -> tuple[int, int]:
    line = text.count("\n", 0, offset) + 1
    return line, offset - (text.rfind("\n", 0, offset) + 1) + 1


def scan_characters(name: str, text: str) -> list[str]:
    problems = []
    previous = ""
    for offset, ch in enumerate(text):
        reason = hidden_reason(ord(ch), previous)
        previous = ch
        if reason:
            line, col = position(text, offset)
            problems.append(f"{name}:{line}:{col}: U+{ord(ch):04X} {describe(ord(ch))}: {reason}")
    return problems


def scan_json_escapes(name: str, text: str) -> list[str]:
    problems = []
    for match in JSON_ESCAPE.finditer(text):
        cp = int(match.group(1), 16)
        hidden = hidden_reason(cp, "") if not 0xD800 <= cp <= 0xDFFF else None
        if 0xD800 <= cp <= 0xDFFF:
            hidden = (
                "tag or variation selector as a surrogate pair"
                if HIGH_SURROGATE_HIDDEN[0] <= cp <= HIGH_SURROGATE_HIDDEN[1]
                else None
            )
        if hidden:
            line, col = position(text, match.start())
            problems.append(f"{name}:{line}:{col}: JSON escape {match.group(0)}: {hidden}")
    return problems


def scan_markup(name: str, text: str) -> list[str]:
    problems = []
    for pattern, what in HIDDEN_MARKUP:
        for match in pattern.finditer(text):
            line, col = position(text, match.start())
            problems.append(f"{name}:{line}:{col}: {what}; say it in visible text or leave it out")
    return problems


def check_file(name: str, text: str) -> list[str]:
    problems = scan_characters(name, text)
    if name.endswith(".json"):
        problems += scan_json_escapes(name, text)
    if not comments_allowed(name):
        problems += scan_markup(name, text)
    return problems


def main() -> int:
    root = repo_root()
    files, _ = gate_files(sys.argv[1:], SUFFIXES)
    files = [f for f in files if is_agent_facing(f)]
    if not sys.argv[1:] and not files:
        print("hidden_chars: no agent-facing files found", file=sys.stderr)
        return 2
    problems = []
    for f in files:
        problems += check_file(f, (root / f).read_text(encoding="utf-8"))
    return report("hidden_chars", problems, len(files), "See the header of tools/gates/hidden_chars.py.")


if __name__ == "__main__":
    sys.exit(main())
