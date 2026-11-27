"""Markdown link gate: every relative link and every `#anchor` in a tracked Markdown file must resolve.

Usage: md_links.py [FILE...]  (no arguments: all tracked Markdown files)

External links (http, https, mailto) are not fetched; this gate runs offline. Links inside fenced code blocks and
inline code are examples, not links, and are skipped. Anchors follow the GitHub rule for headings.
"""

import re
import sys
from pathlib import Path
from urllib.parse import unquote

from common import gate_files, repo_root, report, tracked_files

LINK = re.compile(r"!?\[[^\]\n]*\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
HEADING = re.compile(r"^#{1,6}\s+(.*?)\s*#*\s*$")
FENCE = re.compile(r"^\s*(```|~~~)")
INLINE_CODE = re.compile(r"`[^`\n]*`")
EXTERNAL = ("http://", "https://", "mailto:", "tel:")


def slug(heading: str) -> str:
    """GitHub's heading anchor: lower case, markup and punctuation dropped, spaces to hyphens."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", heading)
    text = text.replace("`", "").replace("*", "").strip().lower()
    text = re.sub(r"[^\w\- ]", "", text)
    return text.replace(" ", "-")


def prose_lines(text: str):
    """(line number, line) outside fenced code blocks."""
    fenced = False
    for n, line in enumerate(text.splitlines(), 1):
        if FENCE.match(line):
            fenced = not fenced
            continue
        if not fenced:
            yield n, line


def anchors(path: Path) -> set[str]:
    found: set[str] = set()
    seen: dict[str, int] = {}
    for _, line in prose_lines(path.read_text(encoding="utf-8", errors="replace")):
        m = HEADING.match(line)
        if m:
            base = slug(m.group(1))
            count = seen.get(base, 0)
            seen[base] = count + 1
            found.add(base if count == 0 else f"{base}-{count}")
    return found


def check_file(root: Path, rel: str, cache: dict[Path, set[str]]) -> list[str]:
    problems = []
    source = root / rel
    for n, line in prose_lines(source.read_text(encoding="utf-8", errors="replace")):
        for target in LINK.findall(INLINE_CODE.sub("", line)):
            if target.startswith(EXTERNAL):
                continue
            path_part, _, anchor = target.partition("#")
            dest = source if not path_part else (source.parent / unquote(path_part)).resolve()
            if not dest.exists():
                problems.append(f"{rel}:{n}: broken link {target}")
            elif anchor and dest.suffix == ".md":
                if dest not in cache:
                    cache[dest] = anchors(dest)
                if unquote(anchor).lower() not in cache[dest]:
                    problems.append(f"{rel}:{n}: no heading for anchor {target}")
    return problems


def main() -> int:
    root = repo_root().resolve()
    if sys.argv[1:]:
        files, _ = gate_files(sys.argv[1:], (".md",))
    else:
        files = tracked_files((".md",))
    cache: dict[Path, set[str]] = {}
    problems = []
    for rel in files:
        problems += check_file(root, rel, cache)
    return report("md-links", problems, len(files), "Fix the link or the target.")


if __name__ == "__main__":
    sys.exit(main())
