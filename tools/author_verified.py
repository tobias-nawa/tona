#!/usr/bin/env python3
"""Reads the AI-use checkbox of an issue form and says which author label applies.

Usage: author_verified.py state FILE     prints `verified`, `unverified` or `none` for the issue body in FILE
       author_verified.py comment        prints the one comment the bot writes when it first sets `author-unverified`

The body is written by an outsider: it is parsed here and never reaches a shell or a template
(docs/maintain/maintainer.md, Automations). `none` is for forms without the box (backend proposal, blank issue).
"""

import re
import sys
from pathlib import Path

# The comment starts with this line; the workflow looks for it so the bot never comments twice on one issue.
COMMENT_FIRST_LINE = "Author check (automated message)"
COMMENT = f"""{COMMENT_FIRST_LINE}

Thanks for opening this issue. Using AI tools is fine here. The box in the "AI use" section is still unticked:
please read the text once yourself, correct what is wrong, and tick the box. The label `author-unverified` then
changes to
`author-verified` by itself. Nothing is closed because of it."""

SECTION = re.compile(r"^#{2,3}\s*AI use\s*$", re.IGNORECASE | re.MULTILINE)
NEXT_HEADING = re.compile(r"^#{1,3}\s", re.MULTILINE)
CHECKBOX = re.compile(r"^\s*[-*]\s*\[([ xX])\]", re.MULTILINE)


def author_state(body: str) -> str:
    """`verified` when every box of the AI-use section is ticked, `unverified` when one is not, else `none`."""
    start = SECTION.search(body)
    if not start:
        return "none"
    rest = body[start.end() :]
    end = NEXT_HEADING.search(rest)
    boxes = CHECKBOX.findall(rest[: end.start()] if end else rest)
    if not boxes:
        return "none"
    return "verified" if all(mark != " " for mark in boxes) else "unverified"


def main(argv: list[str]) -> int:
    if len(argv) == 3 and argv[1] == "state":
        print(author_state(Path(argv[2]).read_text(encoding="utf-8")))
        return 0
    if len(argv) == 2 and argv[1] == "comment":
        print(COMMENT)
        return 0
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
