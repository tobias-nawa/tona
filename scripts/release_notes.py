#!/usr/bin/env python3
"""Print the notes of one version from CHANGELOG.md (docs/maintain/release.md).

    release_notes.py X.Y.Z [--file CHANGELOG.md | --file -]

The notes are the text under `## [X.Y.Z] - date` up to the next `## ` heading. The release flow puts them into the
manifest (what the update dialog of tona shows, and what the signature covers) and into the GitHub release, so a
version without notes is an error and no release is built. The limits are those of tona's manifest parser
(engine/update/manifest.cpp), so notes it would reject are caught here and not after signing.
"""

import argparse
import re
import sys
from pathlib import Path

# kMaxChangelogBytes in engine/update/manifest.cpp
MAX_BYTES = 16384
VERSION = re.compile(r"[0-9]+\.[0-9]+\.[0-9]+")


class NotesError(Exception):
    pass


def notes_of(text: str, version: str) -> str:
    if not VERSION.fullmatch(version):
        raise NotesError(f"version {version!r} is not X.Y.Z")
    heading = re.compile(rf"^## \[{re.escape(version)}\](?: - [0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}})?[ \t]*$")
    lines = text.splitlines()
    start = next((i for i, line in enumerate(lines) if heading.match(line)), None)
    if start is None:
        raise NotesError(f"CHANGELOG.md has no section '## [{version}] - YYYY-MM-DD'")
    body = []
    for line in lines[start + 1 :]:
        if line.startswith("## "):
            break
        body.append(line)
    notes = "\n".join(body).strip("\n")
    if not notes.strip():
        raise NotesError(f"the section of {version} in CHANGELOG.md is empty")
    check(notes)
    return notes


def check(notes: str) -> None:
    if len(notes.encode()) > MAX_BYTES:
        raise NotesError(f"notes are {len(notes.encode())} bytes; tona accepts at most {MAX_BYTES}")
    for char in notes:
        if (ord(char) < 0x20 and char != "\n") or ord(char) == 0x7F:
            raise NotesError(f"notes contain the control character {char!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("version")
    parser.add_argument("--file", default=str(Path(__file__).resolve().parent.parent / "CHANGELOG.md"))
    args = parser.parse_args()
    try:
        text = sys.stdin.read() if args.file == "-" else Path(args.file).read_text(encoding="utf-8")
        print(notes_of(text, args.version))
    except (NotesError, OSError, UnicodeDecodeError) as error:
        print(f"release_notes: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
