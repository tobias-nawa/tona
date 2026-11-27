#!/usr/bin/env python3 -I
"""Builds the data directory of one new polyglot language from an unpacked Exercism track repository.

    prepare.py --aider DIR --track DIR --lang LANG --out DIR

Selection rule: the practice exercises of the track whose slug occurs in Aider's polyglot copy (any of its languages),
without deprecated exercises (`"status": "deprecated"` in the track's config.json). The same slugs across languages keep
the languages comparable. The result is DIR/LANG/exercises/practice/<slug>, symlinks into the track checkout, the layout
that tasks.load reads; DIR/LANG/selection.txt lists the slugs. Java and C++ come from Aider's copy as they are (its own
selection), so they need no preparation. Standard library only; nothing from the track is executed.
"""

import argparse
import json
import sys
from pathlib import Path


def aider_slugs(aider: Path) -> set[str]:
    dirs = [lang / "exercises/practice" for lang in aider.iterdir()]
    return {d.name for practice in dirs if practice.is_dir() for d in practice.iterdir()}


def deprecated(track: Path) -> set[str]:
    cfg = json.loads((track / "config.json").read_text())
    return {e["slug"] for e in cfg["exercises"]["practice"] if e.get("status") == "deprecated"}


def select(aider: Path, track: Path) -> list[str]:
    practice = {d.name for d in (track / "exercises/practice").iterdir() if d.is_dir()}
    return sorted((practice & aider_slugs(aider)) - deprecated(track))


def link(track: Path, lang: str, out: Path, slugs: list[str]) -> None:
    dest = out / lang / "exercises/practice"
    dest.mkdir(parents=True, exist_ok=False)
    for slug in slugs:
        (dest / slug).symlink_to((track / "exercises/practice" / slug).resolve())
    (out / lang / "selection.txt").write_text("\n".join(slugs) + "\n")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--aider", required=True, type=Path)
    p.add_argument("--track", required=True, type=Path)
    p.add_argument("--lang", required=True)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    slugs = select(a.aider, a.track)
    link(a.track, a.lang, a.out, slugs)
    print(f"{a.lang}: {len(slugs)} exercises selected")
    return 0


if __name__ == "__main__":
    sys.exit(main())
