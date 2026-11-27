#!/usr/bin/env python3 -I
"""Writes a pool of source files for run.py from source trees: corpus.py OUT ROOT [ROOT ...] [--max-mb N].
Each file becomes a unit '// ===== path =====' + content. 6 MB of code are about 1.9 M tokens."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import haystack

EXTS = {".cpp", ".h", ".hpp", ".c", ".hip", ".py", ".sh", ".cs", ".ts", ".go", ".rs", ".js", ".cc"}


def main() -> None:
    args = sys.argv[1:]
    cap = 8
    if "--max-mb" in args:
        i = args.index("--max-mb")
        cap = int(args[i + 1])
        del args[i : i + 2]
    if len(args) < 2:
        raise SystemExit(__doc__)
    text = haystack.pool_from_roots(args[1:], EXTS, cap * 1_000_000)
    Path(args[0]).write_text(text)
    print(f"{len(haystack.split_pool(text))} files, {len(text)} characters")


if __name__ == "__main__":
    main()
