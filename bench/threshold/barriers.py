#!/usr/bin/python3 -I
"""List the barrier aborts of the server logs of a main run with the request before them.

Usage: barriers.py LABEL...   (logs: WORK/logs/LABEL-server.log)
Prints each abort line and whether the request before it restored a conversation from RAM or the disk.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common


def kind(request_line: str) -> str:
    if "conversation from RAM" in request_line:
        return "restore from RAM"
    if "from disk" in request_line or "checkpoint from" in request_line:
        return "restore from disk"
    return "no restore" if "Request:" in request_line else "unknown"


def main():
    for label in sys.argv[1:]:
        lines = (common.WORK / "logs" / f"{label}-server.log").read_text().splitlines()
        for i, ln in enumerate(lines):
            if re.search(r"barrier aborted", ln):
                prev = next((x for x in reversed(lines[:i]) if "Request:" in x), "")
                print(f"{label}: {ln[:260]}\n   request before: [{kind(prev)}] {prev[:200]}")


if __name__ == "__main__":
    main()
