#!/usr/bin/env python3
"""Rebuilds preds.json from mini-swe-agent trajectories when the submit marker was not recognised.

    salvage.py RESULT_DIR OUT_PREDS.json

Cause (seen in the first Ornith chunk): the `docker` emulation of Podman prints a notice before the command output,
so the marker COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT was not the first output line, the agent was not stopped and ran
to the step limit. The patch the agent printed with its first submission is the submission; the later steps of such a
run are not used. Without a marker in a trajectory the patch is empty (not submitted). Standard library only.
"""

import json
import re
import sys
from pathlib import Path

MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
OUTPUT = re.compile(r"<output>\n(.*?)(?:</output>|$)", re.S)


def first_submission(messages: list[dict]) -> str:
    for m in messages:
        if m.get("role") != "tool":
            continue
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content)
        found = OUTPUT.search(text)
        body = found.group(1) if found else text
        lines = body.split("\n")
        for i, line in enumerate(lines):
            if line.strip() == MARKER:
                return "\n".join(lines[i + 1 :]).strip() + "\n"
    return ""


def main() -> int:
    preds = {}
    for traj in sorted(Path(sys.argv[1]).glob("*/*.traj.json")):
        data = json.loads(traj.read_text())
        iid = traj.parent.name
        preds[iid] = {"model_patch": first_submission(data["messages"]), "model_name_or_path": "salvaged"}
        print(iid, len(preds[iid]["model_patch"]), "bytes of patch")
    Path(sys.argv[2]).write_text(json.dumps(preds))
    return 0


if __name__ == "__main__":
    sys.exit(main())
