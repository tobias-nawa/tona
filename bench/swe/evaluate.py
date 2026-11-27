#!/usr/bin/env python3
"""Grades mini-swe-agent patches on SWE-bench Verified in rootless Podman (the official docker client of the
swebench harness needs a docker socket; this applies the same steps through the podman command line).

    evaluate.py PREDS.json OUT.jsonl

PREDS.json: {instance_id: {"model_patch": "..."}} as written by `mini-extra swebench`. Per instance: start the
instance image, apply the patch (the harness's chain of `git apply` variants), run the harness's own eval script, grade
with the harness's own log parsers (swebench.harness.grading.get_eval_report). One JSON line per instance.
Needs the swebench package (the venv of the suite); the images are pulled beforehand.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

from datasets import load_dataset
from swebench.harness.grading import get_eval_report
from swebench.harness.utils import make_test_spec

TIMEOUT_S = 1800  # harness default for the eval script
APPLY_CHAIN = [
    "git apply --verbose",
    "git apply --verbose --reject",
    "patch --batch --fast -p1 -i",
]


def image_name(instance_id: str) -> str:
    return f"docker.io/swebench/sweb.eval.x86_64.{instance_id.replace('__', '_1776_')}:latest".lower()


def grade(instance: dict, patch: str, work: Path) -> dict:
    spec = make_test_spec(instance)
    (work / "patch.diff").write_text(patch)
    (work / "eval.sh").write_text(spec.eval_script)
    apply = " || (git checkout -- . ; git clean -fd ; ".join(f"{c} /work/patch.diff" for c in APPLY_CHAIN)
    apply += ")" * (len(APPLY_CHAIN) - 1)
    script = f"cd /testbed && ({apply}) && echo PATCH_APPLIED && /bin/bash /work/eval.sh"
    cmd = [
        "podman",
        "run",
        "--rm",
        f"--timeout={TIMEOUT_S}",
        f"--volume={work}:/work:z",
        image_name(instance["instance_id"]),
    ]
    cmd += ["/bin/bash", "-c", script]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S + 120, check=False)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        out = "TIMEOUT"
    log = work / "test_output.txt"
    log.write_text(out)
    pred = {"instance_id": instance["instance_id"], "model_name_or_path": "m139", "model_patch": patch}
    report = get_eval_report(spec, pred, str(log), True)[instance["instance_id"]]
    return {"applied": "PATCH_APPLIED" in out, "resolved": bool(report.get("resolved")), "report": report}


def main() -> int:
    preds = json.loads(Path(sys.argv[1]).read_text())
    out = Path(sys.argv[2])
    done = {json.loads(line)["id"] for line in out.read_text().splitlines()} if out.exists() else set()
    data = {r["instance_id"]: r for r in load_dataset("SWE-bench/SWE-bench_Verified", split="test")}
    for iid, pred in preds.items():
        if iid in done:
            continue
        patch = pred.get("model_patch") or ""
        with tempfile.TemporaryDirectory(dir="/var/tmp/claude/139") as tmp:
            res = grade(data[iid], patch, Path(tmp)) if patch.strip() else {"applied": False, "resolved": False}
        with out.open("a") as f:
            f.write(json.dumps({"id": iid, "empty_patch": not patch.strip(), **res}, default=str) + "\n")
        print(iid, "resolved" if res["resolved"] else "not resolved", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
