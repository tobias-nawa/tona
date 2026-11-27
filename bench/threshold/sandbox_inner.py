"""Runs INSIDE the container (python -c, code and payload come from the host, see sandbox.py).

Reads one JSON payload from stdin: pythonpath, test_args, overrides {relative path: new content}, timeout, coverage,
src_dir. Copies the read-only repository to the tmpfs, applies the overrides, runs pytest, prints one JSON line.
"""

import json
import os
import shutil
import subprocess
import sys

req = json.load(sys.stdin)
work = "/tmp/w"
shutil.copytree("/repo", work, ignore=shutil.ignore_patterns(".git"))
for rel, content in req["overrides"].items():
    with open(os.path.join(work, rel), "w") as f:
        f.write(content)
env = dict(os.environ, PYTHONPATH=os.path.join(work, req["pythonpath"]), PYTHONDONTWRITEBYTECODE="1", HOME="/tmp")
pytest = ["-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=", *req["test_args"]]
if req["coverage"]:
    cmd = [sys.executable, "-m", "coverage", "run", "--data-file=/tmp/cov.data", "--source=" + req["src_dir"], *pytest]
else:
    cmd = [sys.executable, *pytest, "-x"]
out = {"rc": None, "timed_out": False, "tail": ""}
try:
    p = subprocess.run(cmd, cwd=work, env=env, capture_output=True, text=True, timeout=req["timeout"], check=False)
    out["rc"], out["tail"] = p.returncode, (p.stdout + p.stderr)[-1500:]
except subprocess.TimeoutExpired:
    out["timed_out"] = True
if req["coverage"] and out["rc"] == 0:
    subprocess.run(
        [sys.executable, "-m", "coverage", "json", "--data-file=/tmp/cov.data", "-o", "/tmp/cov.json"],
        cwd=work,
        env=env,
        capture_output=True,
        check=False,
    )
    with open("/tmp/cov.json") as f:
        files = json.load(f)["files"]
    out["coverage"] = {k: [v["executed_lines"], v["missing_lines"]] for k, v in files.items()}
print(json.dumps(out))
