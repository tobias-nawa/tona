"""Run a foreign repository's tests in a rootless Podman container (the only place foreign code is executed).

Container: no network, read-only root, tmpfs for /tmp, only the one repository directory mounted (read-only),
all capabilities dropped, no new privileges, memory/CPU/process limits and a time limit per run. The dependencies
are baked into the image (Containerfile); the repository itself is never installed.
"""

import json
import subprocess
from pathlib import Path

import common

IMAGE = "localhost/tona-w76-py"
INNER = Path(__file__).with_name("sandbox_inner.py")


def podman_cmd(repo: common.Repo, timeout: int) -> list:
    return [
        "podman",
        "run",
        "--rm",
        "-i",
        "--network=none",
        "--read-only",
        "--tmpfs=/tmp:rw,size=512m,exec",
        "--user=65534:65534",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=2g",
        "--memory-swap=2g",
        "--cpus=2",
        "--pids-limit=256",
        f"--timeout={timeout + 20}",
        f"--volume={common.repo_dir(repo)}:/repo:ro,z",
        IMAGE,
        "python",
        "-I",
        "-c",
        INNER.read_text(),
    ]


def run_tests(repo: common.Repo, overrides: dict, timeout: int = 120, coverage: bool = False) -> dict:
    """Run the repo's tests on a copy with overrides ({path in the repo: new text}); returns the inner dict."""
    payload = {
        "pythonpath": repo.pythonpath,
        "test_args": list(repo.test_args),
        "overrides": overrides,
        "timeout": timeout,
        "coverage": coverage,
        "src_dir": repo.src_dir,
    }
    p = subprocess.run(
        podman_cmd(repo, timeout),
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        timeout=timeout + 60,
        check=False,
    )
    lines = p.stdout.strip().splitlines()
    if p.returncode != 0 or not lines:
        return {"rc": None, "timed_out": False, "tail": "sandbox error: " + p.stderr[-800:]}
    return json.loads(lines[-1])
