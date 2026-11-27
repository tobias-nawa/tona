"""scripts/gpu-lock.sh: a GPU script takes the shared lock itself, once, and says so when it has to wait."""

import fcntl
import shutil
import subprocess
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(not shutil.which("flock"), reason="flock")


def make_script(directory: Path, name: str, body: str) -> Path:
    shutil.copy(ROOT / "scripts/gpu-lock.sh", directory / "gpu-lock.sh")
    shutil.copy(ROOT / "scripts/gpu_queue.py", directory / "gpu_queue.py")  # the ticket line the helper starts
    shutil.copy(ROOT / "scripts/desktop_guard.py", directory / "desktop_guard.py")
    script = directory / name
    script.write_text(f'#!/usr/bin/env bash\nset -euo pipefail\nsource "$(dirname "$0")/gpu-lock.sh"\n{body}\n')
    script.chmod(0o755)
    return script


def start(script: Path, lock: Path, env_extra=None):
    env = {"PATH": "/usr/bin:/bin", "TONA_GPU_LOCK": str(lock), "TONA_DESKTOP_GUARD": "off", **(env_extra or {})}
    return subprocess.Popen([str(script)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def test_a_second_run_waits_for_the_holder_and_says_so(tmp_path):
    lock = tmp_path / "gpu.lock"
    script = make_script(tmp_path, "gpu.sh", 'echo "ran with $TONA_GPU_LOCK_HELD"')
    subprocess.Popen(["flock", str(lock), "sleep", "2"])  # another process, not an ancestor
    time.sleep(0.5)
    proc = start(script, lock)
    time.sleep(0.5)
    assert proc.poll() is None  # waits until the holder ends
    out, err = proc.communicate(timeout=10)
    assert out == "ran with 1\n"
    assert "waiting for the GPU lock" in err


def test_a_free_lock_is_taken_without_a_message(tmp_path):
    script = make_script(tmp_path, "gpu.sh", "echo ok")
    out, err = start(script, tmp_path / "gpu.lock").communicate(timeout=10)
    assert out == "ok\n" and err == ""


def test_a_script_started_by_a_holder_does_not_lock_again(tmp_path):
    make_script(tmp_path, "inner.sh", "echo inner")
    outer = make_script(tmp_path, "outer.sh", '"$(dirname "$0")/inner.sh"')
    out, err = start(outer, tmp_path / "gpu.lock").communicate(timeout=10)  # a nested lock would deadlock
    assert out == "inner\n" and err == ""


def test_help_starts_nothing_and_takes_no_lock(tmp_path):
    lock = tmp_path / "gpu.lock"
    script = make_script(tmp_path, "gpu.sh", 'echo "args: $*"')
    with open(lock, "a") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)
        proc = subprocess.run([str(script), "--help"], env={"PATH": "/usr/bin:/bin", "TONA_GPU_LOCK": str(lock)},
                              capture_output=True, text=True, timeout=10, check=False)  # fmt: skip
    assert proc.stdout == "args: --help\n"


def test_an_outer_flock_without_the_variable_does_not_deadlock(tmp_path):
    lock = tmp_path / "gpu.lock"
    script = make_script(tmp_path, "gpu.sh", 'echo "ran with $TONA_GPU_LOCK_HELD"')
    env = {"PATH": "/usr/bin:/bin", "TONA_GPU_LOCK": str(lock)}
    # timeout: a deadlock fails the test instead of hanging it
    done = subprocess.run(["timeout", "10", "flock", str(lock), str(script)], env=env, capture_output=True,
                          text=True, check=False)  # fmt: skip
    assert done.returncode == 0, done.stderr
    assert done.stdout == "ran with 1\n" and "GPU lock held by parent" in done.stderr


def test_an_outer_flock_through_a_shell_chain_does_not_deadlock(tmp_path):
    lock = tmp_path / "gpu.lock"
    script = make_script(tmp_path, "gpu.sh", "echo ok")
    env = {"PATH": "/usr/bin:/bin", "TONA_GPU_LOCK": str(lock)}
    cmd = ["timeout", "10", "flock", str(lock), "bash", "-c", f"{script}; true"]  # the script is not a direct child
    done = subprocess.run(cmd, env=env, capture_output=True, text=True, check=False)
    assert done.returncode == 0 and done.stdout == "ok\n"


def test_another_process_holding_the_lock_is_still_waited_for(tmp_path):
    lock = tmp_path / "gpu.lock"
    script = make_script(tmp_path, "gpu.sh", "echo ok")
    holder = subprocess.Popen(["flock", str(lock), "sleep", "2"])
    time.sleep(0.5)
    try:
        proc = start(script, lock)
        time.sleep(0.5)
        assert proc.poll() is None  # not an ancestor: waits
    finally:
        holder.wait()
    out, err = proc.communicate(timeout=10)
    assert out == "ok\n" and "waiting for the GPU lock" in err
