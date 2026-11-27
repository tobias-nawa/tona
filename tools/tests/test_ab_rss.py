"""scripts/ab.py on Linux (systemd scope) and macOS (resident-size guard), both branches on any host."""

import shutil
import subprocess
import sys
import time

import ab_rss
import ab_server
import pytest
from ab_server import Arm, ServerConfig


def sleeper() -> subprocess.Popen:
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])


def test_parse_limit_reads_systemd_sizes():
    assert ab_rss.parse_limit("54G") == 54 << 30
    assert ab_rss.parse_limit("512m") == 512 << 20
    assert ab_rss.parse_limit("4096") == 4096
    with pytest.raises(ValueError):
        ab_rss.parse_limit("lots")


@pytest.mark.skipif(shutil.which("ps") is None, reason="needs ps (procps-ng; absent from minimal container images)")
def test_rss_bytes_of_a_live_and_a_finished_process():
    proc = sleeper()
    try:
        assert ab_rss.rss_bytes(proc.pid) > 0
    finally:
        proc.kill()
        proc.wait()
    assert ab_rss.rss_bytes(proc.pid) is None


def test_guard_ends_the_process_over_the_limit(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_rss, "rss_bytes", lambda pid: 2000)
    proc, log = sleeper(), tmp_path / "server.log"
    guard = ab_rss.RssGuard(proc, 1000, log, period=0.05, grace=2).start()
    proc.wait(10)
    guard.stop()
    assert guard.tripped == 2000
    assert f"ending PID {proc.pid}" in log.read_text()


def test_guard_kills_a_process_that_ignores_sigterm(monkeypatch):
    monkeypatch.setattr(ab_rss, "rss_bytes", lambda pid: 2)
    proc = subprocess.Popen(
        [sys.executable, "-c", "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"]
    )
    time.sleep(0.5)
    guard = ab_rss.RssGuard(proc, 1, period=0.05, grace=0.3).start()
    proc.wait(10)
    guard.stop()
    assert proc.returncode is not None and guard.tripped == 2


def test_guard_leaves_a_process_under_the_limit_alone():
    proc = sleeper()
    guard = ab_rss.RssGuard(proc, 1 << 40, period=0.05).start()
    time.sleep(0.3)
    assert proc.poll() is None and guard.tripped is None
    ab_server.stop_server(proc, grace=5)
    guard.stop()
    assert proc.poll() is not None


def test_server_command_linux_runs_in_a_scope(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_server.sys, "platform", "linux")
    cfg = ServerConfig(tmp_path, "m.gguf", "d.gguf", 140000, port=8123, memory_max="54G")
    cmd = ab_server.server_command(cfg, Arm("A", "merged"))
    assert cmd == [
        "systemd-run", "--user", "--scope", "-q", "-p", "MemoryMax=54G", "-p", "MemorySwapMax=0",
        str(tmp_path / "work/build/merged/tona-server"), "m.gguf", "--mtp", "d.gguf", "--port", "8123",
        "--ctx", "140000",
    ]  # fmt: skip


def test_server_command_darwin_starts_the_binary_itself(tmp_path, monkeypatch):
    monkeypatch.setattr(ab_server.sys, "platform", "darwin")
    cfg = ServerConfig(tmp_path, "m.gguf", "d.gguf", 16384, port=8123)
    cmd = ab_server.server_command(cfg, Arm("A", "merged"))
    assert cmd[0] == str(tmp_path / "work/build/merged/tona-server") and "systemd-run" not in cmd


def fake_build(tmp_path):
    build = tmp_path / "work/build/x"
    build.mkdir(parents=True)
    binary = build / "tona-server"
    binary.write_text("#!/bin/sh\nexec sleep 60\n")
    binary.chmod(0o755)
    return ServerConfig(tmp_path, "m.gguf", "d.gguf", 1024, memory_max="1T")


@pytest.mark.parametrize("platform,guarded", [("darwin", True), ("linux", False)])
def test_start_server_guards_only_on_darwin(tmp_path, monkeypatch, platform, guarded):
    monkeypatch.setattr(ab_server.sys, "platform", platform)
    started = []
    real_popen = subprocess.Popen

    def popen(cmd, *args, **kw):  # the Linux branch would need systemd: run the binary alone, keep the command
        server = [c for c in cmd if str(c).endswith("tona-server")]
        if not server:  # the guard's ps
            return real_popen(cmd, *args, **kw)
        started.append(cmd)
        return real_popen([*server, "x"], *args, **kw)

    monkeypatch.setattr(ab_server.subprocess, "Popen", popen)
    proc = ab_server.start_server(fake_build(tmp_path), Arm("A", "x"), tmp_path / "log")
    try:
        assert (getattr(proc, "rss_guard", None) is not None) == guarded
        assert (started[0][0] == "systemd-run") == (not guarded)
    finally:
        ab_server.stop_server(proc, grace=5)
    assert proc.poll() is not None


def test_wait_ready_names_the_guard(monkeypatch):
    monkeypatch.setattr(ab_rss, "rss_bytes", lambda pid: 3 << 20)
    proc = sleeper()
    proc.rss_guard = ab_rss.RssGuard(proc, 1, period=0.05, grace=2).start()
    with pytest.raises(RuntimeError, match="resident size 3 MiB over the limit"):
        ab_server.wait_ready(proc, 1, timeout=20)
    ab_server.stop_server(proc)
