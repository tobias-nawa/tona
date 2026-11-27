import subprocess
import sys
import threading
import time
from pathlib import Path

import desktop_guard
import gpu_queue
import pytest


@pytest.fixture(autouse=True)
def _unlocked(monkeypatch, tmp_path):
    """The real session of the test machine may be locked or its guard switched off; every test decides that itself."""
    monkeypatch.setattr(desktop_guard, "session_locked", lambda: False)
    monkeypatch.setattr(desktop_guard, "OFF_FILE", tmp_path / "no-off-file")


def monitor(drm, name, status="connected", dpms="On", enabled="enabled"):
    d = drm / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "status").write_text(status + "\n")
    (d / "dpms").write_text(dpms + "\n")
    (d / "enabled").write_text(enabled + "\n")
    return d


def test_only_a_connected_enabled_monitor_with_dpms_on_counts(tmp_path):
    monitor(tmp_path, "card1-DP-1", dpms="Off")
    monitor(tmp_path, "card1-HDMI-A-1", status="disconnected", dpms="Off", enabled="disabled")
    monitor(tmp_path, "card1-Writeback-1", status="unknown", enabled="disabled")
    assert not desktop_guard.display_on(tmp_path)
    monitor(tmp_path, "card1-DP-2")
    assert desktop_guard.display_on(tmp_path)


def test_no_drm_at_all_counts_as_away(tmp_path):
    assert not desktop_guard.display_on(tmp_path / "missing")


def test_wait_until_away_blocks_while_a_monitor_is_on(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("TONA_DESKTOP_GUARD", "on")
    d = monitor(tmp_path, "card1-DP-1")
    done = threading.Event()
    t = threading.Thread(target=lambda: (desktop_guard.wait_until_away("job", tmp_path, 0.01), done.set()))
    t.start()
    time.sleep(0.1)
    assert not done.is_set()
    (d / "dpms").write_text("Off\n")
    t.join(2)
    assert done.is_set()
    assert "monitor is on" in capsys.readouterr().err


def test_the_switch_turns_the_guard_off(tmp_path, monkeypatch):
    monitor(tmp_path, "card1-DP-1")
    desktop_guard.wait_until_away("job", tmp_path, 0.01)  # conftest sets TONA_DESKTOP_GUARD=off


def test_a_returning_user_stops_the_job_group_and_the_queue_says_75(tmp_path, monkeypatch):
    monkeypatch.setenv("TONA_DESKTOP_GUARD", "on")
    monkeypatch.setenv("TONA_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(desktop_guard, "DRM", tmp_path / "drm")
    d = monitor(tmp_path / "drm", "card1-DP-1", dpms="Off")
    child = tmp_path / "child.pid"
    script = f"sleep 30 & echo $! > {child}; wait"
    real_watch = desktop_guard.Watch
    monkeypatch.setattr(desktop_guard, "Watch", lambda stop: real_watch(stop, drm=tmp_path / "drm", poll_s=0.05))
    threading.Timer(0.5, lambda: (d / "dpms").write_text("On\n")).start()
    start = time.time()
    code = gpu_queue.run(["bash", "-c", script], "test", "gpu")
    assert code == gpu_queue.STOPPED_FOR_DESKTOP
    assert time.time() - start < 10
    assert not alive(int(child.read_text()))  # the grandchild is gone too


def alive(pid: int, wait_s: float = 2.0) -> bool:
    """A killed process that nobody reaps stays a zombie; in a CI container PID 1 often does not reap, and
    `kill -0` still succeeds on a zombie. Gone or zombie counts as dead; the wait covers the signal's delivery."""
    stat = Path(f"/proc/{pid}/stat")
    deadline = time.time() + wait_s
    while time.time() < deadline:
        try:
            if stat.read_text().rsplit(")", 1)[1].split()[0] == "Z":
                return False
        except (FileNotFoundError, ProcessLookupError):
            return False
        time.sleep(0.05)
    return True


def test_a_cpu_job_ignores_the_desktop(tmp_path, monkeypatch):
    monkeypatch.setenv("TONA_DESKTOP_GUARD", "on")
    monkeypatch.setenv("TONA_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setattr(desktop_guard, "DRM", tmp_path / "drm")
    monitor(tmp_path / "drm", "card1-DP-1")
    assert gpu_queue.run([sys.executable, "-c", "pass"], "build", "cpu") == 0


class FakeRun:
    def __init__(self, listed, shown):
        self.listed, self.shown = listed, shown

    def __call__(self, cmd, **_):
        out = self.listed if cmd[1] == "list-sessions" else self.shown
        return subprocess.CompletedProcess(cmd, 0, stdout=out)


SESSIONS = '[{"session":"1","seat":null},{"session":"6","seat":"seat0"}]'
real_session_locked = desktop_guard.session_locked


def test_a_locked_graphical_session_counts_as_locked():
    shown = "Type=wayland\nActive=yes\nLockedHint=yes\n"
    assert real_session_locked(FakeRun(SESSIONS, shown))
    assert not real_session_locked(FakeRun(SESSIONS, shown.replace("LockedHint=yes", "LockedHint=no")))


def test_no_graphical_session_or_no_logind_is_not_locked():
    assert not real_session_locked(FakeRun('[{"session":"1","seat":null}]', ""))
    assert not real_session_locked(FakeRun(SESSIONS, "Type=tty\nActive=yes\nLockedHint=yes\n"))

    def missing(cmd, **_):
        raise FileNotFoundError(cmd[0])

    assert not real_session_locked(missing)


def test_one_unlocked_graphical_session_is_enough_to_count_as_in_use():
    shown = "Type=wayland\nActive=yes\nLockedHint=yes\n\nType=x11\nActive=yes\nLockedHint=no\n"
    assert not real_session_locked(FakeRun('[{"session":"6","seat":"seat0"},{"session":"7","seat":"seat1"}]', shown))


def test_a_monitor_woken_to_the_lock_screen_does_not_stop_the_job(tmp_path, monkeypatch):
    monitor(tmp_path, "card1-DP-1")
    assert desktop_guard.in_use(tmp_path)
    monkeypatch.setattr(desktop_guard, "session_locked", lambda: True)
    assert not desktop_guard.in_use(tmp_path)


def test_off_file_turns_the_guard_off(tmp_path, monkeypatch):
    monkeypatch.setenv("TONA_DESKTOP_GUARD", "on")
    off = tmp_path / "desktop-guard-off"
    monkeypatch.setattr(desktop_guard, "OFF_FILE", off)
    assert desktop_guard.enabled()
    off.touch()
    assert not desktop_guard.enabled()
