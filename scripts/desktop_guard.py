"""Keep GPU jobs away from a desktop someone is using.

The display runs on the same GPU as Tona. On 2026-11-17 a queued measurement planned 13.6 of 16.3 GB VRAM while the
user worked; the compositor then found no memory for its command buffers and died ("Not enough memory for command
submission"). A measurement taken while someone works is disturbed anyway (AGENTS.md rule 2, quiet machine), so a GPU
job starts only while every connected monitor is powered off by the idle daemon, and it is stopped when one comes back.

The signal is the DPMS state in /sys/class/drm: it does not depend on the compositor, and the idle daemon of the
reference machine powers the monitors off after 15 idle minutes. A machine without connected monitors (headless, a
Mac, a container) counts as away. A monitor that is on while the graphical session is locked counts as away too: a
touched mouse wakes the monitors to the lock screen, and nobody works behind it (logind's LockedHint, set by the lock
screen of the reference desktop; 2026-11-18). TONA_DESKTOP_GUARD=off turns the guard off for a run the user explicitly
allows. The file ~/.config/tona/desktop-guard-off turns it off machine-wide while the user has handed the machine over
(every worktree and every waiting job sees it at its next poll); `rm` of the file turns it on again.
"""

import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

DRM = Path("/sys/class/drm")
POLL_S = 1.0  # the compositor needs its memory right after the wake; one second is the longest we let it wait
GRACE_S = 15.0  # ab.py turns SIGTERM into a clean shutdown of its servers; a server frees its VRAM in a few seconds


OFF_FILE = Path.home() / ".config" / "tona" / "desktop-guard-off"


def enabled() -> bool:
    return os.environ.get("TONA_DESKTOP_GUARD", "on") != "off" and not OFF_FILE.exists()


def display_on(drm: Path | None = None) -> bool:
    """True when a connected, enabled monitor has DPMS on: someone may be looking at the desktop."""
    for conn in (drm or DRM).glob("card*-*"):
        try:
            status = (conn / "status").read_text().strip()
            power = (conn / "dpms").read_text().strip()
            on = (conn / "enabled").read_text().strip()
        except OSError:
            continue
        if status == "connected" and on == "enabled" and power == "On":
            return True
    return False


def session_locked(run=subprocess.run) -> bool:
    """True when every active graphical session is locked. Without logind or without a graphical session: False, so
    the monitors alone decide (the guard errs towards waiting)."""
    try:
        listed = json.loads(run(["loginctl", "list-sessions", "--json=short"], **_QUIET).stdout or "[]")
        ids = [str(x["session"]) for x in listed if x.get("seat")]
        if not ids:
            return False
        props = ["-p", "Type", "-p", "Active", "-p", "LockedHint"]
        out = run(["loginctl", "show-session", *ids, *props], **_QUIET).stdout
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return False
    graphical = [b for b in _blocks(out) if b.get("Type") in ("wayland", "x11", "mir") and b.get("Active") == "yes"]
    return bool(graphical) and all(b.get("LockedHint") == "yes" for b in graphical)


_QUIET = {"capture_output": True, "text": True, "timeout": 5, "check": True}


def _blocks(text: str) -> list[dict[str, str]]:
    """loginctl show-session prints one block of key=value lines per session, separated by a blank line."""
    return [dict(line.split("=", 1) for line in block.splitlines() if "=" in line) for block in text.split("\n\n")]


def in_use(drm: Path | None = None) -> bool:
    """Someone may be working at the desktop: a monitor is on and the session is not locked."""
    return display_on(drm) and not session_locked()


def wait_until_away(label: str, drm: Path | None = None, poll_s: float = 10.0) -> None:
    """Block while the desktop is in use; the caller keeps its place in the queue meanwhile."""
    if not enabled():
        return
    announced = False
    while enabled() and in_use(drm):
        if not announced:
            print(
                f"desktop guard: a monitor is on and the session is unlocked, {label} waits until the monitors power "
                "off or the screen is locked (TONA_DESKTOP_GUARD=off skips this when the user allows it)",
                file=sys.stderr,
                flush=True,
            )
            announced = True
        time.sleep(poll_s)


class Watch:
    """Calls stop() once when a monitor comes on during the job; started and ended around the job."""

    def __init__(self, stop, drm: Path | None = None, poll_s: float = POLL_S):
        self.stop, self.drm, self.poll_s = stop, drm, poll_s
        self.done = threading.Event()
        self.fired = False
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def _loop(self) -> None:
        while not self.done.wait(self.poll_s):
            if enabled() and in_use(self.drm):
                self.fired = True
                print(
                    "desktop guard: someone is at the unlocked desktop, the GPU job is stopped",
                    file=sys.stderr,
                    flush=True,
                )
                self.stop()
                return

    def __enter__(self):
        if enabled():
            self.thread.start()
        return self

    def __exit__(self, *exc):
        self.done.set()


def stop_group(pgid: int, grace_s: float = GRACE_S) -> None:
    """SIGTERM to the job's process group, SIGKILL after the grace period."""
    with_signal(pgid, signal.SIGTERM)
    end = time.time() + grace_s
    while time.time() < end:
        if not group_alive(pgid):
            return
        time.sleep(0.2)
    with_signal(pgid, signal.SIGKILL)


def with_signal(pgid: int, sig: int) -> None:
    with contextlib.suppress(ProcessLookupError, PermissionError):
        os.killpg(pgid, sig)


def group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
