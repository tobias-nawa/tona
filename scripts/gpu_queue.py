#!/usr/bin/env python3
"""First come, first served in front of the GPU lock, with three kinds of jobs.

    scripts/gpu_queue.py [--kind K] [--label NAME] -- COMMAND [ARGS...]   wait for your turn, then run COMMAND
    scripts/gpu_queue.py --status                                          who holds the locks, who waits

Kinds (--kind, default exclusive):
    exclusive  the whole machine: speed measurements, anything whose numbers a second job would distort
    gpu        the GPU only: quality runs, goldens, correctness checks, micro benchmarks without timing claims
    cpu        no GPU at all: builds, CPU oracle runs, replays, training on the CPU
A gpu job and a cpu job run side by side; an exclusive job waits for both and blocks both. Two jobs of the same kind
never overlap. The GPU side is the flock on TONA_GPU_LOCK (default work/gpu.lock), the CPU side a second flock on
<lock>.cpu; an exclusive job takes the CPU side first, so a cpu job that also needs the GPU cannot deadlock with it.

`flock` alone wakes a random waiter when the lock is released, so a five-minute build can wait behind an hour of
measurements that arrived later. Here every caller draws a ticket (a number under a short mutex), waits until no live
ticket is older than its own, and only then takes the flock on TONA_GPU_LOCK (default work/gpu.lock). The flock stays
the one lock every GPU run of this machine shares, so plain `flock work/gpu.lock ...` callers and scripts/gpu-lock.sh
keep working; they only skip the line. COMMAND runs with TONA_GPU_LOCK_HELD=1, so the scripts it starts do not take the
lock again (docs/contribute/testing.md). Tickets of dead processes are dropped, so a killed waiter never blocks the
line. A cpu job gets TONA_CPU_LOCK_HELD=1 instead and must not start GPU work; a gpu_queue call inside it is refused.

A gpu or exclusive job also waits, with its ticket, until nobody uses the desktop, and is stopped (exit 75) when
someone comes back (scripts/desktop_guard.py). It runs in its own process group, so stopping the queue
(SIGTERM, Ctrl-C) stops the whole job, servers included.
"""

import argparse
import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import desktop_guard

ROOT = Path(__file__).resolve().parent.parent
POLL_S = 0.5  # a waiter notices its turn within half a second; the cost is one directory listing
KINDS = ("exclusive", "gpu", "cpu")
STOPPED_FOR_DESKTOP = 75  # EX_TEMPFAIL: run it again once the desktop is free


def conflicts(a: str, b: str) -> bool:
    """Two kinds may not run at the same time: equal kinds, or either one exclusive."""
    return a == b or "exclusive" in (a, b)


def shared_root(root: Path = ROOT) -> Path:
    """The main checkout, also from a git worktree: every worktree must queue at the same lock. A worktree with its
    own work/ directory instead of the link had its own lock and ran GPU jobs beside the others (2026-11-17)."""
    try:
        common = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--path-format=absolute", "--git-common-dir"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return root
    return Path(common).parent if common.endswith("/.git") else root


def lock_path() -> Path:
    return Path(os.environ.get("TONA_GPU_LOCK", str(shared_root() / "work" / "gpu.lock")))


def queue_dir(lock: Path) -> Path:
    return lock.with_name(lock.name + ".queue")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def ticket_kind(path: Path) -> str:
    """Kind written in the ticket's third line; tickets of older callers have none and count as exclusive."""
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "exclusive"
    return lines[2] if len(lines) > 2 and lines[2] in KINDS else "exclusive"


def tickets(qdir: Path) -> list[tuple[int, int, Path]]:
    """Live tickets as (number, pid, path), oldest first; tickets of dead processes are removed on the way."""
    out = []
    for p in qdir.glob("*.ticket"):
        try:
            num, pid = (int(x) for x in p.stem.split("-"))
        except ValueError:
            continue
        if alive(pid):
            out.append((num, pid, p))
        else:
            p.unlink(missing_ok=True)
    return sorted(out)


def draw(qdir: Path, label: str, kind: str = "exclusive") -> Path:
    """Next ticket number under a short mutex, so two callers never get the same one."""
    qdir.mkdir(parents=True, exist_ok=True)
    with open(qdir / "counter", "a+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        num = int(f.read().strip() or 0) + 1
        f.seek(0)
        f.truncate()
        f.write(str(num))
    path = qdir / f"{num:010d}-{os.getpid()}.ticket"
    path.write_text(f"{label}\n{time.time():.0f}\n{kind}\n")
    return path


def wait_turn(qdir: Path, mine: Path) -> None:
    """Return once no older live ticket conflicts with this one (arrival order among conflicting kinds)."""
    kind = ticket_kind(mine)
    announced = False
    while True:
        ahead = [t for t in tickets(qdir) if t[2] < mine and conflicts(kind, ticket_kind(t[2]))]
        if not ahead:
            return
        if not announced:
            print(f"gpu queue: {len(ahead)} ahead of you ({kind})", file=sys.stderr, flush=True)
            announced = True
        time.sleep(POLL_S)


def cpu_lock_path(lock: Path) -> Path:
    return lock.with_name(lock.name + ".cpu")


def lock_files(lock: Path, kind: str) -> list[Path]:
    """Flocks a kind takes, in this order: the CPU side before the GPU side (no lock cycle with a plain GPU flock)."""
    return {"exclusive": [cpu_lock_path(lock), lock], "gpu": [lock], "cpu": [cpu_lock_path(lock)]}[kind]


def held_env(kind: str) -> dict[str, str]:
    env = dict(os.environ)
    if kind == "cpu":
        env["TONA_CPU_LOCK_HELD"] = "1"
    else:
        env["TONA_GPU_LOCK_HELD"] = "1"
    return env


def run(command: list[str], label: str, kind: str = "exclusive") -> int:
    lock = lock_path()
    lock.parent.mkdir(parents=True, exist_ok=True)
    qdir = queue_dir(lock)
    mine = draw(qdir, label, kind)
    try:
        wait_turn(qdir, mine)
        if kind != "cpu":
            desktop_guard.wait_until_away(label)
        with contextlib.ExitStack() as stack:
            fds = [stack.enter_context(open(path, "a")) for path in lock_files(lock, kind)]
            for fd in fds:
                fcntl.flock(fd, fcntl.LOCK_EX)  # a plain `flock` caller may still hold it; wait for that one
            # The command inherits the locked descriptors: if this wrapper dies, the locks live as long as the job
            return supervise(command, kind, tuple(f.fileno() for f in fds))
    finally:
        mine.unlink(missing_ok=True)


def supervise(command: list[str], kind: str, fds: tuple[int, ...]) -> int:
    """Run the job in its own process group; stop the group on SIGTERM, Ctrl-C or (GPU kinds) a returning user."""
    proc = subprocess.Popen(command, env=held_env(kind), pass_fds=fds, start_new_session=True)
    main = threading.current_thread() is threading.main_thread()  # signal handlers only exist there
    previous = signal.signal(signal.SIGTERM, interrupt) if main else None
    watch = desktop_guard.Watch(lambda: desktop_guard.stop_group(proc.pid))
    try:
        if kind == "cpu":
            return proc.wait()
        with watch:
            code = proc.wait()
        return STOPPED_FOR_DESKTOP if watch.fired else code
    except KeyboardInterrupt:
        desktop_guard.stop_group(proc.pid)
        return 130
    finally:
        if main:
            signal.signal(signal.SIGTERM, previous)


def interrupt(*_) -> None:
    raise KeyboardInterrupt


def holder(lock: Path) -> str:
    """PID that holds a flock on the lock file, from /proc/locks; empty when free or unknown."""
    try:
        inode = lock.stat().st_ino
        for line in Path("/proc/locks").read_text().splitlines():
            f = line.split()
            if len(f) > 5 and f[1] == "FLOCK" and f[5].split(":")[-1] == str(inode):
                return f[4]
    except OSError:
        pass
    return ""


def status() -> int:
    lock = lock_path()
    for name, path in (("gpu", lock), ("cpu", cpu_lock_path(lock))):
        pid = holder(path) if path.exists() else ""
        print(f"{name} lock {path}: " + (f"held by pid {pid}" if pid else "free"))
    now = time.time()
    for n, (num, tpid, p) in enumerate(tickets(queue_dir(lock)), 1):
        label, started = ([*p.read_text().splitlines(), "", ""])[:2]
        age = int(now - float(started or now))
        print(f"{n:3d}. ticket {num} pid {tpid} {ticket_kind(p):9s} waiting {age // 60} min  {label}")
    return 0


class GpuLock:
    """flock on the lock file for the whole measurement; released on exit and when the process ends."""

    def __init__(self, path: str, enabled: bool):
        self.path, self.enabled, self.fd, self.cpu_fd, self.ticket = path, enabled, None, None, None
        self.watch = None

    def __enter__(self):
        if self.enabled:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self.fd = os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o664)
            print(f"waiting for the GPU lock {self.path} ...", file=sys.stderr, flush=True)
            qdir = queue_dir(Path(self.path))  # first come, first served
            self.ticket = draw(qdir, "ab.py")  # a speed measurement: exclusive, the CPU side too
            wait_turn(qdir, self.ticket)
            desktop_guard.wait_until_away("ab.py")
            self.cpu_fd = os.open(cpu_lock_path(Path(self.path)), os.O_WRONLY | os.O_CREAT, 0o664)
            fcntl.flock(self.cpu_fd, fcntl.LOCK_EX)
            fcntl.flock(self.fd, fcntl.LOCK_EX)
            os.environ["TONA_GPU_LOCK_HELD"] = "1"  # the scripts that ab.py starts must not take the lock again
            # ab.py turns SIGTERM into a clean shutdown of its servers
            self.watch = desktop_guard.Watch(lambda: os.kill(os.getpid(), signal.SIGTERM)).__enter__()
        return self

    def __exit__(self, *exc):
        if self.watch is not None:
            self.watch.__exit__()
        if self.ticket is not None:
            self.ticket.unlink(missing_ok=True)
        if self.fd is not None:
            os.close(self.fd)
            os.environ.pop("TONA_GPU_LOCK_HELD", None)
        if self.cpu_fd is not None:
            os.close(self.cpu_fd)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--status", action="store_true", help="show the holder and the line")
    ap.add_argument("--label", default="", help="shown by --status (default: the command)")
    ap.add_argument("--kind", choices=KINDS, default="exclusive", help="exclusive (default), gpu or cpu")
    ap.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARGS...]")
    args = ap.parse_args(argv)
    if args.status:
        return status()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("no command (scripts/gpu_queue.py -- COMMAND ...)")
    if os.environ.get("TONA_GPU_LOCK_HELD"):
        return subprocess.call(command)  # an outer caller holds the lock already; locks must not nest
    if os.environ.get("TONA_CPU_LOCK_HELD"):
        if args.kind == "cpu":
            return subprocess.call(command)
        print("gpu_queue: refused, a cpu job must not start GPU work; queue it as gpu or exclusive", file=sys.stderr)
        return 2
    return run(command, args.label or " ".join(command)[:80], args.kind)


if __name__ == "__main__":
    sys.exit(main())
