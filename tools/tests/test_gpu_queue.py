import os
import subprocess
import sys
import threading
import time

import gpu_queue
from conftest import ROOT


def test_tickets_are_numbered_in_order_and_dead_ones_dropped(tmp_path):
    q = tmp_path / "gpu.lock.queue"
    a = gpu_queue.draw(q, "a")
    b = gpu_queue.draw(q, "b")
    assert [t[2] for t in gpu_queue.tickets(q)] == [a, b]
    dead = q / "0000000000-999999999.ticket"  # a PID that does not exist
    dead.write_text("x\n0\n")
    assert dead not in [t[2] for t in gpu_queue.tickets(q)]
    assert not dead.exists()


def test_wait_turn_returns_only_for_the_oldest_live_ticket(tmp_path, monkeypatch):
    q = tmp_path / "gpu.lock.queue"
    first = gpu_queue.draw(q, "first")
    second = gpu_queue.draw(q, "second")
    monkeypatch.setattr(gpu_queue, "POLL_S", 0.01)
    done = threading.Event()
    t = threading.Thread(target=lambda: (gpu_queue.wait_turn(q, second), done.set()))
    t.start()
    time.sleep(0.1)
    assert not done.is_set()  # first is still in line
    first.unlink()
    t.join(2)
    assert done.is_set()


def test_run_holds_the_lock_and_sets_the_held_variable(tmp_path, monkeypatch):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("TONA_GPU_LOCK", str(lock))
    monkeypatch.delenv("TONA_GPU_LOCK_HELD", raising=False)
    out = tmp_path / "out"
    code = (
        "import fcntl, os, sys; "
        f"fd = open({str(lock)!r}, 'a'); "
        "busy = False\n"
        "try: fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)\n"
        "except BlockingIOError: busy = True\n"
        f"open({str(out)!r}, 'w').write(os.environ.get('TONA_GPU_LOCK_HELD', '') + ' ' + str(busy))"
    )
    assert gpu_queue.main(["--", sys.executable, "-c", code]) == 0
    assert out.read_text() == "1 True"  # the child sees the lock held (by its parent) and the variable set
    assert not list((tmp_path / "gpu.lock.queue").glob("*.ticket"))  # the ticket is gone afterwards


def test_an_outer_holder_runs_the_command_directly(tmp_path, monkeypatch):
    monkeypatch.setenv("TONA_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.setenv("TONA_GPU_LOCK_HELD", "1")
    assert gpu_queue.main(["--", sys.executable, "-c", "pass"]) == 0
    assert not (tmp_path / "gpu.lock.queue").exists()


def test_two_callers_run_in_arrival_order(tmp_path):
    lock = tmp_path / "gpu.lock"
    log = tmp_path / "log"
    env = {k: v for k, v in os.environ.items() if k != "TONA_GPU_LOCK_HELD"}
    env["TONA_GPU_LOCK"] = str(lock)
    script = str(ROOT / "scripts" / "gpu_queue.py")

    def start(name, sleep):
        cmd = f"import time; open({str(log)!r}, 'a').write('{name}'); time.sleep({sleep})"
        return subprocess.Popen([sys.executable, script, "--", sys.executable, "-c", cmd], env=env)

    first = start("A", 0.5)
    time.sleep(0.2)
    second = start("B", 0)
    time.sleep(0.1)
    third = start("C", 0)
    for p in (first, second, third):
        assert p.wait(10) == 0
    assert log.read_text() == "ABC"


def test_status_lists_the_line(tmp_path, monkeypatch, capsys):
    lock = tmp_path / "gpu.lock"
    monkeypatch.setenv("TONA_GPU_LOCK", str(lock))
    gpu_queue.draw(gpu_queue.queue_dir(lock), "build")
    assert gpu_queue.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "free" in out and "build" in out
    ticket = next(gpu_queue.queue_dir(lock).glob("*.ticket"))
    ticket.write_text("build\n0\n")  # drawn long ago
    gpu_queue.main(["--status"])
    assert "waiting 0 min" not in capsys.readouterr().out


def test_kinds_conflict_only_with_equal_kinds_or_exclusive():
    c = gpu_queue.conflicts
    assert not c("gpu", "cpu") and not c("cpu", "gpu")
    assert c("gpu", "gpu") and c("cpu", "cpu")
    assert all(c("exclusive", k) and c(k, "exclusive") for k in gpu_queue.KINDS)


def test_old_tickets_without_kind_count_as_exclusive(tmp_path):
    q = tmp_path / "gpu.lock.queue"
    t = gpu_queue.draw(q, "old")
    t.write_text("old\n0\n")
    assert gpu_queue.ticket_kind(t) == "exclusive"


def _start(script, env, log, name, kind, sleep):
    write = f"open({str(log)!r}, 'a').write"
    cmd = f"import time; {write}('{name}'); time.sleep({sleep}); {write}('{name.lower()}')"
    return subprocess.Popen([sys.executable, script, "--kind", kind, "--", sys.executable, "-c", cmd], env=env)


def test_a_gpu_job_and_a_cpu_job_overlap_but_exclusive_waits(tmp_path):
    log = tmp_path / "log"
    env = {k: v for k, v in os.environ.items() if k not in ("TONA_GPU_LOCK_HELD", "TONA_CPU_LOCK_HELD")}
    env["TONA_GPU_LOCK"] = str(tmp_path / "gpu.lock")
    script = str(ROOT / "scripts" / "gpu_queue.py")
    g = _start(script, env, log, "G", "gpu", 0.6)
    time.sleep(0.2)
    c = _start(script, env, log, "C", "cpu", 0.6)
    time.sleep(0.2)
    x = _start(script, env, log, "X", "exclusive", 0)
    for p in (g, c, x):
        assert p.wait(10) == 0
    out = log.read_text()
    assert out.index("C") < out.index("g")  # the cpu job started while the gpu job ran
    assert out.index("X") > max(out.index("g"), out.index("c"))  # exclusive waited for both


def test_a_cpu_job_may_not_start_gpu_work(tmp_path, monkeypatch):
    monkeypatch.setenv("TONA_GPU_LOCK", str(tmp_path / "gpu.lock"))
    monkeypatch.delenv("TONA_GPU_LOCK_HELD", raising=False)
    monkeypatch.setenv("TONA_CPU_LOCK_HELD", "1")
    assert gpu_queue.main(["--kind", "gpu", "--", sys.executable, "-c", "pass"]) == 2
    assert gpu_queue.main(["--kind", "cpu", "--", sys.executable, "-c", "pass"]) == 0


def test_a_worktree_queues_at_the_lock_of_the_main_checkout(tmp_path):
    main = tmp_path / "main"
    main.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main", str(main)], check=True)
    subprocess.run(
        ["git", "-C", str(main), "commit", "-q", "--allow-empty", "-m", "x"],
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    )
    wt = tmp_path / "wt"
    subprocess.run(["git", "-C", str(main), "worktree", "add", "-q", str(wt)], check=True)
    assert gpu_queue.shared_root(wt).resolve() == main.resolve()
    assert gpu_queue.shared_root(main).resolve() == main.resolve()
