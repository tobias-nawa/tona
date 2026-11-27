"""scripts/ab_load_linux.py on a fake /proc and /sys tree."""

import os

import ab_load
import ab_load_linux as L
import pytest

UID = os.getuid()


def stat_line(pid, comm, ppid, utime, stime):
    # fields after comm: state ppid pgrp session tty tpgid flags minflt cminflt majflt cmajflt utime stime
    return f"{pid} ({comm}) S {ppid} 1 1 0 -1 0 0 0 0 0 {utime} {stime} 0 0 20 0 1\n"


def add_proc(root, pid, comm, ppid, **opts):
    """opts: ticks (utime, stime), uid, exe (None: unreadable)"""
    utime, stime = opts.get("ticks", (0, 0))
    uid, exe = opts.get("uid", UID), opts.get("exe", "/usr/bin/x")
    d = root / "proc" / str(pid)
    d.mkdir(parents=True)
    (d / "stat").write_text(stat_line(pid, comm, ppid, utime, stime))
    (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    if exe:
        (d / "exe").symlink_to(exe)
    return d


def add_drm_fd(d, fd, client, gfx, compute=0):
    (d / "fd").mkdir(exist_ok=True)
    (d / "fdinfo").mkdir(exist_ok=True)
    (d / "fd" / str(fd)).symlink_to("/dev/dri/renderD128")
    (d / "fdinfo" / str(fd)).write_text(
        f"pos:\t0\ndrm-driver:\tamdgpu\ndrm-client-id:\t{client}\ndrm-engine-gfx:\t{gfx} ns\n"
        f"drm-engine-compute:\t{compute} ns\ndrm-engine-dec:\t999 ns\n"
    )


@pytest.fixture
def tree(tmp_path, monkeypatch):
    (tmp_path / "proc").mkdir()
    monkeypatch.setattr(L, "PROC", tmp_path / "proc")
    monkeypatch.setattr(L, "SYS", tmp_path / "sys")
    monkeypatch.setattr(ab_load.sys, "platform", "linux")
    return tmp_path


def test_stat_parsing_survives_spaces_and_parens_in_comm():
    tick = os.sysconf("SC_CLK_TCK")
    line = stat_line(7, "evil) S 99 (name", 5, 3 * tick, 1 * tick)
    assert L.parse_stat(line, tick) == (5, 4.0)
    assert L.parse_stat("garbage", tick) is None


def test_processes_group_by_uid_and_unreadable_executable(tree):
    add_proc(tree, 10, "py thon", 1, ticks=(100, 100), exe="/usr/bin/python3")
    add_proc(tree, 11, "server)", 10, exe="/work/tona-server")
    add_proc(tree, 20, "rootd", 1, uid=UID + 1, exe="/usr/bin/rootd")
    add_proc(tree, 21, "kthread", 2, exe=None)  # no readable exe: kernel thread
    add_proc(tree, 22, "helper", 1, exe="/usr/libexec/helper")
    procs = L.processes()
    tick = os.sysconf("SC_CLK_TCK")
    assert procs[10] == (1, 200 / tick, "/usr/bin/python3")
    own = ab_load.own_pids(procs, {10})
    assert own == {10, 11}
    groups = [ab_load.group_of(p, procs, own) for p in (11, 20, 21, 22, 10, 99)]
    assert groups == ["own", "system", "system", "system", "own", "other"]
    other = add_proc(tree, 30, "firefox", 1, exe="/usr/lib64/firefox/firefox")
    assert ab_load.group_of(30, L.processes(), own) == "other" and other.exists()


def test_gpu_counts_each_client_once_per_pid_and_skips_unreadable(tree):
    d = add_proc(tree, 10, "a", 1)
    add_drm_fd(d, 5, client=7, gfx=1000, compute=500)
    add_drm_fd(d, 6, client=7, gfx=1000, compute=500)  # dup of the same client
    add_drm_fd(d, 7, client=8, gfx=200)
    (d / "fd" / "9").symlink_to("/tmp/not-a-gpu")
    (d / "fdinfo" / "9").write_text("drm-client-id:\t99\ndrm-engine-gfx:\t5 ns\n")  # not a DRM node
    add_proc(tree, 11, "b", 1)  # no fd directory (other user)
    assert L.gpu_ns([10, 11, 12]) == {10: 1700}


def test_memory_reads_vmstat_and_meminfo(tree):
    page_kib = os.sysconf("SC_PAGE_SIZE") / 1024
    (tree / "proc" / "vmstat").write_text("pgpgin 2048\npgpgout 5\npswpin 256\npswpout 512\n")
    (tree / "proc" / "meminfo").write_text("MemTotal:  1000 kB\nMemFree: 1 kB\nMemAvailable:  250 kB\n")
    m = L.memory()
    assert m["pageins"] == 2.0
    assert m["swapins"] == 256 * page_kib / 1024 and m["compressions"] == 512 * page_kib / 1024
    assert m["level"] == 25 and m["backup"] is False


def test_vmstat_counters_become_interval_deltas(tree):
    (tree / "proc").joinpath("meminfo").write_text("MemTotal: 100 kB\nMemAvailable: 40 kB\n")
    (tree / "proc" / "vmstat").write_text("pgpgin 1024\npswpin 0\npswpout 0\n")
    add_proc(tree, 10, "a", 1)
    a = ab_load.sample({10})
    (tree / "proc" / "vmstat").write_text("pgpgin 4096\npswpin 0\npswpout 0\n")
    b = ab_load.sample({10})
    row = ab_load.interval(a, b)
    assert row["pageins"] == 3.0 and row["level"] == 40 and row["backup"] is False


def test_power_from_sys(tree):
    assert L.power() == {"warning": False, "cpu_limit": None, "battery": False, "mode": None}
    sysroot = tree / "sys"
    (sysroot / "class/power_supply/BAT0").mkdir(parents=True)
    (sysroot / "class/power_supply/BAT0/status").write_text("Discharging\n")
    (sysroot / "firmware/acpi").mkdir(parents=True)
    (sysroot / "firmware/acpi/platform_profile").write_text("performance\n")
    hw = sysroot / "class/drm/card1/device/hwmon/hwmon3"
    hw.mkdir(parents=True)
    (hw / "temp2_input").write_text("95000\n")
    (hw / "temp2_crit").write_text("100000\n")
    assert L.power() == {"warning": True, "cpu_limit": None, "battery": True, "mode": 2}
    (hw / "temp2_input").write_text("50000\n")
    assert L.power()["warning"] is False
