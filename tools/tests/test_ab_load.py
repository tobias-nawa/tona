"""scripts/ab_load.py: the load beside each server start, split into ours, system and other, without names."""

import ab_load
import ab_report
import pytest


def test_ps_times_parse_with_and_without_hours_and_days():
    assert ab_load.cpu_seconds("0:02.26") == 2.26
    assert ab_load.cpu_seconds("313:54.16") == 313 * 60 + 54.16
    assert ab_load.cpu_seconds("01:00:01.00") == 3601.0
    assert ab_load.cpu_seconds("2-00:00:00.00") == 2 * 86400


PROCS = {
    10: (1, 0.0, "/usr/bin/python3"),  # the measuring process
    11: (10, 0.0, "/work/build/mtl/tona-server"),
    12: (11, 0.0, "/bin/sh"),  # a grandchild is still ours
    20: (1, 0.0, "/System/Library/CoreServices/backupd"),
    30: (1, 0.0, "/Applications/Some.app/Contents/MacOS/Some"),
}


def test_children_of_the_measuring_process_are_ours_and_paths_decide_the_rest(monkeypatch):
    monkeypatch.setattr(ab_load.sys, "platform", "darwin")  # the system paths are per platform; the table is macOS's
    own = ab_load.own_pids(PROCS, {10})
    assert own == {10, 11, 12}
    assert [ab_load.group_of(p, PROCS, own) for p in (11, 20, 30, 99)] == ["own", "system", "other", "other"]


def test_deltas_count_only_processes_seen_twice_and_never_go_negative(monkeypatch):
    monkeypatch.setattr(ab_load.sys, "platform", "darwin")
    before = {11: 5.0, 20: 1.0, 30: 3.0}
    after = {11: 7.0, 20: 1.5, 30: 2.0, 40: 9.0}  # 30 restarted (counter fell), 40 is new
    got = ab_load.deltas(before, after, PROCS, {10, 11, 12})
    assert got == {"own": 2.0, "system": 0.5, "other": 0.0}


def row(seconds, cpu_other, backup=False, level=80):
    r = {f"cpu_{g}": 0.0 for g in ab_load.GROUPS} | {f"gpu_{g}": 0.0 for g in ab_load.GROUPS}
    r |= {"cpu_other": cpu_other, "pageins": 10.0, "swapins": 0.0, "compressions": 1.0}
    return r | {"level": level, "backup": backup, "seconds": seconds}


def test_power_states_are_shares_of_the_time_and_the_lowest_limit():
    hot = row(1.0, 0.0) | {"power_warning": True, "power_cpu_limit": 70, "power_battery": False, "power_mode": 2}
    cool = row(3.0, 0.0) | {"power_warning": False, "power_cpu_limit": None, "power_battery": False, "power_mode": 0}
    s = ab_load.summary([hot, cool])
    assert s["warning_share"] == 0.25 and s["cpu_limit_min"] == 70 and s["battery_share"] == 0.0
    assert s["power_mode"] == "0/2"
    assert ab_load.summary([row(1.0, 0.0)])["power_mode"] is None  # a platform without power data


def test_summary_weights_by_time_and_sums_the_traffic():
    s = ab_load.summary([row(2.0, 10.0), row(6.0, 50.0, backup=True, level=70)])
    assert s["cpu_other"] == 40.0
    assert s["pageins"] == 20.0 and s["level_min"] == 70 and s["backup_share"] == 0.75 and s["seconds"] == 8
    assert ab_load.summary([]) == {}


def test_the_report_lists_every_start_without_program_names():
    start = {"start": 3, "arm": "B"} | ab_load.summary([row(2.0, 10.0)])
    rows = ab_load.load_rows([start])
    assert rows[2].startswith("| 3 | B | ") and len(rows) == 3
    data = {"config": {"A": {"build": "a", "env": {}}, "B": {"build": "b", "env": {}}, "deterministic": False}}
    text = ab_report.markdown(data | {"rounds": [], "stop_reason": "", "loads": [start]}, ab_report.Plan())
    assert text == "No round completed.\n"  # nothing to put the loads beside


def test_on_by_default_only_under_macos(monkeypatch):
    monkeypatch.setattr(ab_load.sys, "platform", "linux")
    assert not ab_load.enabled("auto") and not ab_load.enabled("off")  # Linux follows after an A/A run
    assert ab_load.enabled("on")
    monkeypatch.setattr(ab_load.sys, "platform", "plan9")
    with pytest.raises(SystemExit):  # no snapshot for this platform
        ab_load.enabled("on")
    monkeypatch.setattr(ab_load.sys, "platform", "darwin")
    assert ab_load.enabled("auto") and not ab_load.enabled("off")
