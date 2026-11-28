import quiet_check as qc

QUIET = {
    "taken": "2026-11-11 10:00:00",
    "threads": 12,
    "cpu_busy_percent": 0.5,
    "busy_processes": {"firefox": 4.0},
    "loadavg1": 0.2,
    "gpus": [{"busy": 1, "vram_gib": 16.0, "used_gib": 0.5}],
    "ram_total_gib": 64.0,
    "ram_available_gib": 50.0,
    "swap_pages_moved": 0,
    "on_battery": False,
}


def sig(**kw):
    return {**QUIET, **kw}


def test_quiet_machine_is_ok_and_asks_nothing():
    r = qc.evaluate(QUIET)
    assert r["state"] == qc.OK and r["findings"] == []
    assert "quiet enough" in qc.request(r)


def test_levels_by_threshold():
    assert qc.level(4, qc.CPU_BUSY_PERCENT) == qc.OK
    assert qc.level(5, qc.CPU_BUSY_PERCENT) == qc.WARN
    assert qc.level(20, qc.CPU_BUSY_PERCENT) == qc.STOP
    assert qc.level(15, qc.RAM_AVAILABLE_PERCENT, below=True) == qc.WARN
    assert qc.level(9, qc.RAM_AVAILABLE_PERCENT, below=True) == qc.STOP
    assert qc.level(None, qc.CPU_BUSY_PERCENT) == qc.OK


def test_busy_cpu_is_a_warning_with_numbers():
    r = qc.evaluate(sig(cpu_busy_percent=8.0, busy_processes={"steam": 60.0, "firefox": 30.0}))
    assert r["state"] == qc.WARN
    text = qc.request(r, {"CPU": "steam (60.0 % of a core)"})
    assert "50.0 GiB RAM" in text and "15.5 GiB VRAM" in text and "steam" in text


def test_load_average_only_warns():
    r = qc.evaluate(sig(loadavg1=30.0))
    assert r["state"] == qc.WARN


def test_battery_stops_and_request_merges_closes():
    r = qc.evaluate(
        sig(on_battery=True, gpus=[{"busy": 50, "vram_gib": 16.0, "used_gib": 6.0}], busy_processes={"steam": 9.0})
    )
    assert r["state"] == qc.STOP
    text = qc.request(r)
    assert "plug in the charger" in text
    assert "closing browsers" in text and "You decide" in text


def test_system_tools_are_never_named(monkeypatch):
    monkeypatch.setattr(qc, "ram_by_program", lambda: {"clamd": 4.0, "browser": 3.0})
    monkeypatch.setattr(qc, "vram_by_program", lambda: {})
    users = qc.top_users(sig(busy_processes={"baloo_file": 50.0, "steam": 9.0}))
    assert "browser" in users["RAM"] and "steam" in users["CPU"]
    assert "clamd" not in str(users) and "baloo_file" not in str(users)


def test_swap_warns():
    r = qc.evaluate(sig(swap_pages_moved=40))
    assert [f["level"] for f in r["findings"]] == [qc.WARN]


def test_compare_flags_new_load():
    before = qc.evaluate(QUIET)
    after = qc.evaluate(sig(on_battery=True))
    cmp = qc.compare(before, after)
    assert cmp["changed"] and cmp["new_findings"]
    assert not qc.compare(before, before)["changed"]
    assert "DISTURBED" in qc.render(after, cmp)


def test_render_marks_state():
    assert "- Quiet check (2026-11-11 10:00:00): ok" in qc.render(qc.evaluate(QUIET))


def test_saved_output_has_no_program_names(capsys):
    r = qc.evaluate(sig(cpu_busy_percent=30.0, busy_processes={"secretgame": 90.0, "baloo_file": 9.0}))
    assert "secretgame" in qc.request(r, qc.top_users(r["signals"]))  # spoken to the user only
    shown = qc.public(r)
    text = qc.render(shown) + str(shown)
    assert "secretgame" not in text and "baloo_file" not in text
    assert "busy_processes" not in shown["signals"]


def test_request_leaves_the_decision_to_the_user():
    text = qc.request(qc.evaluate(sig(cpu_busy_percent=8.0)), {"RAM": "browser (3.0 GiB)"})
    assert "browser (3.0 GiB)" in text and "You decide" in text and "system tools can keep running" in text
    assert "'measure'" in text


def test_quiet_request_says_so():
    assert "quiet enough" in qc.request(qc.evaluate(QUIET))


def fake_proc(tmp_path, pid, uid, cgroup, comm):
    d = tmp_path / str(pid)
    d.mkdir()
    (d / "status").write_text(f"Name:\t{comm}\nUid:\t{uid}\t{uid}\t{uid}\t{uid}\n")
    (d / "cgroup").write_text(f"0::{cgroup}\n")
    return d


def test_only_own_non_service_processes_are_proposed(tmp_path):
    me = 1000
    app = fake_proc(
        tmp_path, 1, me, "/user.slice/user-1000.slice/user@1000.service/app.slice/app-gnome-steam-1.scope", "steam"
    )
    session = fake_proc(tmp_path, 2, me, "/user.slice/user-1000.slice/session-2.scope", "browser")
    unit = fake_proc(
        tmp_path, 3, me, "/user.slice/user-1000.slice/user@1000.service/app.slice/syncthing.service", "syncthing"
    )
    system = fake_proc(tmp_path, 4, me, "/system.slice/clamd.service", "scanner")
    other = fake_proc(tmp_path, 5, 0, "/user.slice/user-0.slice/session-1.scope", "rootprog")
    assert qc.is_user_program_linux(app, me) and qc.is_user_program_linux(session, me)
    assert not qc.is_user_program_linux(unit, me)  # a user service with its own unit
    assert not qc.is_user_program_linux(system, me)
    assert not qc.is_user_program_linux(other, me)  # another UID


def test_top_users_filters_to_own_programs(monkeypatch):
    monkeypatch.setattr(qc, "ram_by_program", lambda: {"steam": 4.0, "syncthing": 5.0, "browser": 3.0})
    monkeypatch.setattr(qc, "vram_by_program", lambda: {})
    users = qc.top_users(sig(busy_processes={}), mine={"steam", "browser"})
    assert "steam" in users["RAM"] and "syncthing" not in users["RAM"]


def test_apple_gpu_counts_the_second_read():
    reads = iter(['"Device Utilization %"=93', '"Device Utilization %"=0'])
    assert qc.apple_gpu_busy(read=lambda: next(reads), wait=lambda s: None) == 0


def test_apple_gpu_absent_is_none_without_waiting():
    waited = []
    assert qc.apple_gpu_busy(read=lambda: "", wait=waited.append) is None and not waited


def test_windows_quiet_check_evaluate_and_measure(monkeypatch):
    import hw_win32

    monkeypatch.setattr(qc.sys, "platform", "win32")
    monkeypatch.setattr(qc.hw.sys, "platform", "win32")
    monkeypatch.setattr(hw_win32, "cpu_busy_percent", lambda w: 2.0)
    monkeypatch.setattr(
        hw_win32,
        "gpus",
        lambda run_fn: [
            {"vendor": "NVIDIA", "name": "RTX 5090", "vram_gib": 31.8, "vram_used_gib": 0.8, "busy_percent": 1}
        ],
    )
    monkeypatch.setattr(hw_win32, "memory", lambda: {"total_gib": 64.0, "available_gib": 50.0})
    monkeypatch.setattr(hw_win32, "power", lambda: {"on_battery": False})
    signals = qc.measure()
    result = qc.evaluate(signals)
    assert result["state"] == qc.OK and signals["cpu_busy_percent"] == 2.0
    assert signals["gpus"] == [{"busy": 1, "vram_gib": 31.8, "used_gib": 0.8}]
    assert "quiet enough" in qc.request(result)

    monkeypatch.setattr(hw_win32, "cpu_busy_percent", lambda w: None)
    monkeypatch.setattr(hw_win32, "gpus", lambda run_fn: [])
    signals_unknown = qc.measure()
    result_unknown = qc.evaluate(signals_unknown)
    assert result_unknown["state"] == qc.OK
    assert "50.0 GiB RAM" in qc.request(result_unknown)
