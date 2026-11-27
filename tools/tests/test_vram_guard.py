"""The VRAM safety rules without a GPU (tools/vram_guard.py, tools/vram_balloon.py ladder)."""

import time

import vram_balloon
import vram_guard as vg

GIB, MIB = vg.GIB, vg.MIB


def card(tmp_path, name, total, used, bus="0000:03:00.0"):
    real = tmp_path / "pci" / bus
    real.mkdir(parents=True, exist_ok=True)
    (real / "mem_info_vram_total").write_text(f"{total}\n")
    (real / "mem_info_vram_used").write_text(f"{used}\n")
    link = tmp_path / "drm" / name
    link.mkdir(parents=True, exist_ok=True)
    (link / "device").symlink_to(real)
    return link / "device"


def test_free_bytes_fails_closed():
    assert vg.free_bytes(16 * GIB, 4 * GIB) == 12 * GIB
    for total, used in ((None, 1), (1, None), (0, 0), (10, -1), (10, 11)):
        assert vg.free_bytes(total, used) is None


def test_floor_is_one_and_a_half_gib_and_step_is_64_mib():
    assert vg.FLOOR_BYTES == 1536 * MIB
    assert vg.STEP_BYTES == 64 * MIB


def test_may_allocate_never_goes_below_the_floor():
    assert vg.may_allocate(vg.FLOOR_BYTES + vg.STEP_BYTES)  # leaves exactly the floor
    assert not vg.may_allocate(vg.FLOOR_BYTES + vg.STEP_BYTES - 1)
    assert not vg.may_allocate(vg.FLOOR_BYTES)
    assert not vg.may_allocate(0)
    assert not vg.may_allocate(None)
    assert not vg.may_allocate(10 * GIB, step=0)


def test_steps_available_is_capped_and_zero_below_the_floor():
    assert vg.steps_available(None) == 0
    assert vg.steps_available(vg.FLOOR_BYTES) == 0
    assert vg.steps_available(vg.FLOOR_BYTES + 3 * vg.STEP_BYTES) == 3
    assert vg.steps_available(16 * GIB) == vg.MAX_STEPS


def test_pick_card_takes_the_big_card_and_ignores_the_igpu(tmp_path):
    card(tmp_path, "card0", 2 * GIB, 0, bus="0000:0f:00.0")
    big = card(tmp_path, "card1", 16 * GIB, GIB)
    assert vg.pick_card(tmp_path / "drm").resolve() == big.resolve()
    assert vg.read_free(big) == 15 * GIB
    assert vg.bus_id_of(big) == "0000:03:00.0"


def test_pick_card_refuses_when_only_a_small_card_exists(tmp_path):
    card(tmp_path, "card0", 2 * GIB, 0)
    assert vg.pick_card(tmp_path / "drm") is None


def test_same_card_compares_the_bus_id(tmp_path):
    dev = card(tmp_path, "card1", 16 * GIB, 0)
    assert vg.same_card("0000:03:00.0\n", dev)
    assert vg.same_card("0000:03:00.0".upper(), dev)
    assert not vg.same_card("0000:0f:00.0", dev)


def test_read_free_is_none_when_a_counter_is_missing(tmp_path):
    dev = card(tmp_path, "card1", 16 * GIB, 0)
    (dev / "mem_info_vram_used").unlink()
    assert vg.read_free(dev) is None


def test_kernel_abort_lines():
    assert vg.kernel_abort("amdgpu 0000:03:00.0: GPU reset begin!")
    assert vg.kernel_abort("amdgpu: ring gfx_0.0.0 timeout, signaled seq=5")
    assert vg.kernel_abort("[drm:amdgpu_cs_ioctl] Not enough memory for command submission!")
    assert not vg.kernel_abort("amdgpu: Failed to find memory space for buffer eviction")
    assert not vg.kernel_abort("usb 1-1: new device")


def test_scrub_removes_program_names():
    assert "firefox" not in vg.scrub("VM memory stats for proc firefox(123) task x:cs0(1)")


def state(**kw):
    base = dict(
        elapsed_s=1.0, stop_file=False, abort_line=None, canary_failed=False, heartbeat_age_s=None, free=8 * GIB
    )
    base.update(kw)
    return vg.StopState(**base)


def test_stop_reason_each_rule():
    assert vg.stop_reason(state()) is None
    assert "kernel log" in vg.stop_reason(state(abort_line="GPU reset"))
    assert vg.stop_reason(state(stop_file=True)) == "STOP file"
    assert vg.stop_reason(state(canary_failed=True)) == "canary failed"
    assert vg.stop_reason(state(free=None)) == "free VRAM unreadable"
    assert "below the floor" in vg.stop_reason(state(free=vg.FLOOR_BYTES - 1))
    assert vg.stop_reason(state(heartbeat_age_s=6.0)) == "heartbeat stopped"
    assert vg.stop_reason(state(heartbeat_age_s=1.0)) is None
    assert vg.stop_reason(state(elapsed_s=vg.TIME_CAP_S + 1)) == "time cap"


def test_stop_reason_prefers_the_kernel_log_over_everything():
    s = state(abort_line="x", stop_file=True, canary_failed=True, free=None)
    assert vg.stop_reason(s).startswith("kernel log")


def test_watchdog_kills_on_stale_or_missing_heartbeat_or_dead_parent():
    assert not vg.watchdog_kills(1.0, True)
    assert vg.watchdog_kills(5.5, True)
    assert vg.watchdog_kills(None, True)
    assert vg.watchdog_kills(0.1, False)


class FakeCard:
    """A card whose free memory the fake allocations use up; `other` is memory some other program takes per step."""

    def __init__(self, free, other=0):
        self.free, self.other, self.allocs, self.freed = free, other, [], []

    def read(self):
        return self.free

    def alloc(self, size):
        # the real allocator would just succeed: the rule must have said no before this is called
        self.allocs.append((size, self.free))
        self.free -= size + self.other
        return len(self.allocs)


def ladder(fake, check=lambda free: None, max_steps=vg.MAX_STEPS):
    events = []
    held = vram_balloon.run_ladder(
        fake.read, fake.alloc, check, lambda e, **kw: events.append((e, kw)), lambda: None, max_steps
    )
    return held, events


def test_ladder_never_allocates_when_the_step_would_cross_the_floor():
    fake = FakeCard(vg.FLOOR_BYTES + 3 * vg.STEP_BYTES + 5)
    held, events = ladder(fake)
    assert len(held) == 3
    assert all(free - size >= vg.FLOOR_BYTES for size, free in fake.allocs)
    assert fake.free >= vg.FLOOR_BYTES
    assert events[-1][0] == "stop" and "no headroom" in events[-1][1]["reason"]


def test_ladder_takes_nothing_when_the_card_starts_below_the_floor():
    fake = FakeCard(vg.FLOOR_BYTES - 1)
    held, events = ladder(fake, check=lambda free: vg.stop_reason(state(free=free)))
    assert held == [] and fake.allocs == []
    assert events[-1][0] == "stop"


def test_ladder_rereads_free_before_every_step():
    # another program takes 256 MiB per step: the rule must see it and stop earlier than the plan
    fake = FakeCard(vg.FLOOR_BYTES + 10 * vg.STEP_BYTES, other=256 * MIB)
    held, _ = ladder(fake)
    assert len(held) == 2  # 640 MiB above the floor, 320 MiB gone per step
    assert all(free - size >= vg.FLOOR_BYTES for size, free in fake.allocs)


def test_ladder_stops_at_the_step_cap():
    fake = FakeCard(16 * GIB)
    held, events = ladder(fake, max_steps=4)
    assert len(held) == 4 and events[-1][1]["reason"] == "step cap"


def test_ladder_stops_on_a_stop_rule_before_allocating():
    fake = FakeCard(16 * GIB)
    seen = []

    def check(free):
        seen.append(free)
        return "STOP file" if len(seen) == 3 else None

    held, events = ladder(fake, check=check)
    assert len(held) == 2 and events[-1][1]["reason"] == "STOP file"


def test_ladder_stops_on_an_unreadable_free_value():
    class Broken(FakeCard):
        def read(self):
            return None

    fake = Broken(16 * GIB)
    held, _ = ladder(fake)
    assert held == [] and fake.allocs == []


def test_ladder_stops_when_an_allocation_fails():
    fake = FakeCard(16 * GIB)
    fake.alloc = lambda size: None
    held, events = ladder(fake)
    assert held == [] and events[-1][0] == "alloc_failed"


def test_hold_returns_the_first_reason():
    calls = []

    def check():
        calls.append(1)
        return "STOP file" if len(calls) == 2 else None

    t0 = time.monotonic()
    assert vram_balloon.hold(check, 30.0, lambda s: None) == "STOP file"
    assert time.monotonic() - t0 < 5
    assert vram_balloon.hold(lambda: None, 0.0, lambda s: None) is None


def test_watchdog_kills_the_victim_by_pid_when_the_parent_is_gone(tmp_path, monkeypatch):
    import argparse
    import subprocess

    import vram_watchdog

    monkeypatch.setattr(vram_watchdog, "START_GRACE_S", 0.0)
    done = subprocess.Popen(["true"])
    done.wait()
    victim = subprocess.Popen(["sleep", "60"])
    args = argparse.Namespace(heartbeat=tmp_path / "hb", parent=done.pid, kill=[victim.pid], watch=None)
    assert vram_watchdog.watch(args) == 1
    assert victim.wait(5) == -9


def test_watchdog_ends_quietly_when_the_balloon_ended_by_itself(tmp_path):
    import argparse
    import subprocess

    import vram_watchdog

    done = subprocess.Popen(["true"])
    done.wait()
    args = argparse.Namespace(heartbeat=tmp_path / "hb", parent=done.pid, kill=[1], watch=done.pid)
    assert vram_watchdog.watch(args) == 0


def test_phase_summary_uses_only_the_window():
    import vram_level0 as lv

    rec = lv.Recorder()
    rec.decodes = [(1.0, 70.0), (5.0, 60.0), (6.0, 62.0), (11.0, 10.0)]
    rec.canary = [(5.0, 100.0), (5.5, 300.0), (12.0, 9999.0)]
    out = lv.phase_summary(rec, 5.0, 10.0, [{"t": 5.0, "free_mib": 1700}, {"t": 6.0, "free_mib": 1600}])
    assert out["decode_requests"] == 2 and out["decode_tok_s_mean"] == 61.0
    assert out["canary_us_max"] == 300.0 and out["free_mib_min"] == 1600
    assert lv.percentile([], 0.5) is None


def test_a_request_cut_off_by_the_shutdown_is_not_a_decode_failure(monkeypatch):
    import vram_level0 as lv

    def cut(*_a, **_k):
        rec.stop.set()  # the shutdown happens while the request is in flight
        raise ConnectionError

    monkeypatch.setattr(lv.ab_server, "post", cut)
    rec = lv.Recorder()
    rec.decode_loop("http://127.0.0.1:1")
    assert rec.decode_error is None

    rec2 = lv.Recorder()
    monkeypatch.setattr(lv.ab_server, "post", lambda *_a, **_k: (_ for _ in ()).throw(ConnectionError()))
    rec2.decode_loop("http://127.0.0.1:1")
    assert rec2.decode_error == "decode request failed: ConnectionError"
