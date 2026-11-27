"""tools/vram_watch.py: the scrubbing of program names, the card choice and the summary."""

import vram_watch as vw


def test_scrub_removes_program_names():
    line = "amdgpu 0000:03:00.0: VM memory stats for proc kitty(4970) task kitty:cs0(4969) is non-zero when fini"
    out = vw.scrub(line)
    assert "kitty" not in out
    assert "proc <name>" in out and "task <name>" in out


def test_kernel_pattern_matches_the_failure_messages_only():
    assert vw.KERNEL_PATTERN.search("[drm] *ERROR* Not enough memory for command submission!")
    assert vw.KERNEL_PATTERN.search("Failed to find memory space for buffer 0x000000 eviction")
    assert not vw.KERNEL_PATTERN.search("amdgpu: Topology: Add CPU node")


def test_pick_card_takes_the_largest_vram(tmp_path):
    for name, size in (("card0", 2 << 30), ("card1", 16 << 30)):
        d = tmp_path / name / "device"
        d.mkdir(parents=True)
        (d / "mem_info_vram_total").write_text(str(size))
    (tmp_path / "card2").mkdir()  # a card without the counters is skipped
    assert vw.pick_card(tmp_path) == tmp_path / "card1" / "device"
    assert vw.pick_card(tmp_path / "none") is None


def test_summary_tracks_the_lowest_headroom():
    s = vw.Summary()
    gib = 1 << 30

    def sample(t, used):
        return {
            "mem_info_gtt_used": used,
            "mem_info_gtt_total": 32 * gib,
            "mem_info_vram_used": 8 * gib,
            "mem_info_vram_total": 16 * gib,
        }

    s.add(0.0, sample(0.0, 20 * gib))
    s.add(0.1, sample(0.1, 31 * gib))
    s.add(0.2, sample(0.2, 25 * gib))
    assert s.min_headroom == gib and s.min_headroom_t == 0.1
    assert s.max_gtt_used == 31 * gib and s.samples == 3
    assert "lowest headroom 1.00 GiB at 0.1 s" in s.text()


def test_unreadable_counters_do_not_break_the_summary(tmp_path):
    c = vw.read_counters(tmp_path)  # no files
    assert set(c.values()) == {-1}
    s = vw.Summary()
    s.add(0.0, c)
    assert s.min_headroom is None
