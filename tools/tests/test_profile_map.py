"""tools/coder_requant/profile_map.py: router rows of a cut matched to the source, the profile file round trip."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from coder_requant import profile_map


def test_each_cut_row_finds_its_source_row_and_unknown_or_twice_gives_minus_one():
    source = b"aa" + b"bb" + b"cc" + b"cc"  # rows of 2 bytes; "cc" twice
    cut = b"bb" + b"zz" + b"aa" + b"cc"
    assert profile_map.match_rows(cut, source, 2) == [1, -1, 0, -1]


def test_profile_round_trip(tmp_path):
    p = tmp_path / "p.bin"
    profile_map.write_profile(p, 2, 3, [1, 2, 3, 4, 5, 6])
    assert profile_map.read_profile(p) == (2, 3, [1, 2, 3, 4, 5, 6])
