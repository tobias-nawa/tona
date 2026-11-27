"""The arithmetic of scripts/expert_bw.py and tools/prefill_breakdown.py (no GPU, no model)."""

import expert_bw
import prefill_breakdown


def test_expert_bytes_counts_whole_blocks():
    # Q2_0: 64 values in 18 bytes; 2 rows of 128 values are 4 blocks
    assert expert_bw.expert_bytes(("Q2_0", 128, 2)) == 72
    assert expert_bw.CEILING_GBS > 0


def test_busy_union_counts_overlaps_once():
    assert prefill_breakdown.busy_union([(0, 10), (5, 15), (20, 30)]) == 25


def test_runs_split_at_long_gaps():
    kernels = [(0, 10), (12, 20), (100, 110)]
    assert prefill_breakdown.runs(kernels, gap_ns=50) == [(0, 2), (2, 3)]
    assert prefill_breakdown.PEAK_FP8 == 2 * prefill_breakdown.PEAK_F16
