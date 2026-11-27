## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `<worktree>/work/build/w159` against B `<worktree>/work/build/w159`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the prompt lengths of coding agents; lever 3 acts at every group boundary (1 at 30K, 8 at 130K), lever 2 at every layer chunk (decode tokens per prompt as in the targets, default 200)
- Duration: 8-8 min (2-2 rounds of 3.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-breakdown.md levers 2 and 3):
  - prefill@30000: +0.3 to +2 %
  - prefill@130000: +2.2 to +5.2 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-942-gbd0e9ba6d8-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_PF_LOAN_KEEP=0, TONA_PF_WCACHE=0
  - B: build commit v0.0.2-942-gbd0e9ba6d8-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1', 'TONA_PF_WCACHE': '0', 'TONA_PF_LOAN_KEEP': '0'}; B: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1918.6 / 1916.6 | 1929.3 / 1927.4 | 1917.6 (1916.6-1918.6) | 1928.4 (1927.4-1929.3) | +0.56 % [+0.49, +0.64] | **B faster** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 70.2 / 69.1 | 70.4 / 70.5 | 69.7 (69.1-70.2) | 70.4 (70.4-70.5) | +1.13 % [+0.24, +2.01] | open (undecided at the last round) |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1795.3 / 1794.3 | 1826.5 / 1824.6 | 1794.8 (1794.3-1795.3) | 1825.6 (1824.6-1826.5) | +1.71 % [+1.67, +1.75] | **B faster** after round 2 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 63.0 / 61.7 | 63.1 / 63.0 | 62.4 (61.7-63.0) | 63.0 (63.0-63.1) | +1.07 % [-0.12, +2.27] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.56 % | +0.31 % | +1.74 % (off plan) | +0.06 % | 231 |
| 2 | BA | +0.56 % | +1.94 % (outlier) | +1.69 % | +2.09 % (outlier) | 227 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.56 % | +0.3 to +2 % | within the expectation |
| decode@30000 | +1.13 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@130000 | +1.71 % | +2.2 to +5.2 % | **OUTSIDE the expectation: finding** |
| decode@130000 | +1.07 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71098, 8207 / 71098, 8207 | 71098, 8207 / 71098, 8207 | yes |
| 130000 | 76068, 7533 / 76068, 7533 | 76068, 7533 / 76068, 7533 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 9.6 min for 2 rounds (3.8 min per round); a fixed 3 rounds would take 11.4 min (+3.8 min saved).
