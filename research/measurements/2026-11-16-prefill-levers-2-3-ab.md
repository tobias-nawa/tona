## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `<worktree>/work/build/w159` against B `<worktree>/work/build/w159`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the prompt lengths of coding agents; lever 3 acts at every group boundary (1 at 30K, 8 at 130K), lever 2 at every layer chunk (decode tokens per prompt as in the targets, default 200)
- Duration: 8-23 min (2-6 rounds of 3.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-breakdown.md levers 2 and 3):
  - prefill@30000: +0.3 to +2 %
  - prefill@130000: +2.2 to +5.2 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_PF_LOAN_KEEP=0, TONA_PF_WCACHE=0
  - B: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1', 'TONA_PF_WCACHE': '0', 'TONA_PF_LOAN_KEEP': '0'}; B: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 6, stop: all metrics decided after round 4. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1916.9 / 1917.0 / 1917.2 / 1917.0 | 1927.6 / 1927.3 / 1924.5 / 1928.0 | 1917.0 (1916.9-1917.2) | 1927.4 (1924.5-1928.0) | +0.51 % [+0.42, +0.60] | **B faster** after round 1 |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.5 / 69.2 / 69.0 / 68.5 | 69.5 / 69.6 / 68.9 / 70.3 | 69.1 (68.5-69.5) | 69.6 (68.9-70.3) | +0.78 % [-0.37, +1.92] | open (undecided at the last round) |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1794.5 / 1793.4 / 1791.3 / 1793.7 | 1825.5 / 1822.1 / 1825.2 / 1824.8 | 1793.5 (1791.3-1794.5) | 1825.0 (1822.1-1825.5) | +1.74 % [+1.64, +1.84] | **B faster** after round 4 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 61.6 / 62.0 / 62.1 / 61.4 | 61.2 / 62.3 / 61.3 / 62.7 | 61.8 (61.4-62.1) | 61.8 (61.2-62.7) | +0.15 % [-1.15, +1.45] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.56 % | -0.02 % | +1.73 % (off plan) | -0.64 % (off plan) | 227 |
| 2 | BA | +0.54 % | +0.65 % | +1.60 % (outlier) | +0.38 % | 227 |
| 3 | AB | +0.38 % | -0.22 % | +1.89 % (outlier) | -1.20 % | 227 |
| 4 | BA | +0.57 % | +2.69 % (outlier) | +1.73 % | +2.06 % (outlier) | 227 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.51 % | +0.3 to +2 % | within the expectation |
| decode@30000 | +0.78 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| prefill@130000 | +1.74 % | +2.2 to +5.2 % | **OUTSIDE the expectation: finding** |
| decode@130000 | +0.15 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 | 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 | yes |
| 130000 | 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 | 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 17.1 min for 4 rounds (3.8 min per round); a fixed 3 rounds would take 11.4 min (-3.8 min saved).
