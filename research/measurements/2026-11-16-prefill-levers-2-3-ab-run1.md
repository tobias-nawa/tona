## Plan (written before the run)

- Scope: workload mid,long (30000:200 130000:200 at --ctx 140000); A `<worktree>/work/build/w159` against B `<worktree>/work/build/w159`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (prefill) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 30K and 130K are the prompt lengths of coding agents; lever 3 acts at every group boundary (1 at 30K, 8 at 130K), lever 2 at every layer chunk (decode tokens per prompt as in the targets, default 200)
- Duration: 8-23 min (2-6 rounds of 3.8 min, from round 1)
- Known noise (bench/noise.toml; one round may decide these metrics): prefill@30000 sd 0.04 % (df 9), decode@30000 sd 0.279 % (df 9), prefill@130000 sd 0.018 % (df 9), decode@130000 sd 0.42 % (df 9)
- Expectation (B against A, source: research/measurements/2026-11-16-prefill-breakdown.md levers 2 and 3):
  - prefill@30000: +1.2 to +4.2 %
  - prefill@130000: +2.2 to +5.2 %
  - decode: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin, TONA_PF_LOAN_KEEP=0, TONA_PF_WCACHE=0
  - B: build commit v0.0.2-941-ga7f0f5d18b-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_EXPERT_PROFILE=<repo>/work/profile/experts-coder.bin

## Result

A: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1', 'TONA_PF_WCACHE': '0', 'TONA_PF_LOAN_KEEP': '0'}; B: `<worktree>/work/build/w159` env {'TONA_DETERMINISTIC': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@30000: Prefill tok/s after 30000 tokens of prompt | 1878.2 / 1916.2 / 1916.0 / 1915.8 / 1913.7 / 1918.0 | 1913.7 / 1914.3 / 1915.2 / 1915.5 / 1914.9 / 1914.4 | 1915.9 (1878.2-1918.0) | 1914.6 (1913.7-1915.5) | +0.27 % [-0.37, +0.91] | open (undecided at the last round) |
| decode@30000: Decode tok/s after 30000 tokens of prompt | 69.8 / 69.7 / 69.6 / 69.3 / 70.1 / 68.3 | 69.7 / 69.7 / 69.7 / 69.5 / 69.9 / 69.6 | 69.7 (68.3-70.1) | 69.7 (69.5-69.9) | +0.32 % [-0.40, +1.04] | **equal** after round 3 |
| prefill@130000: Prefill tok/s after 130000 tokens of prompt | 1795.4 / 1792.5 / 1794.0 / 1793.8 / 1793.6 / 1794.9 | 1821.8 / 1823.4 / 1823.1 / 1822.1 / 1824.2 / 1822.9 | 1793.9 (1792.5-1795.4) | 1823.0 (1821.8-1824.2) | +1.61 % [+1.53, +1.69] | **B faster** after round 3 |
| decode@130000: Decode tok/s after 130000 tokens of prompt | 62.0 / 62.1 / 62.1 / 61.8 / 62.3 / 62.3 | 61.5 / 61.6 / 61.9 / 62.3 / 62.0 / 61.5 | 62.1 (61.8-62.3) | 61.7 (61.5-62.3) | -0.46 % [-1.18, +0.25] | open (undecided at the last round) |

| Round | Order | prefill@30000 B vs A | decode@30000 B vs A | prefill@130000 B vs A | decode@130000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.89 % | -0.20 % | +1.47 % (off plan) | -0.74 % (off plan) | 228 |
| 2 | BA | -0.10 % (outlier) | +0.05 % | +1.73 % (outlier) | -0.80 % | 227 |
| 3 | AB | -0.04 % (outlier) | +0.11 % | +1.62 % | -0.22 % | 227 |
| 4 | BA | -0.01 % | +0.27 % | +1.58 % | +0.77 % | 227 |
| 5 | AB | +0.06 % | -0.27 % | +1.70 % (outlier) | -0.51 % | 227 |
| 6 | BA | -0.19 % (outlier) | +1.94 % (outlier) | +1.56 % | -1.28 % | 227 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@30000 | +0.27 % | +1.2 to +4.2 % | **OUTSIDE the expectation: finding** |
| decode@30000 | +0.32 % | -0.5 to +0.5 % | within the expectation |
| prefill@130000 | +1.61 % | +2.2 to +5.2 % | **OUTSIDE the expectation: finding** |
| decode@130000 | -0.46 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 30000 | 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 | 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 / 71098, 8207 | yes |
| 130000 | 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 | 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 / 76068, 7533 | yes |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 30000 | ee58880d3f5b558f | ee58880d3f5b558f | yes | yes | yes | ee58880d3f5b558f (B = ref) |
| 130000 | 7d2996546076d1b0 | 7d2996546076d1b0 | yes | yes | yes | 7d2996546076d1b0 (B = ref) |

Runtime: 24.7 min for 6 rounds (3.8 min per round); a fixed 3 rounds would take 11.4 min (-11.4 min saved).
