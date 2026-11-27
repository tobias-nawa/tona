## Plan (written before the run)

- Scope: workload prose (2000:prose:400 16000:prose:400 at --ctx 32768); A `ep175` against B `ep175`; rounds 4 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 2K shows the cache start, 16K the steady state (bolt-opt2 E3) (decode tokens per prompt as in the targets, default 200)
- Duration: 5-5 min (4-4 rounds of 1.2 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-19-expert-profiles.md plan):
  - prefill@2000:prose: +2.8 to +8.8 %
  - decode@2000:prose: -1 to +0.4 %
  - prefill@16000:prose: -1.9 to +2.1 %
  - decode@16000:prose: +1.8 to +4.8 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin

## Result

A: `ep175` env {'TONA_EXPERT_PROFILE': 'none'}; B: `ep175` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000:prose: Prefill tok/s after 2000:prose tokens of prompt | 1357.6 / 1357.5 / 1356.4 / 1357.1 | 1426.8 / 1426.2 / 1426.4 / 1427.5 | 1357.3 (1356.4-1357.6) | 1426.6 (1426.2-1427.5) | +5.13 % [+4.68, +5.58] | **B faster** after round 4 |
| decode@2000:prose: Decode tok/s after 2000:prose tokens of prompt | 41.8 / 42.1 / 42.0 / 42.1 | 41.5 / 41.5 / 41.5 / 41.5 | 42.1 (41.8-42.1) | 41.5 (41.5-41.5) | -1.20 % [-1.72, -0.69] | **B slower** after round 4 |
| prefill@16000:prose: Prefill tok/s after 16000:prose tokens of prompt | 1848.7 / 1850.2 / 1850.1 / 1847.1 | 1853.1 / 1851.5 / 1850.8 / 1851.1 | 1849.4 (1847.1-1850.2) | 1851.3 (1850.8-1853.1) | +0.14 % [-0.31, +0.59] | open (undecided at the last round) |
| decode@16000:prose: Decode tok/s after 16000:prose tokens of prompt | 42.4 / 42.5 / 42.4 / 42.4 | 44.2 / 44.2 / 44.2 / 44.2 | 42.4 (42.4-42.5) | 44.2 (44.2-44.2) | +4.13 % [+3.67, +4.58] | **B faster** after round 4 |

| Round | Order | prefill@2000:prose B vs A | decode@2000:prose B vs A | prefill@16000:prose B vs A | decode@16000:prose B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +5.10 % | -0.79 % | +0.24 % | +4.24 % | 72 |
| 2 | BA | +5.06 % | -1.36 % | +0.07 % | +4.02 % | 72 |
| 3 | AB | +5.16 % | -1.13 % | +0.04 % | +4.08 % | 72 |
| 4 | BA | +5.19 % | -1.54 % | +0.21 % | +4.16 % | 72 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000:prose | +5.13 % | +2.8 to +8.8 % | within the expectation |
| decode@2000:prose | -1.20 % | -1 to +0.4 % | **OUTSIDE the expectation: finding** |
| prefill@16000:prose | +0.14 % | -1.9 to +2.1 % | within the expectation |
| decode@16000:prose | +4.13 % | +1.8 to +4.8 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000:prose | 107906, 36094 / 107906, 36094 / 107906, 36094 / 107906, 36094 | 86058, 57942 / 86058, 57942 / 86058, 57942 / 86058, 57942 | NO |
| 16000:prose | 111821, 32179 / 111821, 32179 / 111821, 32179 / 111821, 32179 | 138972, 5028 / 138972, 5028 / 138972, 5028 / 138972, 5028 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000:prose | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) |
| 16000:prose | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000:prose | 03bc3042e3ebf8fa | 03bc3042e3ebf8fa | yes | yes | yes | - |
| 16000:prose | 90eabd7be104c3f2 | 90eabd7be104c3f2 | yes | yes | yes | - |

Runtime: 5.4 min for 4 rounds (1.2 min per round); a fixed 3 rounds would take 3.6 min (-1.2 min saved).
