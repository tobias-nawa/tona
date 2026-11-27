## Plan (written before the run)

- Scope: workload server (2000:300 16000:300 at --ctx 32768); A `ep175` against B `ep175`; rounds 4 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: code at 2K and 16K, as the prose pair (decode tokens per prompt as in the targets, default 200)
- Duration: 4-4 min (4-4 rounds of 1.0 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-19-expert-profiles.md plan):
  - prefill@2000: +3 to +9 %
  - decode@2000: -0.4 to +2.6 %
  - prefill@16000: -2 to +2 %
  - decode@16000: +1.8 to +4.8 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/nemotron-bolt-74b-a9b-24gb.bin

## Result

A: `ep175` env {'TONA_EXPERT_PROFILE': 'none'}; B: `ep175` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 1385.4 / 1381.8 / 1385.8 / 1383.1 | 1455.8 / 1450.5 / 1454.1 / 1452.1 | 1384.2 (1381.8-1385.8) | 1453.1 (1450.5-1455.8) | +4.99 % [+4.54, +5.44] | **B faster** after round 4 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 41.8 / 41.8 / 41.8 / 41.8 | 41.6 / 41.6 / 41.7 / 41.7 | 41.8 (41.8-41.8) | 41.7 (41.6-41.7) | -0.38 % [-0.84, +0.07] | open (undecided at the last round) |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 1855.4 / 1855.5 / 1855.3 / 1856.1 | 1859.0 / 1859.0 / 1861.0 / 1859.2 | 1855.4 (1855.3-1856.1) | 1859.1 (1859.0-1861.0) | +0.21 % [-0.24, +0.66] | open (undecided at the last round) |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 42.6 / 42.5 / 42.6 / 42.5 | 44.2 / 44.2 / 44.2 / 44.2 | 42.5 (42.5-42.6) | 44.2 (44.2-44.2) | +3.92 % [+3.47, +4.37] | **B faster** after round 4 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +5.08 % | -0.47 % | +0.19 % | +3.85 % | 62 |
| 2 | BA | +4.98 % | -0.39 % | +0.19 % | +4.01 % | 62 |
| 3 | AB | +4.93 % | -0.24 % | +0.31 % | +3.85 % | 62 |
| 4 | BA | +4.99 % | -0.42 % | +0.16 % | +3.97 % | 62 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +4.99 % | +3 to +9 % | within the expectation |
| decode@2000 | -0.38 % | -0.4 to +2.6 % | within the expectation |
| prefill@16000 | +0.21 % | -2 to +2 % | within the expectation |
| decode@16000 | +3.92 % | +1.8 to +4.8 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 82796, 25204 / 82796, 25204 / 82796, 25204 / 82796, 25204 | 66946, 41054 / 66946, 41054 / 66946, 41054 / 66946, 41054 | NO |
| 16000 | 85562, 22438 / 85562, 22438 / 85562, 22438 / 85562, 22438 | 105719, 2281 / 105719, 2281 / 105719, 2281 / 105719, 2281 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) |
| 16000 | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) | 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) / 1.00 (0.00) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | 32ce54598b07b41b | 32ce54598b07b41b | yes | yes | yes | - |
| 16000 | 4f33573361f2e52e | 4f33573361f2e52e | yes | yes | yes | - |

Runtime: 4.7 min for 4 rounds (1.0 min per round); a fixed 3 rounds would take 3.1 min (-1.0 min saved).
