## Plan (written before the run)

- Scope: workload server (2000:300 16000:100 at --ctx 32768); A `ep175` against B `ep175`; rounds 4 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: 2K start and 16K steady state; 100 decode tokens give a rate (decode tokens per prompt as in the targets, default 200)
- Duration: 2-2 min (4-4 rounds of 0.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-19-expert-profiles.md plan):
  - decode: +3 to +10 %
  - prefill@2000: +2 to +8 %
  - prefill@16000: +0 to +4 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin

## Result

A: `ep175` env {'TONA_EXPERT_PROFILE': 'none'}; B: `ep175` env -. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: all metrics decided after round 4. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 2293.7 / 2531.4 / 2293.2 / 2289.8 | 3220.1 / 2757.6 / 2765.2 / 3219.3 | 2293.5 (2289.8-2531.4) | 2992.2 (2757.6-3220.1) | +27.62 % [+15.54, +39.71] | **B faster** after round 4 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 121.7 / 122.3 / 121.6 / 121.8 | 125.3 / 124.1 / 124.1 / 125.3 | 121.7 (121.6-122.3) | 124.7 (124.1-125.3) | +2.32 % [+1.62, +3.03] | **B faster** after round 4 |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 4088.3 / 4090.0 / 4086.7 / 4086.0 | 4087.0 / 4087.6 / 4091.1 / 4092.3 | 4087.5 (4086.0-4090.0) | 4089.3 (4087.0-4092.3) | +0.04 % [-0.41, +0.50] | **equal** after round 4 |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 149.1 / 148.9 / 148.8 / 148.8 | 147.8 / 147.7 / 147.6 / 147.9 | 148.9 (148.8-149.1) | 147.7 (147.6-147.9) | -0.78 % [-1.24, -0.33] | **B slower** after round 4 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +40.39 % | +2.91 % | -0.03 % | -0.86 % | 30 |
| 2 | BA | +8.94 % | +1.44 % | -0.06 % | -0.84 % | 30 |
| 3 | AB | +20.58 % | +2.07 % | +0.11 % | -0.80 % | 30 |
| 4 | BA | +40.59 % | +2.88 % | +0.15 % | -0.63 % | 30 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +27.62 % | +2 to +8 % | **OUTSIDE the expectation: finding** |
| decode@2000 | +2.32 % | +3 to +10 % | **OUTSIDE the expectation: finding** |
| prefill@16000 | +0.04 % | +0 to +4 % | within the expectation |
| decode@16000 | -0.78 % | +3 to +10 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 83699, 3842 / 83699, 3842 / 83699, 3842 / 83699, 3842 | 83647, 3894 / 83647, 3894 / 83647, 3894 / 83647, 3894 | NO |
| 16000 | 21832, 1580 / 21832, 1580 / 21832, 1580 / 21832, 1580 | 21740, 1672 / 21740, 1672 / 21740, 1672 / 21740, 1672 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) | 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) |
| 16000 | 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) | 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | b82b2b4dad3975d4 | b82b2b4dad3975d4 | yes | yes | yes | - |
| 16000 | e85b642df9178deb | e85b642df9178deb | yes | yes | yes | - |

Runtime: 2.2 min for 4 rounds (0.5 min per round); a fixed 3 rounds would take 1.5 min (-0.5 min saved).
