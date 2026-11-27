## Plan (written before the run)

- Scope: workload server (2000:300 16000:100 at --ctx 32768); A `ep175` against B `ep175`; rounds 6 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: repeat of run 1 with 6 rounds, same sizes (decode tokens per prompt as in the targets, default 200)
- Duration: 3-3 min (6-6 rounds of 0.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: run 1 of this file):
  - decode: -1.2 to +3 %
  - prefill@2000: +10 to +40 %
  - prefill@16000: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0
  - B: build commit v0.0.2-1339-g5034f45806-dirty; ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/toolchain/profiles/ornith-1.5-35b-a3b.bin

## Result

A: `ep175` env {'TONA_EXPERT_PROFILE': 'none'}; B: `ep175` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: all metrics decided after round 6. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@2000: Prefill tok/s after 2000 tokens of prompt | 2291.6 / 2290.6 / 2294.5 / 2290.9 / 2291.1 / 2287.7 | 2761.1 / 2756.1 / 2752.0 / 2753.8 / 2751.4 / 2749.7 | 2291.0 (2287.7-2294.5) | 2752.9 (2749.7-2761.1) | +20.21 % [+19.87, +20.54] | **B faster** after round 6 |
| decode@2000: Decode tok/s after 2000 tokens of prompt | 121.8 / 121.8 / 121.8 / 121.8 / 121.8 / 121.8 | 124.2 / 124.2 / 124.3 / 123.8 / 124.4 / 124.0 | 121.8 (121.8-121.8) | 124.2 (123.8-124.4) | +1.94 % [+1.61, +2.27] | **B faster** after round 6 |
| prefill@16000: Prefill tok/s after 16000 tokens of prompt | 4089.7 / 4089.7 / 4088.7 / 4089.3 / 4090.6 / 4087.1 | 4090.1 / 4090.3 / 4086.0 / 4090.0 / 4089.0 / 4085.4 | 4089.5 (4087.1-4090.6) | 4089.5 (4085.4-4090.3) | -0.02 % [-0.33, +0.29] | **equal** after round 6 |
| decode@16000: Decode tok/s after 16000 tokens of prompt | 149.1 / 148.7 / 148.8 / 149.0 / 149.2 / 148.4 | 148.1 / 147.8 / 148.0 / 148.3 / 147.8 / 148.0 | 148.9 (148.4-149.2) | 148.0 (147.8-148.3) | -0.59 % [-0.94, -0.24] | **B slower** after round 6 |

| Round | Order | prefill@2000 B vs A | decode@2000 B vs A | prefill@16000 B vs A | decode@16000 B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +20.49 % | +1.92 % | +0.01 % | -0.66 % | 30 |
| 2 | BA | +20.33 % | +2.03 % | +0.02 % | -0.59 % | 30 |
| 3 | AB | +19.94 % | +2.11 % | -0.07 % | -0.53 % | 30 |
| 4 | BA | +20.21 % | +1.68 % | +0.01 % | -0.48 % | 30 |
| 5 | AB | +20.09 % | +2.12 % | -0.04 % | -0.99 % | 30 |
| 6 | BA | +20.20 % | +1.81 % | -0.04 % | -0.27 % | 30 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@2000 | +20.21 % | +10 to +40 % | within the expectation |
| decode@2000 | +1.94 % | -1.2 to +3 % | within the expectation |
| prefill@16000 | -0.02 % | -0.5 to +0.5 % | within the expectation |
| decode@16000 | -0.59 % | -1.2 to +3 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 2000 | 83699, 3842 / 83699, 3842 / 83699, 3842 / 83699, 3842 / 83699, 3842 / 83699, 3842 | 83647, 3894 / 83647, 3894 / 83647, 3894 / 83647, 3894 / 83647, 3894 / 83647, 3894 | NO |
| 16000 | 21832, 1580 / 21832, 1580 / 21832, 1580 / 21832, 1580 / 21832, 1580 / 21832, 1580 | 21740, 1672 / 21740, 1672 / 21740, 1672 / 21740, 1672 / 21740, 1672 / 21740, 1672 | NO |

| Request | A tokens per pass (draft steps per pass) | B tokens per pass (draft steps per pass) |
|---|---|---|
| 2000 | 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) | 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) / 2.47 (2.58) |
| 16000 | 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) | 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) / 3.57 (1.64) |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 2000 | b82b2b4dad3975d4 | b82b2b4dad3975d4 | yes | yes | yes | - |
| 16000 | e85b642df9178deb | e85b642df9178deb | yes | yes | yes | - |

Runtime: 3.2 min for 6 rounds (0.5 min per round); a fixed 3 rounds would take 1.5 min (-1.5 min saved).
