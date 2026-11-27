## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: probe only, 2 rounds minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-early-commit.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1040-gaf228094be-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1040-gaf228094be-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 165.8 / 170.2 | 170.2 / 168.8 | 168.0 (165.8-170.2) | 169.5 (168.8-170.2) | +0.89 % [-1.29, +3.08] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.1 / 25.0 | 25.0 / 25.0 | 25.1 (25.0-25.1) | 25.0 (25.0-25.0) | -0.10 % [-0.96, +0.76] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 38.7 | 38.7 / 38.7 | 38.7 (38.7-38.7) | 38.7 (38.7-38.7) | +0.00 % [-0.81, +0.81] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 27.0 | 27.0 / 27.0 | 27.0 (27.0-27.0) | 27.0 (27.0-27.0) | +0.00 % [-0.81, +0.81] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +2.65 % | -0.34 % | +0.00 % | +0.00 % | 748 |
| 2 | BA | -0.87 % | +0.14 % | +0.00 % | +0.00 % | 747 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.89 % | -1 to +1 % | within the expectation |
| decode@8000 | -0.10 % | -1 to +1 % | within the expectation |
| decode-code | +0.00 % | -1 to +1 % | within the expectation |
| decode-text | +0.00 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.8 | 17.7 | 8.2 | 93.4 | 0.2 | 0.3 | 3141.5 | 6.4 | 13361.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.2 | 14.3 | 7.6 | 94.6 | 0.2 | 0.3 | 2377.4 | 3.3 | 12446.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.2 | 16.0 | 8.0 | 94.7 | 0.2 | 0.4 | 2478.6 | 5.5 | 13267.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.3 | 16.2 | 8.3 | 94.4 | 0.2 | 0.3 | 2927.8 | 2.7 | 13107.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.2 | 14.9 | 7.7 | 94.6 | 0.2 | 0.3 | 2333.1 | 2.4 | 12653.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.2 | 15.6 | 7.6 | 94.4 | 0.2 | 0.2 | 2464.2 | 1.6 | 12221.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.2 | 14.6 | 7.6 | 94.6 | 0.2 | 0.3 | 2402.7 | 1.6 | 12183.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.2 | 16.0 | 7.9 | 94.8 | 0.2 | 0.2 | 2396.4 | 1.6 | 12390.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.2 | 15.7 | 7.4 | 94.6 | 0.2 | 0.3 | 2330.2 | 3.5 | 12498.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 28.1 min for 2 rounds (12.5 min per round); a fixed 3 rounds would take 37.4 min (+12.5 min saved).
