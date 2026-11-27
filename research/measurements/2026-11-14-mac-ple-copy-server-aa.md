## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: A/A, as many starts as the pin run, to see slow and fast starts (decode tokens per prompt as in the targets, default 200)
- Duration: 19-57 min (2-6 rounds of 9.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-ab-order.md):
  - decode-code: -3 to +3 %
  - decode-text: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-998-ga4ca6db1f4-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_STATS=1
  - B: build commit stand-2026-10-25-998-ga4ca6db1f4-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_STATS=1

## Result

A: `work/build/mtl` env {'TONA_PLE_STATS': '1'}; B: `work/build/mtl` env {'TONA_PLE_STATS': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 37.0 / 36.8 / 37.1 / 38.3 / 37.6 / 38.1 | 36.8 / 36.6 / 37.1 / 38.2 / 36.6 / 37.5 | 37.4 (36.8-38.3) | 37.0 (36.6-38.2) | -0.93 % [-2.14, +0.28] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 25.4 / 25.6 / 25.6 / 26.3 / 25.0 / 26.4 | 25.6 / 25.3 / 25.3 / 26.7 / 24.9 / 26.7 | 25.6 (25.0-26.4) | 25.5 (24.9-26.7) | +0.12 % [-1.28, +1.51] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -0.54 % | +0.79 % | 580 |
| 2 | BA | -0.54 % | -1.17 % | 580 |
| 3 | AB | +0.00 % | -1.17 % | 576 |
| 4 | BA | -0.26 % | +1.52 % | 555 |
| 5 | AB | -2.66 % | -0.40 % | 573 |
| 6 | BA | -1.57 % | +1.14 % | 560 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | -0.93 % | -3 to +3 % | within the expectation |
| decode-text | +0.12 % | -3 to +3 % | within the expectation |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.0 | 45.5 | 14.3 | 87.8 | 0.1 | 0.1 | 4368.4 | 75.1 | 15645.8 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 6.2 | 53.3 | 10.9 | 88.5 | 0.3 | 0.1 | 3084.7 | 32.8 | 9785.1 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.8 | 87.5 | 11.3 | 89.2 | 0.3 | 0.1 | 3207.2 | 18.7 | 9110.4 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.8 | 75.6 | 10.8 | 89.1 | 0.1 | 0.1 | 3570.5 | 17.9 | 10457.4 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.9 | 89.5 | 11.6 | 89.1 | 0.1 | 0.1 | 3956.8 | 9.0 | 12473.8 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.8 | 69.3 | 10.6 | 89.1 | 0.1 | 0.1 | 3363.5 | 10.4 | 9292.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.6 | 47.8 | 9.7 | 89.3 | 0.1 | 0.1 | 2258.7 | 8.1 | 6159.5 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.9 | 77.2 | 10.9 | 89.0 | 0.1 | 0.1 | 3850.7 | 9.6 | 10866.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.9 | 71.4 | 10.8 | 88.9 | 0.1 | 0.1 | 2843.4 | 10.1 | 8135.7 | 5 | 0.59 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 6.5 | 114.4 | 11.3 | 88.7 | 0.1 | 0.1 | 4191.4 | 6.3 | 13294.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.5 | 94.5 | 11.2 | 88.4 | 0.1 | 0.1 | 4399.6 | 30.5 | 11339.6 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 6.0 | 51.1 | 8.5 | 89.0 | 0.1 | 0.1 | 1535.5 | 116.1 | 2575.2 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.8 | 66.3 | 7.9 | 89.3 | 0.1 | 0.1 | 1363.9 | 15.3 | 2094.2 | 9 | 0.79 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.1 | 18.8 | 8.2 | 94.1 | 0.4 | 0.1 | 248.8 | 29.2 | 1673.1 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 4.6 | 15.3 | 7.3 | 94.7 | 0.1 | 0.1 | 217.8 | 12.2 | 1310.4 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 4.6 | 17.4 | 7.2 | 94.7 | 0.1 | 0.1 | 214.8 | 5.1 | 957.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 4.8 | 28.2 | 7.9 | 94.2 | 0.1 | 0.1 | 580.0 | 34.7 | 2221.3 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 4.7 | 17.8 | 7.2 | 94.5 | 0.1 | 0.1 | 280.3 | 8.9 | 1363.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 5.3 | 55.1 | 17.0 | 90.9 | 0.1 | 0.1 | 1531.2 | 318.3 | 6396.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 5.4 | 64.1 | 10.5 | 89.4 | 0.1 | 0.1 | 1920.7 | 35.5 | 3289.5 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 5.4 | 19.5 | 10.1 | 89.2 | 0.1 | 0.1 | 1542.9 | 10.0 | 3871.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 5.2 | 19.0 | 8.0 | 91.6 | 0.3 | 0.1 | 923.5 | 10.7 | 2061.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 5.2 | 17.6 | 9.8 | 92.7 | 0.5 | 0.2 | 491.2 | 3.2 | 1516.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 4.7 | 16.0 | 8.0 | 94.8 | 0.1 | 0.1 | 213.4 | 2.5 | 1915.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 4.7 | 14.3 | 7.2 | 94.3 | 0.1 | 0.1 | 291.8 | 1.6 | 1547.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |


Runtime: 59.5 min for 6 rounds (9.5 min per round); a fixed 3 rounds would take 28.5 min (-28.5 min saved).
