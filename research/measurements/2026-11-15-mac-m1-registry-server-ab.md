## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `<repo>/work/build/main` against B `<repo>/work/build/m1`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: no compute change; 2 to 6 rounds until decided at margin 0.5 % (decode tokens per prompt as in the targets, default 200)
- Duration: 25-74 min (2-6 rounds of 12.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-m1-registry.md):
  - decode-code: -0.5 to +0.5 %
  - decode-text: -0.5 to +0.5 %
  - decode@8000: -0.5 to +0.5 %
  - prefill@8000: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1078-g47c5a41a10; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1081-g221ebd6792; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `<repo>/work/build/main` env -; B: `<repo>/work/build/m1` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 161.8 / 166.2 / 167.7 / 168.1 / 170.2 / 172.5 | 161.4 / 168.7 / 166.3 / 171.8 / 173.1 / 172.6 | 167.9 (161.8-172.5) | 170.3 (161.4-173.1) | +0.75 % [-0.69, +2.19] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.4 / 25.5 / 25.0 / 25.4 / 25.0 / 25.0 | 25.5 / 25.5 / 25.4 / 25.0 / 25.1 / 25.0 | 25.2 (25.0-25.5) | 25.2 (25.0-25.5) | +0.07 % [-1.26, +1.39] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 39.4 / 39.6 / 38.9 / 39.1 / 38.7 / 38.7 | 39.4 / 39.4 / 39.2 / 39.2 / 38.7 / 38.6 | 39.0 (38.7-39.6) | 39.2 (38.6-39.4) | +0.04 % [-0.65, +0.74] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.5 / 27.8 / 27.0 / 26.9 / 27.0 / 27.0 | 27.7 / 27.7 / 27.4 / 27.4 / 27.0 / 27.0 | 27.0 (26.9-27.8) | 27.4 (27.0-27.7) | +0.62 % [-0.49, +1.72] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.20 % | +0.59 % | +0.00 % | +0.73 % | 746 |
| 2 | BA | +1.55 % | +0.10 % | -0.51 % | -0.36 % | 740 |
| 3 | AB | -0.85 % | +1.48 % | +0.77 % | +1.48 % | 746 |
| 4 | BA | +2.21 % | -1.89 % | +0.26 % | +1.86 % | 741 |
| 5 | AB | +1.68 % | +0.29 % | +0.00 % | +0.00 % | 746 |
| 6 | BA | +0.08 % | -0.18 % | -0.26 % | +0.00 % | 744 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.75 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +0.07 % | -0.5 to +0.5 % | within the expectation |
| decode-code | +0.04 % | -0.5 to +0.5 % | within the expectation |
| decode-text | +0.62 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 8.3 | 63.6 | 30.4 | 92.2 | 0.5 | 0.4 | 5631.4 | 19.0 | 27458.1 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 7.4 | 36.4 | 106.4 | 92.9 | 0.3 | 0.3 | 7097.1 | 1301.9 | 26693.5 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 7.5 | 56.4 | 112.4 | 92.9 | 0.3 | 0.3 | 8906.5 | 222.3 | 32279.7 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 7.5 | 67.3 | 113.8 | 93.1 | 0.2 | 0.3 | 9508.9 | 161.7 | 33776.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 7.5 | 51.4 | 114.2 | 93.0 | 0.2 | 0.3 | 9131.3 | 128.8 | 31758.5 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 7.6 | 49.8 | 112.7 | 92.7 | 0.2 | 0.3 | 9255.5 | 94.4 | 31148.1 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 7.3 | 54.0 | 102.3 | 94.1 | 0.2 | 0.3 | 8326.8 | 561.8 | 34305.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 7.1 | 15.0 | 71.9 | 93.9 | 0.2 | 0.3 | 2869.4 | 24.4 | 19801.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 7.3 | 23.8 | 69.0 | 93.7 | 0.2 | 0.3 | 3660.2 | 5.3 | 25396.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 6.8 | 22.0 | 10.5 | 94.3 | 0.2 | 0.3 | 3358.4 | 5.6 | 16004.9 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.7 | 35.0 | 9.0 | 94.2 | 0.2 | 0.3 | 3360.5 | 62.8 | 14862.7 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 6.7 | 54.6 | 9.9 | 94.2 | 0.3 | 0.3 | 4403.0 | 11.4 | 18664.4 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 7.2 | 50.6 | 8.9 | 93.8 | 0.3 | 0.3 | 3527.5 | 43.0 | 16353.5 | 6 | 0.91 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 7.7 | 88.0 | 35.2 | 93.2 | 0.2 | 0.3 | 6677.2 | 566.0 | 26775.8 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 7.1 | 70.9 | 11.3 | 94.5 | 0.2 | 0.3 | 2678.1 | 65.2 | 15055.6 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 7.2 | 64.6 | 16.8 | 94.3 | 0.2 | 0.3 | 2849.0 | 47.6 | 16913.0 | 8 | 0.79 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 7.0 | 25.2 | 8.2 | 94.1 | 0.2 | 0.4 | 2729.2 | 67.0 | 13878.4 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 18 | A | 7.3 | 22.1 | 10.2 | 93.3 | 0.6 | 0.5 | 2880.9 | 69.3 | 13896.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 6.5 | 25.2 | 8.9 | 94.7 | 0.2 | 0.3 | 2941.4 | 36.8 | 13169.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 6.4 | 15.8 | 8.7 | 95.0 | 0.2 | 0.3 | 2444.1 | 9.4 | 13151.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 6.4 | 16.3 | 7.7 | 95.0 | 0.2 | 0.3 | 2515.4 | 17.0 | 13661.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 6.4 | 16.1 | 7.5 | 94.7 | 0.2 | 0.3 | 2447.0 | 3.4 | 12006.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 6.3 | 14.5 | 7.3 | 95.0 | 0.2 | 0.3 | 2171.4 | 3.4 | 11508.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 6.3 | 15.3 | 8.0 | 95.3 | 0.2 | 0.3 | 2314.0 | 4.1 | 11471.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 6.3 | 13.7 | 7.4 | 95.1 | 0.2 | 0.3 | 2228.9 | 2.0 | 11675.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 77.5 min for 6 rounds (12.4 min per round); a fixed 3 rounds would take 37.2 min (-37.2 min saved).
