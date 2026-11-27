## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: 6 rounds as the early commit: gains of 0.2 to 5 % against a spread of about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 25-75 min (2-6 rounds of 12.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-pin-own.md):
  - decode-code: +0 to +0.5 %
  - decode-text: +0 to +0.5 %
  - decode@8000: +2 to +6 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1045-g132dfa4cbc-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1045-g132dfa4cbc-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_PIN_OWN=1

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_PIN_OWN': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 165.9 / 157.2 / 154.8 / 165.7 / 166.5 / 167.2 | 161.4 / 157.4 / 147.7 / 157.1 / 157.0 / 156.5 | 165.8 (154.8-167.2) | 157.0 (147.7-161.4) | -4.09 % [-6.78, -1.41] | **B slower** after round 4 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.0 / 25.3 / 25.5 / 24.9 / 25.0 / 25.0 | 25.1 / 25.4 / 25.7 / 25.8 / 25.2 / 25.2 | 25.0 (24.9-25.5) | 25.3 (25.1-25.8) | +1.09 % [-0.38, +2.55] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 39.5 / 39.4 / 38.7 / 38.8 / 38.7 | 38.7 / 39.2 / 39.4 / 39.6 / 38.8 / 38.7 | 38.8 (38.7-39.5) | 39.0 (38.7-39.6) | +0.26 % [-1.00, +1.52] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 27.7 / 27.7 / 27.0 / 27.0 / 27.0 | 27.0 / 27.6 / 27.6 / 27.7 / 27.1 / 27.0 | 27.0 (27.0-27.7) | 27.4 (27.0-27.7) | +0.37 % [-0.96, +1.70] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -2.73 % | +0.30 % | +0.00 % | +0.00 % | 753 |
| 2 | BA | +0.08 % | +0.33 % | -0.76 % | -0.36 % | 751 |
| 3 | AB | -4.58 % | +0.74 % | +0.00 % | -0.36 % | 758 |
| 4 | BA | -5.21 % | +3.60 % | +2.33 % | +2.59 % | 749 |
| 5 | AB | -5.70 % | +0.85 % | +0.00 % | +0.37 % | 756 |
| 6 | BA | -6.41 % | +0.72 % | +0.00 % | +0.00 % | 754 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -4.09 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +1.09 % | +2 to +6 % | **OUTSIDE the expectation: finding** |
| decode-code | +0.26 % | +0 to +0.5 % | within the expectation |
| decode-text | +0.37 % | +0 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.0 | 21.8 | 11.6 | 92.4 | 0.6 | 1.4 | 3318.7 | 10.3 | 16885.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.3 | 14.7 | 7.8 | 94.3 | 0.2 | 0.2 | 2556.2 | 2.8 | 12789.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.2 | 15.1 | 7.5 | 94.5 | 0.2 | 0.3 | 2480.5 | 3.9 | 12100.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 4.7 | 15.6 | 8.0 | 93.2 | 0.2 | 0.3 | 3237.8 | 4.1 | 4990.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 4.5 | 17.3 | 7.9 | 93.7 | 0.2 | 0.4 | 3080.0 | 7.5 | 5070.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 4.8 | 20.1 | 8.0 | 93.3 | 0.3 | 0.3 | 3410.5 | 12.6 | 5193.8 | 7 | 0.17 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.3 | 50.0 | 8.2 | 92.9 | 0.2 | 0.3 | 3297.5 | 4.2 | 5870.8 | 6 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 6.8 | 88.5 | 45.0 | 92.4 | 0.2 | 0.3 | 6692.1 | 1251.7 | 31321.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 6.7 | 76.5 | 108.8 | 92.1 | 0.2 | 0.3 | 8048.4 | 360.4 | 25193.4 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 7.0 | 108.6 | 108.4 | 91.9 | 0.2 | 0.3 | 10334.7 | 132.2 | 35242.1 | 5 | 0.45 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.9 | 51.8 | 114.4 | 91.8 | 0.3 | 0.3 | 10924.0 | 211.7 | 33106.5 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.9 | 61.0 | 112.6 | 91.0 | 0.2 | 0.3 | 13171.6 | 97.1 | 28825.0 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.7 | 63.4 | 109.8 | 91.2 | 0.2 | 0.3 | 13011.1 | 185.6 | 26344.3 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.6 | 51.0 | 102.9 | 91.4 | 0.2 | 0.2 | 10476.1 | 564.9 | 23855.5 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 5.2 | 17.1 | 71.9 | 92.8 | 0.2 | 0.3 | 4001.6 | 39.7 | 12299.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 6.1 | 17.7 | 68.9 | 93.3 | 0.2 | 0.3 | 3196.1 | 14.3 | 20904.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 5.4 | 20.5 | 11.7 | 94.0 | 0.3 | 0.3 | 3350.5 | 5.4 | 14491.1 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 6.4 | 31.0 | 11.8 | 92.6 | 0.6 | 0.4 | 4082.2 | 29.5 | 15258.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 5.4 | 34.6 | 9.1 | 93.8 | 0.5 | 0.3 | 3460.2 | 74.5 | 15131.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 4.8 | 17.3 | 7.9 | 93.0 | 0.2 | 0.2 | 3299.8 | 1.5 | 4201.3 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 4.8 | 26.9 | 8.6 | 93.1 | 0.2 | 0.2 | 3782.8 | 4.4 | 5512.8 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 4.9 | 29.1 | 9.2 | 92.8 | 0.2 | 0.4 | 4720.3 | 2.4 | 7508.1 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 4.8 | 24.6 | 8.2 | 93.0 | 0.2 | 0.3 | 3901.1 | 1.2 | 5519.1 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 5.4 | 40.9 | 9.1 | 94.3 | 0.2 | 0.3 | 3583.2 | 8.6 | 15685.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 5.4 | 20.6 | 8.5 | 94.3 | 0.2 | 0.2 | 3059.7 | 14.7 | 14243.5 | 6 | 0.06 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 78.5 min for 6 rounds (12.6 min per round); a fixed 3 rounds would take 37.7 min (-37.7 min saved).
