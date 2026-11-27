## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the workload of the Coder A/B, whose load table showed DM-T's displacement, so the compressed memory per start compares (decode tokens per prompt as in the targets, default 200)
- Duration: 43-129 min (2-6 rounds of 21.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: estimate from 2026-11-15-mac-coder-arena.md (recall 69 % at 16 GB) and 68.8 % of DM-T's experts resident):
  - decode-code: -40 to -5 %
  - decode-text: -40 to -5 %
  - decode@8000: -40 to -5 %
  - prefill@8000: -30 to +0 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1156-g58513e336f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1156-g58513e336f-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_ARENA_GB=auto

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; 2 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -1.10 % | -46.58 % | -57.84 % | -48.71 % | 1484 |
| 2 | BA | -4.03 % | -46.81 % | -58.14 % | -49.08 % | 1101 |

- AB: a system backup ran
- BA: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 170.8 / 169.0 | 164.1 / 161.0 | 169.9 (169.0-170.8) | 162.6 (161.0-164.1) | -4.32 % [-5.51, -3.14] | **B slower** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.2 / 25.0 | 13.1 / 13.4 | 25.1 (25.0-25.2) | 13.3 (13.1-13.4) | -47.15 % [-48.86, -45.44] | **B slower** after round 2 |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 38.6 | 16.8 / 16.2 | 38.7 (38.6-38.7) | 16.5 (16.2-16.8) | -57.31 % [-58.78, -55.84] | **B slower** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 27.0 | 14.4 / 14.0 | 27.0 (27.0-27.0) | 14.2 (14.0-14.4) | -47.41 % [-48.90, -45.92] | **B slower** after round 2 |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -3.92 % | -48.09 % | -56.59 % | -46.67 % | 1079 |
| 2 | BA | -4.73 % | -46.22 % | -58.03 % | -48.15 % | 1508 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -4.32 % | -30 to +0 % | within the expectation |
| decode@8000 | -47.15 % | -40 to -5 % | **OUTSIDE the expectation: finding** |
| decode-code | -57.31 % | -40 to -5 % | **OUTSIDE the expectation: finding** |
| decode-text | -47.41 % | -40 to -5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.2 | 14.4 | 7.7 | 94.0 | 0.2 | 0.4 | 2521.6 | 0.8 | 10635.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 6.3 | 48.7 | 8.9 | 94.1 | 0.2 | 0.4 | 3321.9 | 2.3 | 13281.5 | 7 | 0.84 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 6.2 | 48.7 | 9.6 | 93.8 | 0.2 | 0.3 | 4002.9 | 6.3 | 19353.9 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 390.6 | 2.0 |
| 4 | B | 78.5 | 16.1 | 7.3 | 58.2 | 0.2 | 0.2 | 5197.9 | 5.2 | 30076.9 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 5 | B | 77.3 | 15.2 | 7.9 | 59.0 | 0.2 | 0.2 | 371.7 | 1.1 | 1847.5 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | A | 5.9 | 18.1 | 8.2 | 92.6 | 0.2 | 0.3 | 16984.6 | 0.5 | 16945.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 7 | A | 5.6 | 14.9 | 7.6 | 94.5 | 0.2 | 0.4 | 2593.4 | 0.5 | 9987.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | B | 76.9 | 17.4 | 7.9 | 59.1 | 0.2 | 0.3 | 5857.5 | 7.2 | 30937.6 | 13 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 9 | B | 76.6 | 13.5 | 6.8 | 59.6 | 0.2 | 0.2 | 116.2 | 0.5 | 52.8 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 10 | B | 76.9 | 13.8 | 6.8 | 58.6 | 0.2 | 0.2 | 121.9 | 0.7 | 70.7 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 11 | B | 77.4 | 14.0 | 6.8 | 58.5 | 0.2 | 0.2 | 185.9 | 0.6 | 53.6 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 12 | A | 5.8 | 15.6 | 7.7 | 92.6 | 0.2 | 0.3 | 16326.9 | 0.1 | 15790.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 13 | A | 5.8 | 19.6 | 7.8 | 94.3 | 0.2 | 0.4 | 2791.3 | 0.3 | 12276.1 | 7 | 0.13 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 77.0 | 20.9 | 10.1 | 59.5 | 0.2 | 0.1 | 5792.4 | 2.1 | 35620.2 | 10 | 0.0 | 0.0 | - | 0.0 | 2 | 420.5 | 4.0 |
| 15 | B | 77.3 | 20.6 | 7.1 | 58.4 | 0.2 | 0.2 | 1396.0 | 3.8 | 1162.9 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 16 | A | 6.1 | 17.7 | 8.2 | 91.7 | 0.2 | 0.3 | 17755.6 | 0.4 | 18501.5 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 17 | A | 5.9 | 47.1 | 9.5 | 94.4 | 0.2 | 0.2 | 3934.9 | 1.5 | 14687.0 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 89.3 min for 2 rounds (21.6 min per round); a fixed 3 rounds would take 64.7 min (+21.6 min saved).
