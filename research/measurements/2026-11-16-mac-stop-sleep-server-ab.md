## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the workload of the DM-T arena A/Bs, so the CPU load compares (decode tokens per prompt as in the targets, default 200)
- Duration: 35-105 min (2-6 rounds of 17.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-16-mac-arena-stop-cpu.md (half of each stop asleep, approaching the arrival from below)):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1162-g77b2b62ec3; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1
  - B: build commit stand-2026-10-25-1162-g77b2b62ec3; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1, TONA_METAL_STOP_SLEEP=0.5

## Result

A: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1'}; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_STOP_SLEEP': '0.5'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; 2 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | BA | +0.80 % | +1.79 % | +3.75 % | +5.38 % | 1197 |
| 2 | BA | -0.12 % | -3.02 % | -0.66 % | +1.56 % | 932 |

- BA: a system backup ran
- BA: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 162.0 / 162.5 / 162.3 / 162.6 / 162.0 / 157.6 | 161.8 / 162.2 / 161.7 / 161.7 / 161.7 / 161.3 | 162.1 (157.6-162.6) | 161.7 (161.3-162.2) | +0.15 % [-1.14, +1.45] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 19.3 / 19.8 / 19.4 / 19.5 / 19.4 / 17.9 | 19.3 / 19.3 / 18.8 / 19.1 / 18.9 / 19.0 | 19.4 (17.9-19.8) | 19.1 (18.8-19.3) | -0.70 % [-4.62, +3.22] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 30.7 / 31.5 / 31.1 / 31.3 / 31.1 / 28.9 | 31.0 / 31.1 / 29.8 / 30.5 / 30.3 / 31.0 | 31.1 (28.9-31.5) | 30.8 (29.8-31.1) | -0.39 % [-4.96, +4.18] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 20.1 / 20.6 / 19.8 / 20.4 / 19.8 / 18.4 | 20.0 / 20.0 / 19.3 / 20.1 / 19.5 / 20.1 | 20.0 (18.4-20.6) | 20.0 (19.3-20.1) | +0.05 % [-5.02, +5.12] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.15 % | +0.11 % | +0.98 % | -0.50 % | 1315 |
| 2 | BA | -0.23 % | -2.29 % | -1.27 % | -2.91 % | 905 |
| 3 | AB | -0.35 % | -3.02 % | -4.18 % | -2.53 % | 924 |
| 4 | BA | -0.56 % | -2.31 % | -2.56 % | -1.47 % | 921 |
| 5 | AB | -0.13 % | -2.83 % | -2.57 % | -1.52 % | 918 |
| 6 | BA | +2.35 % | +6.13 % | +7.27 % | +9.24 % | 1341 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.15 % | -1 to +1 % | within the expectation |
| decode@8000 | -0.70 % | -1 to +1 % | within the expectation |
| decode-code | -0.39 % | -1 to +1 % | within the expectation |
| decode-text | +0.05 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 76.0 | 49.0 | 8.0 | 85.2 | 0.2 | 0.3 | 3244.2 | 1.2 | 2102.9 | 14 | 0.92 | 0.0 | - | 0.0 | 2 | - | 4.0 |
| 2 | A | 79.9 | 14.4 | 6.8 | 81.2 | 0.2 | 0.2 | 437.3 | 0.1 | 1545.0 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 390.7 | 2.0 |
| 3 | A | 78.2 | 43.1 | 8.5 | 83.2 | 0.2 | 0.2 | 320.9 | 0.2 | 2777.1 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 4 | B | 43.8 | 16.9 | 7.2 | 83.5 | 0.2 | 0.2 | 88.6 | 0.1 | 933.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 5 | B | 43.0 | 15.6 | 7.2 | 84.0 | 0.2 | 0.3 | 125.2 | 0.1 | 837.5 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 6 | B | 42.2 | 14.1 | 7.3 | 83.5 | 0.2 | 0.2 | 98.0 | 0.0 | 404.8 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 7 | B | 43.1 | 13.8 | 7.8 | 84.2 | 0.2 | 0.2 | 465.9 | 0.1 | 863.5 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 8 | A | 76.3 | 13.1 | 6.9 | 84.1 | 0.2 | 0.2 | 84.2 | 0.0 | 35.7 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 9 | A | 77.7 | 14.0 | 6.7 | 84.6 | 0.2 | 0.2 | 35.8 | 0.1 | 0.0 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 10 | A | 78.1 | 14.6 | 6.8 | 84.2 | 0.3 | 0.2 | 118.9 | 1.6 | 45.8 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 11 | A | 78.2 | 15.0 | 6.8 | 83.7 | 0.2 | 0.2 | 74.9 | 0.1 | 36.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 12 | B | 43.7 | 14.2 | 7.0 | 82.5 | 0.2 | 0.3 | 63.0 | 0.0 | 31.3 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 13 | B | 43.9 | 18.0 | 9.0 | 81.4 | 0.4 | 0.3 | 200.1 | 0.0 | 297.0 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 14 | B | 43.8 | 14.5 | 7.0 | 82.3 | 0.2 | 0.3 | 103.2 | 0.1 | 161.5 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 15 | B | 45.6 | 48.5 | 7.5 | 82.7 | 0.2 | 0.3 | 407.1 | 0.2 | 2451.2 | 16 | 0.88 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 16 | A | 81.2 | 15.6 | 6.7 | 79.2 | 0.2 | 0.2 | 80.5 | 0.1 | 1620.7 | 21 | 0.0 | 0.0 | - | 0.0 | 2 | 240.5 | 2.0 |
| 17 | A | 79.7 | 20.6 | 7.0 | 81.1 | 0.2 | 0.2 | 289.4 | 0.3 | 1892.0 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 18 | B | 43.8 | 17.5 | 7.2 | 82.2 | 0.2 | 0.3 | 215.6 | 0.0 | 6913.4 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 19 | B | 42.9 | 18.5 | 7.8 | 82.7 | 0.2 | 0.3 | 176.4 | 0.2 | 1799.3 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 20 | A | 77.1 | 23.9 | 14.0 | 81.6 | 0.8 | 1.2 | 564.6 | 0.1 | 1570.3 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 21 | A | 78.1 | 19.7 | 11.5 | 83.7 | 0.4 | 0.7 | 332.4 | 2.7 | 592.7 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 22 | A | 78.1 | 15.8 | 8.8 | 83.2 | 0.4 | 0.9 | 542.3 | 0.0 | 2128.4 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 23 | A | 78.4 | 15.3 | 6.6 | 83.6 | 0.2 | 0.2 | 78.5 | 0.1 | 27.9 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 24 | B | 43.8 | 17.0 | 8.0 | 83.0 | 0.4 | 0.3 | 108.2 | 0.1 | 49.9 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 25 | B | 44.1 | 15.3 | 7.3 | 82.9 | 0.2 | 0.2 | 76.3 | 0.5 | 265.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 26 | B | 44.8 | 24.0 | 7.9 | 82.6 | 0.2 | 0.2 | 297.8 | 0.3 | 1266.5 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 27 | B | 44.6 | 15.3 | 7.1 | 82.8 | 0.2 | 0.3 | 108.2 | 0.1 | 265.9 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 28 | A | 78.4 | 14.2 | 6.8 | 82.9 | 0.2 | 0.2 | 92.8 | 0.3 | 291.9 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 29 | A | 78.0 | 27.6 | 7.2 | 82.2 | 0.2 | 0.3 | 153.0 | 0.0 | 1180.0 | 17 | 0.43 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 30 | B | 47.7 | 19.3 | 7.4 | 80.4 | 0.2 | 0.2 | 224.0 | 0.2 | 6624.5 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 390.6 | 2.0 |
| 31 | B | 43.3 | 16.2 | 7.2 | 83.4 | 0.2 | 0.3 | 163.5 | 0.1 | 3478.0 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 32 | A | 77.0 | 30.3 | 9.9 | 82.3 | 0.2 | 0.3 | 3573.0 | 856.9 | 5066.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 33 | A | 78.8 | 37.1 | 10.4 | 79.7 | 0.2 | 0.2 | 1409.1 | 0.3 | 3067.8 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 144.6 min for 6 rounds (17.6 min per round); a fixed 3 rounds would take 52.7 min (-52.7 min saved).
