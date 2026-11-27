## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the workload of the DM-T arena A/Bs; both arms plan the arena up to the working set, so the arms differ only in the purgeable marks (decode tokens per prompt as in the targets, default 200)
- Duration: 30-91 min (2-6 rounds of 15.2 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/53-mac-purgeable-arena.md (calls of 0.3 us per block, tona-check wait per stop 10.5 ms with 180 s warm window)):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1169-gf904f86a8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_CEILING=1, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1
  - B: build commit stand-2026-10-25-1169-gf904f86a8f; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1, TONA_METAL_ARENA_PURGEABLE=1

## Result

A: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_CEILING': '1'}; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_ARENA_PURGEABLE': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; 3 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +9.57 % | +1.82 % | +0.29 % | +1.30 % | 1534 |
| 2 | AB | -3.85 % | +1.57 % | +0.58 % | +0.43 % | 1466 |
| 3 | BA | -0.44 % | -5.13 % | -2.61 % | +0.43 % | 1220 |

- AB: a system backup ran
- AB: a system backup ran
- BA: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 156.7 / 151.6 / 153.0 / 155.1 / 153.0 / 171.1 | 148.2 / 148.9 / 153.5 / 152.6 / 144.8 / 146.8 | 154.1 (151.6-171.1) | 148.5 (144.8-153.5) | -4.68 % [-10.44, +1.08] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 19.8 / 19.5 / 19.7 / 19.4 / 19.8 / 12.9 | 19.6 / 19.8 / 19.8 / 19.5 / 17.3 / 19.6 | 19.6 (12.9-19.8) | 19.6 (17.3-19.8) | +6.93 % [-17.91, +31.77] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 34.7 / 34.8 / 34.8 / 34.7 / 34.8 / 22.4 | 34.7 / 34.8 / 34.8 / 34.7 / 34.1 / 33.5 | 34.8 (22.4-34.8) | 34.7 (33.5-34.8) | +7.92 % [-14.57, +30.42] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 23.4 / 23.4 / 23.3 / 23.4 / 23.3 / 12.1 | 23.3 / 23.5 / 23.5 / 23.3 / 23.4 / 22.4 | 23.4 (12.1-23.4) | 23.4 (22.4-23.5) | +14.33 % [-23.89, +52.55] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -5.41 % | -0.85 % | +0.00 % | -0.43 % | 850 |
| 2 | BA | -1.81 % | +1.70 % | +0.00 % | +0.43 % | 850 |
| 3 | AB | +0.35 % | +0.53 % | +0.00 % | +0.86 % | 850 |
| 4 | BA | -1.58 % | +0.82 % | +0.00 % | -0.43 % | 848 |
| 5 | AB | -5.36 % | -12.32 % | -2.01 % | +0.43 % | 856 |
| 6 | BA | -14.25 % | +51.70 % | +49.55 % | +85.12 % | 1221 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -4.68 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +6.93 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-code | +7.92 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-text | +14.33 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 51.6 | 38.1 | 15.7 | 85.5 | 0.7 | 1.3 | 6297.0 | 126.3 | 22877.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | - | 4.1 |
| 2 | A | 52.5 | 81.6 | 11.7 | 87.6 | 0.2 | 0.3 | 6313.0 | 105.1 | 30385.7 | 4 | 0.87 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 3 | A | 53.6 | 54.0 | 10.9 | 84.5 | 0.2 | 0.2 | 5817.8 | 44.5 | 45256.0 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 390.7 | 4.0 |
| 4 | B | 54.6 | 50.2 | 9.9 | 42.7 | 0.2 | 0.2 | 7008.7 | 155.3 | 89615.8 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.4 | 4.0 |
| 5 | B | 52.1 | 48.5 | 10.1 | 87.3 | 0.2 | 0.2 | 5148.1 | 29.8 | 23506.6 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | A | 52.0 | 48.5 | 10.4 | 88.0 | 0.2 | 0.3 | 4534.4 | 24.9 | 23466.0 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 7 | A | 52.1 | 55.6 | 10.5 | 88.3 | 0.2 | 0.2 | 5618.7 | 40.3 | 24634.0 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 8 | B | 52.1 | 46.2 | 9.5 | 87.8 | 0.2 | 0.2 | 4756.8 | 14.1 | 22999.5 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 9 | B | 52.3 | 75.0 | 11.5 | 86.8 | 0.2 | 0.2 | 7136.4 | 31.7 | 33710.8 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 10 | B | 52.1 | 41.4 | 9.4 | 87.3 | 0.2 | 0.3 | 4372.3 | 21.7 | 20975.5 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 11 | B | 51.8 | 34.8 | 8.5 | 87.6 | 0.2 | 0.2 | 3667.0 | 7.3 | 17007.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 12 | A | 52.1 | 42.1 | 9.8 | 88.5 | 0.2 | 0.2 | 4906.6 | 25.2 | 23058.3 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 13 | A | 52.1 | 63.5 | 10.5 | 87.5 | 0.2 | 0.2 | 5603.9 | 24.0 | 25524.1 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 14 | A | 52.3 | 70.7 | 10.9 | 87.5 | 0.2 | 0.3 | 6185.4 | 30.4 | 30088.4 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 15 | A | 52.5 | 62.3 | 17.0 | 87.4 | 0.2 | 0.2 | 6142.6 | 216.6 | 33612.2 | 4 | 0.05 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 16 | B | 52.9 | 54.6 | 9.9 | 86.0 | 0.2 | 0.2 | 4362.5 | 18.3 | 24888.5 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 600.9 | 4.0 |
| 17 | B | 52.1 | 53.0 | 9.7 | 86.9 | 0.2 | 0.2 | 5377.2 | 8.6 | 26702.9 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 18 | A | 52.0 | 53.8 | 11.9 | 87.7 | 0.2 | 0.3 | 5906.1 | 23.5 | 27308.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 19 | A | 52.2 | 58.2 | 10.0 | 87.7 | 0.2 | 0.2 | 5407.9 | 9.1 | 26260.3 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 20 | B | 51.7 | 22.0 | 8.3 | 87.4 | 0.2 | 0.3 | 3523.5 | 1.7 | 16611.8 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 21 | B | 51.8 | 44.0 | 9.9 | 88.2 | 0.2 | 0.2 | 4522.7 | 4.7 | 19118.8 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 22 | B | 52.1 | 21.9 | 8.5 | 87.6 | 0.2 | 0.3 | 3944.7 | 2.1 | 19081.0 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 23 | B | 52.1 | 41.9 | 9.0 | 87.7 | 0.2 | 0.3 | 4079.5 | 4.2 | 18506.6 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 24 | A | 51.9 | 26.8 | 9.1 | 88.5 | 0.2 | 0.3 | 4001.9 | 1.9 | 18740.0 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 25 | A | 52.1 | 38.2 | 16.1 | 87.9 | 0.2 | 0.2 | 4806.6 | 4.0 | 28965.3 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 26 | A | 52.1 | 66.8 | 10.8 | 88.0 | 0.2 | 0.3 | 5443.8 | 5.3 | 25164.8 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 27 | A | 52.0 | 39.2 | 9.8 | 87.9 | 0.2 | 0.3 | 5170.0 | 2.9 | 23541.7 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 28 | B | 52.4 | 67.8 | 10.8 | 86.8 | 0.2 | 0.3 | 5777.7 | 6.8 | 28575.0 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 29 | B | 53.1 | 60.7 | 10.7 | 85.6 | 0.6 | 0.2 | 5942.7 | 62.3 | 29514.6 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 30 | B | 55.4 | 79.7 | 13.9 | 85.4 | 0.7 | 0.3 | 6554.6 | 115.0 | 32118.7 | 4 | 0.96 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 31 | B | 53.3 | 43.0 | 9.4 | 85.8 | 2.2 | 0.2 | 3430.0 | 11.6 | 25600.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 360.7 | 4.0 |
| 32 | A | 52.6 | 46.4 | 9.3 | 87.6 | 0.9 | 0.3 | 4379.1 | 7.5 | 21085.4 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 33 | A | 52.6 | 48.7 | 9.9 | 88.2 | 0.4 | 0.2 | 4984.1 | 2.8 | 23979.3 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 34 | B | 53.0 | 74.9 | 11.1 | 85.4 | 1.4 | 0.2 | 5991.3 | 16.3 | 28054.2 | 4 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 35 | B | 54.0 | 54.1 | 9.5 | 83.8 | 2.1 | 0.2 | 5056.5 | 8.8 | 22845.4 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 36 | A | 53.8 | 27.7 | 7.3 | 65.2 | 1.3 | 0.2 | 3472.8 | 8.0 | 0.0 | 23 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 2.0 |
| 37 | A | 51.1 | 21.3 | 6.8 | 76.3 | 0.8 | 0.2 | 194.6 | 3.4 | 3865.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 165.2 min for 6 rounds (15.2 min per round); a fixed 3 rounds would take 45.6 min (-45.6 min saved).
