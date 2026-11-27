## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the workload of the DM-T arena A/Bs, so the CPU load compares (decode tokens per prompt as in the targets, default 200)
- Duration: 32-97 min (2-6 rounds of 16.1 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-16-mac-arena-stop-cpu.md and the screening of 2026-11-16 08:20 (a quarter of each stop asleep: 250-token requests within 1 % of no sleep, CPU 58 to 64 % against 84 to 93 %)):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1166-gda7793b2ac; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1
  - B: build commit stand-2026-10-25-1166-gda7793b2ac; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto, TONA_METAL_ARENA_GOVERN=1, TONA_METAL_STOP_SLEEP=0.25

## Result

A: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1'}; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin', 'TONA_METAL_ARENA_GOVERN': '1', 'TONA_METAL_STOP_SLEEP': '0.25'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; 3 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -1.58 % | -6.40 % | -5.11 % | -6.36 % | 1415 |
| 2 | AB | -3.36 % | -13.96 % | -9.63 % | -15.02 % | 1356 |
| 3 | AB | -1.18 % | -6.92 % | -7.49 % | -6.15 % | 1327 |

- AB: a system backup ran
- AB: a system backup ran
- AB: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 162.3 / 163.3 / 160.5 / 162.2 / 162.7 / 162.6 | 158.8 / 162.7 / 162.0 / 162.4 / 162.3 / 162.6 | 162.4 (160.5-163.3) | 162.3 (158.8-162.7) | -0.28 % [-1.51, +0.96] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 19.5 / 20.5 / 18.9 / 19.4 / 20.2 / 20.1 | 20.0 / 20.2 / 19.8 / 19.6 / 20.0 / 19.9 | 19.8 (18.9-20.5) | 20.0 (19.6-20.2) | +0.91 % [-1.82, +3.63] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 31.1 / 32.4 / 29.5 / 30.3 / 32.4 / 31.8 | 31.9 / 32.4 / 32.0 / 31.3 / 31.9 / 31.8 | 31.5 (29.5-32.4) | 31.9 (31.3-32.4) | +2.13 % [-1.85, +6.12] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 20.0 / 21.2 / 19.1 / 19.9 / 21.3 / 20.8 | 20.8 / 21.4 / 21.0 / 19.7 / 20.7 / 20.5 | 20.4 (19.1-21.3) | 20.8 (19.7-21.4) | +1.60 % [-3.62, +6.83] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -2.16 % | +2.41 % | +2.57 % | +4.00 % | 909 |
| 2 | BA | -0.39 % | -1.29 % | +0.00 % | +0.94 % | 887 |
| 3 | AB | +0.94 % | +4.85 % | +8.47 % | +9.95 % | 1309 |
| 4 | BA | +0.13 % | +1.34 % | +3.30 % | -1.01 % | 915 |
| 5 | AB | -0.22 % | -0.96 % | -1.54 % | -2.82 % | 894 |
| 6 | BA | +0.03 % | -0.90 % | +0.00 % | -1.44 % | 897 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -0.28 % | -1 to +1 % | within the expectation |
| decode@8000 | +0.91 % | -1 to +1 % | within the expectation |
| decode-code | +2.13 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-text | +1.60 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 73.5 | 23.4 | 12.9 | 86.7 | 0.6 | 1.2 | 3526.1 | 23.0 | 613.4 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | - | 4.0 |
| 2 | A | 74.0 | 17.5 | 11.9 | 87.1 | 0.5 | 1.1 | 496.1 | 145.0 | 4634.6 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 3 | A | 75.2 | 45.6 | 7.7 | 86.5 | 0.2 | 0.3 | 1067.6 | 67.1 | 4104.8 | 15 | 0.96 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 4 | B | 62.1 | 24.0 | 7.4 | 79.0 | 0.2 | 0.2 | 1203.0 | 5.6 | 2999.8 | 20 | 0.0 | 0.0 | - | 0.0 | 2 | 510.8 | 2.0 |
| 5 | B | 58.7 | 18.2 | 8.5 | 84.7 | 0.2 | 0.2 | 221.4 | 10.6 | 9311.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 6 | A | 77.8 | 18.6 | 7.1 | 83.7 | 0.2 | 0.3 | 227.3 | 7.3 | 427.1 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 7 | A | 77.9 | 38.3 | 49.7 | 83.7 | 0.2 | 0.2 | 121.6 | 13.8 | 638.4 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 8 | B | 60.0 | 80.6 | 7.9 | 82.8 | 0.2 | 0.2 | 764.6 | 5.1 | 1448.5 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 9 | B | 58.3 | 72.1 | 7.2 | 84.0 | 0.2 | 0.3 | 188.3 | 15.9 | 807.0 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 10 | B | 57.0 | 14.9 | 6.9 | 84.4 | 0.2 | 0.3 | 77.5 | 1.2 | 258.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 11 | B | 56.4 | 22.0 | 7.0 | 85.5 | 0.2 | 0.3 | 144.6 | 6.8 | 1221.8 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 12 | A | 75.1 | 14.4 | 7.2 | 85.2 | 0.2 | 0.2 | 111.4 | 28.4 | 853.9 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 13 | A | 75.2 | 15.6 | 6.9 | 85.2 | 0.2 | 0.3 | 74.4 | 1.4 | 392.5 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 14 | A | 76.1 | 18.8 | 8.8 | 85.4 | 0.5 | 0.3 | 233.4 | 1.7 | 509.6 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 15 | A | 75.6 | 15.5 | 6.9 | 85.1 | 0.2 | 0.3 | 81.0 | 0.9 | 183.0 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 16 | B | 57.3 | 111.3 | 7.3 | 85.4 | 0.2 | 0.3 | 645.7 | 4.2 | 1283.4 | 16 | 0.12 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 17 | B | 62.2 | 75.0 | 7.4 | 80.1 | 0.2 | 0.2 | 549.3 | 2.9 | 12673.4 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 450.6 | 2.0 |
| 18 | A | 77.6 | 15.1 | 6.8 | 54.2 | 0.2 | 0.2 | 305.6 | 4.1 | 0.0 | 43 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 2.0 |
| 19 | A | 78.8 | 15.5 | 7.1 | 81.5 | 0.2 | 0.2 | 309.6 | 4.4 | 3718.6 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 20 | B | 57.5 | 15.3 | 7.0 | 84.6 | 0.2 | 0.2 | 120.5 | 0.8 | 2256.5 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 21 | B | 57.3 | 14.6 | 14.8 | 84.6 | 0.2 | 0.2 | 479.2 | 0.9 | 4864.9 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 22 | B | 58.1 | 14.9 | 7.0 | 83.8 | 0.2 | 0.3 | 155.3 | 0.4 | 194.8 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 23 | B | 58.7 | 14.1 | 6.8 | 83.9 | 0.2 | 0.2 | 49.4 | 0.2 | 12.9 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 24 | A | 77.7 | 13.8 | 6.7 | 82.8 | 0.2 | 0.3 | 68.3 | 0.3 | 105.5 | 18 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 25 | A | 77.7 | 14.0 | 6.9 | 82.7 | 0.2 | 0.2 | 62.0 | 5.2 | 43.4 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 26 | A | 78.0 | 14.6 | 6.8 | 82.6 | 0.2 | 0.3 | 76.7 | 1.1 | 39.6 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 27 | A | 78.5 | 14.6 | 6.7 | 83.0 | 0.2 | 0.2 | 52.5 | 0.1 | 14.6 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 28 | B | 59.8 | 18.2 | 6.9 | 83.4 | 0.2 | 0.2 | 150.3 | 0.7 | 391.4 | 19 | 0.14 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 29 | B | 62.0 | 19.4 | 7.3 | 80.0 | 0.2 | 0.2 | 221.4 | 0.4 | 3934.9 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 390.6 | 2.0 |
| 30 | A | 76.6 | 16.2 | 7.0 | 84.1 | 0.2 | 0.3 | 121.7 | 0.2 | 3210.5 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 31 | A | 76.2 | 17.0 | 7.3 | 85.3 | 0.2 | 0.2 | 166.0 | 4.5 | 2013.8 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 32 | B | 56.5 | 14.8 | 7.1 | 84.6 | 0.2 | 0.2 | 87.5 | 0.1 | 323.1 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 33 | B | 56.7 | 16.4 | 7.2 | 84.4 | 0.2 | 0.3 | 112.2 | 0.6 | 521.2 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 34 | B | 57.0 | 15.2 | 6.9 | 84.3 | 0.2 | 0.2 | 69.9 | 2.6 | 96.2 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 35 | B | 57.1 | 16.7 | 7.8 | 84.0 | 0.2 | 0.3 | 448.8 | 0.4 | 868.3 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 36 | A | 76.6 | 14.6 | 6.9 | 84.9 | 0.2 | 0.3 | 75.4 | 0.2 | 75.6 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 37 | A | 76.1 | 15.6 | 7.2 | 84.0 | 0.2 | 0.3 | 3387.8 | 0.6 | 3451.7 | 10 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 168.7 min for 6 rounds (16.1 min per round); a fixed 3 rounds would take 48.4 min (-48.4 min saved).
