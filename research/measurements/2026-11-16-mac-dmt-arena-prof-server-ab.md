## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the workload of the first DM-T arena A/B, so the two compare (decode tokens per prompt as in the targets, default 200)
- Duration: 35-106 min (2-6 rounds of 17.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: 2026-11-16-mac-dmt-arena-ab.md (first A/B) scaled by the cold share of 2026-11-16-mac-dmt-profile.md):
  - decode-code: -30 to -5 %
  - decode-text: -30 to -5 %
  - decode@8000: -30 to -5 %
  - prefill@8000: -10 to +0 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1159-g6601e28b7c-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-1159-g6601e28b7c-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<repo>/work/profiles/dmt-12.bin, TONA_METAL_ARENA_GB=auto

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_ARENA_GB': 'auto', 'TONA_EXPERT_PROFILE': '<repo>/work/profiles/dmt-12.bin'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 6, stop: all metrics decided after round 4. All rounds are listed; 2 discarded, below.

Discarded (repeated) rounds:

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -1.75 % | -13.00 % | -10.82 % | -13.28 % | 793 |
| 2 | BA | -1.51 % | -16.90 % | -14.07 % | -18.55 % | 802 |

- AB: a system backup ran
- BA: a system backup ran

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 165.8 / 172.1 / 169.6 / 168.6 | 164.4 / 162.8 / 164.9 / 165.0 | 169.1 (165.8-172.1) | 164.6 (162.8-165.0) | -2.79 % [-5.22, -0.35] | **B slower** after round 4 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.2 / 25.2 / 25.2 / 25.3 | 21.0 / 20.1 / 21.4 / 21.0 | 25.2 (25.2-25.3) | 21.0 (20.1-21.4) | -17.28 % [-19.96, -14.60] | **B slower** after round 2 |
| decode-code: Decode tok/s (bench-set code) | 38.9 / 38.9 / 38.7 / 39.0 | 33.4 / 32.1 / 34.2 / 33.5 | 38.9 (38.7-39.0) | 33.5 (32.1-34.2) | -14.34 % [-17.37, -11.30] | **B slower** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 27.5 / 27.0 / 27.0 / 27.7 | 22.3 / 21.2 / 23.2 / 22.7 | 27.2 (27.0-27.7) | 22.5 (21.2-23.2) | -18.13 % [-21.97, -14.29] | **B slower** after round 2 |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -0.86 % | -16.80 % | -14.14 % | -18.91 % | 1320 |
| 2 | BA | -5.36 % | -20.21 % | -17.48 % | -21.48 % | 818 |
| 3 | AB | -2.79 % | -15.21 % | -11.63 % | -14.07 % | 796 |
| 4 | BA | -2.12 % | -16.89 % | -14.10 % | -18.05 % | 1297 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -2.79 % | -10 to +0 % | within the expectation |
| decode@8000 | -17.28 % | -30 to -5 % | within the expectation |
| decode-code | -14.34 % | -30 to -5 % | within the expectation |
| decode-text | -18.13 % | -30 to -5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.4 | 38.5 | 9.8 | 91.6 | 0.2 | 0.3 | 14801.6 | 4.9 | 16131.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.6 | 24.5 | 8.4 | 94.4 | 0.2 | 0.2 | 3341.6 | 0.5 | 11582.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.7 | 25.0 | 8.1 | 94.2 | 0.2 | 0.4 | 3135.9 | 1.6 | 11373.4 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 75.5 | 14.1 | 7.9 | 87.3 | 0.2 | 0.3 | 5374.5 | 0.2 | 30102.2 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 5 | B | 72.6 | 53.2 | 9.1 | 89.9 | 0.2 | 0.2 | 297.2 | 1.5 | 2107.1 | 13 | 0.16 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 6 | A | 7.0 | 54.6 | 10.4 | 91.2 | 0.2 | 0.3 | 19050.6 | 2.6 | 24644.9 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 510.8 | 2.0 |
| 7 | A | 6.3 | 68.8 | 10.6 | 93.7 | 0.2 | 0.3 | 5633.1 | 4.3 | 20674.5 | 5 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | B | 77.1 | 24.0 | 8.0 | 86.1 | 0.2 | 0.3 | 5709.9 | 0.2 | 28864.5 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 9 | B | 74.8 | 20.5 | 8.1 | 87.4 | 0.2 | 0.3 | 260.1 | 0.2 | 1291.9 | 16 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 10 | B | 75.7 | 17.2 | 7.1 | 85.5 | 0.2 | 0.2 | 343.6 | 0.4 | 229.8 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 11 | B | 76.4 | 14.3 | 6.8 | 84.9 | 0.2 | 0.2 | 82.6 | 0.2 | 0.0 | 19 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 12 | A | 6.2 | 39.4 | 10.0 | 92.6 | 0.2 | 0.2 | 17905.8 | 1.0 | 22065.9 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 13 | A | 5.8 | 38.7 | 9.3 | 94.7 | 0.2 | 0.4 | 3250.7 | 0.5 | 13150.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | A | 6.7 | 54.2 | 12.0 | 93.0 | 0.6 | 0.5 | 3981.4 | 0.6 | 13773.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | A | 5.8 | 17.1 | 8.2 | 94.4 | 0.2 | 0.3 | 2984.6 | 0.3 | 11397.5 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | B | 74.7 | 19.6 | 7.6 | 87.5 | 0.2 | 0.2 | 5615.5 | 0.3 | 30482.3 | 13 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 4.0 |
| 17 | B | 72.8 | 14.6 | 7.3 | 89.1 | 0.2 | 0.3 | 185.0 | 0.0 | 524.3 | 15 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 18 | B | 73.6 | 15.4 | 7.0 | 88.1 | 0.2 | 0.3 | 145.2 | 0.8 | 29.4 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 19 | B | 74.4 | 14.7 | 6.9 | 87.6 | 0.2 | 0.3 | 104.1 | 0.3 | 44.4 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 4.0 |
| 20 | A | 5.9 | 15.4 | 7.9 | 92.3 | 0.2 | 0.3 | 16499.8 | 0.1 | 16590.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 21 | A | 6.3 | 75.8 | 10.0 | 93.8 | 0.2 | 0.3 | 4122.1 | 1.3 | 16290.5 | 6 | 0.88 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 22 | B | 84.4 | 31.7 | 8.8 | 83.9 | 0.2 | 0.2 | 4932.7 | 0.0 | 31162.8 | 17 | 0.0 | 0.0 | - | 0.0 | 2 | 480.6 | 4.0 |
| 23 | B | 74.0 | 26.6 | 7.4 | 87.6 | 0.2 | 0.2 | 519.2 | 0.4 | 4184.4 | 14 | 0.0 | 0.0 | - | 0.0 | 2 | 0.3 | 4.0 |
| 24 | A | 6.6 | 23.2 | 10.5 | 90.6 | 0.2 | 0.2 | 18623.6 | 0.3 | 20839.9 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 25 | A | 6.5 | 73.0 | 25.5 | 93.9 | 0.2 | 0.3 | 5581.6 | 1.3 | 27082.1 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 100.3 min for 4 rounds (17.6 min per round); a fixed 3 rounds would take 52.9 min (-17.6 min saved).
