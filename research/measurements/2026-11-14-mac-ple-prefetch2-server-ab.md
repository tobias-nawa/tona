## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the set for decode, one 8000-token prompt for the prefill chunks read ahead (4 chunks of 2048) (decode tokens per prompt as in the targets, default 200)
- Duration: 25-50 min (2-4 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-ple-prefetch.md):
  - decode-code: +0 to +2 %
  - decode-text: +0 to +2 %
  - decode@8000: +0 to +2 %
  - prefill@8000: -1 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1005-g7f085e5015-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_STATS=1
  - B: build commit stand-2026-10-25-1005-g7f085e5015-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_PREFETCH=2, TONA_PLE_STATS=1

## Result

A: `work/build/mtl` env {'TONA_PLE_STATS': '1'}; B: `work/build/mtl` env {'TONA_PLE_STATS': '1', 'TONA_PLE_PREFETCH': '2'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 169.7 / 170.6 / 175.6 / 170.1 | 166.0 / 172.6 / 173.5 / 171.8 | 170.3 (169.7-175.6) | 172.2 (166.0-173.5) | -0.29 % [-2.38, +1.79] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.7 / 24.7 / 24.6 / 24.5 | 25.2 / 25.1 / 24.7 / 24.7 | 24.6 (24.5-24.7) | 24.9 (24.7-25.2) | +1.23 % [+0.23, +2.23] | **B faster** after round 2 |
| decode-code: Decode tok/s (bench-set code) | 38.1 / 38.1 / 38.1 / 38.1 | 38.7 / 38.7 / 38.2 / 38.2 | 38.1 (38.1-38.1) | 38.5 (38.2-38.7) | +0.92 % [-0.21, +2.04] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.4 / 26.4 / 26.5 / 26.4 | 27.2 / 25.8 / 26.6 / 26.6 | 26.4 (26.4-26.5) | 26.6 (25.8-27.2) | +0.47 % [-2.19, +3.13] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -2.17 % | +1.86 % | +1.57 % | +3.03 % | 754 |
| 2 | BA | +1.19 % | +1.64 % | +1.57 % | -2.27 % | 753 |
| 3 | AB | -1.21 % | +0.58 % | +0.26 % | +0.38 % | 749 |
| 4 | BA | +1.03 % | +0.85 % | +0.26 % | +0.76 % | 752 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -0.29 % | -1 to +3 % | within the expectation |
| decode@8000 | +1.23 % | +0 to +2 % | within the expectation |
| decode-code | +0.92 % | +0 to +2 % | within the expectation |
| decode-text | +0.47 % | +0 to +2 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.5 | 31.7 | 8.8 | 88.7 | 0.2 | 0.4 | 22667.3 | 3.4 | 13949.3 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.7 | 15.6 | 11.2 | 93.0 | 0.2 | 0.3 | 2573.7 | 1.2 | 13765.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.7 | 15.0 | 7.9 | 93.2 | 0.2 | 0.2 | 2497.9 | 1.6 | 13446.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.7 | 19.7 | 7.9 | 93.1 | 0.2 | 0.3 | 3587.6 | 3.0 | 10376.8 | 7 | 0.12 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.3 | 53.1 | 8.8 | 92.7 | 0.2 | 0.2 | 3882.2 | 1.4 | 13067.8 | 6 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 6.8 | 56.4 | 8.8 | 91.6 | 0.3 | 0.2 | 5272.0 | 152.4 | 16259.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 6.8 | 75.3 | 8.7 | 92.5 | 0.4 | 0.3 | 3339.6 | 131.1 | 9738.3 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 6.4 | 24.7 | 8.6 | 92.8 | 0.3 | 0.2 | 2412.2 | 14.0 | 14415.0 | 8 | 0.05 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.6 | 16.1 | 7.6 | 93.2 | 0.2 | 0.3 | 2325.7 | 7.6 | 12132.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 5.4 | 14.7 | 7.7 | 94.0 | 0.3 | 0.3 | 1730.3 | 4.1 | 12082.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 5.4 | 16.7 | 7.6 | 93.8 | 0.2 | 0.3 | 1915.3 | 10.7 | 11722.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.2 | 20.3 | 7.6 | 94.1 | 0.3 | 0.3 | 3175.0 | 29.4 | 6866.9 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.2 | 16.8 | 7.8 | 94.1 | 0.2 | 0.3 | 3111.7 | 20.4 | 6297.5 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.2 | 15.1 | 7.5 | 94.2 | 0.2 | 0.3 | 3027.2 | 5.2 | 6087.3 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 5.2 | 18.9 | 7.7 | 93.9 | 0.2 | 0.2 | 3195.3 | 13.1 | 6513.1 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.4 | 15.0 | 10.7 | 93.6 | 0.2 | 0.2 | 2310.1 | 113.4 | 12040.9 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 5.6 | 14.9 | 13.8 | 93.2 | 0.2 | 0.3 | 3300.0 | 13.6 | 12086.7 | 10 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 53.4 min for 4 rounds (12.5 min per round); a fixed 3 rounds would take 37.6 min (-12.5 min saved).
