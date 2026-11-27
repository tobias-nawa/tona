## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl-pick2a` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: coordinator: 6 rounds for the default decision; the set for decode, one 8000-token request with an answer to compare (decode tokens per prompt as in the targets, default 200)
- Duration: 25-75 min (2-6 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-mtp-gpu-pick.md):
  - decode-code: +0.3 to +1.2 %
  - decode-text: +0.3 to +1.2 %
  - decode@8000: +0.3 to +1.2 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1023-g5e672218c1-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DRAFT_LOG=work/draft-a3.log, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1023-g5e672218c1-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DRAFT_LOG=work/draft-b3.log, TONA_GPU_GAPS=500, TONA_METAL_GPU_PICK=1

## Result

A: `work/build/mtl-pick2a` env {'TONA_GPU_GAPS': '500', 'TONA_DRAFT_LOG': 'work/draft-a3.log'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_DRAFT_LOG': 'work/draft-b3.log', 'TONA_METAL_GPU_PICK': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 169.1 / 170.1 / 160.2 / 175.6 / 165.5 / 169.8 | 173.4 / 169.7 / 170.2 / 176.0 / 168.1 / 169.9 | 169.5 (160.2-175.6) | 170.0 (168.1-176.0) | +1.74 % [-1.01, +4.48] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.7 / 24.9 / 25.2 / 25.2 / 24.8 / 24.8 | 24.9 / 24.8 / 25.3 / 25.0 / 25.2 / 24.9 | 24.9 (24.7-25.2) | 24.9 (24.8-25.3) | +0.22 % [-0.84, +1.28] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.1 / 38.2 / 38.7 / 38.9 / 38.5 / 38.1 | 38.4 / 38.4 / 39.0 / 38.4 / 38.4 / 38.4 | 38.4 (38.1-38.9) | 38.4 (38.4-39.0) | +0.22 % [-0.83, +1.27] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.6 / 26.7 / 26.9 / 27.1 / 26.6 / 26.6 | 26.9 / 26.8 / 27.3 / 26.7 / 26.8 / 26.8 | 26.6 (26.6-27.1) | 26.8 (26.7-27.3) | +0.50 % [-0.75, +1.75] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +2.53 % | +0.60 % | +0.79 % | +1.13 % | 752 |
| 2 | BA | -0.22 % | -0.60 % | +0.52 % | +0.37 % | 750 |
| 3 | AB | +6.26 % | +0.40 % | +0.78 % | +1.49 % | 747 |
| 4 | BA | +0.24 % | -0.95 % | -1.29 % | -1.48 % | 745 |
| 5 | AB | +1.55 % | +1.38 % | -0.26 % | +0.75 % | 754 |
| 6 | BA | +0.05 % | +0.51 % | +0.79 % | +0.75 % | 751 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +1.74 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +0.22 % | +0.3 to +1.2 % | **OUTSIDE the expectation: finding** |
| decode-code | +0.22 % | +0.3 to +1.2 % | **OUTSIDE the expectation: finding** |
| decode-text | +0.50 % | +0.3 to +1.2 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.2 | 37.2 | 9.6 | 92.3 | 0.2 | 0.4 | 3669.5 | 57.5 | 13171.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.7 | 17.8 | 8.3 | 93.3 | 0.2 | 0.3 | 2975.4 | 2.4 | 13427.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.6 | 14.8 | 7.7 | 93.5 | 0.2 | 0.3 | 2383.1 | 0.9 | 12789.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.1 | 21.8 | 14.8 | 94.0 | 0.2 | 0.3 | 3279.9 | 61.8 | 16305.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.0 | 17.0 | 10.9 | 94.5 | 0.2 | 0.3 | 2369.1 | 1.1 | 13868.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 4.9 | 18.0 | 13.7 | 94.4 | 0.2 | 0.3 | 2379.5 | 1.4 | 13569.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.1 | 15.0 | 7.4 | 94.1 | 0.3 | 0.3 | 2367.9 | 0.9 | 11821.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.5 | 17.1 | 7.9 | 94.1 | 0.2 | 0.3 | 2209.0 | 6.3 | 12466.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.7 | 23.1 | 8.0 | 93.3 | 0.3 | 0.3 | 2624.1 | 1.9 | 13912.3 | 8 | 0.2 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 6.3 | 60.0 | 8.2 | 92.9 | 0.2 | 0.3 | 2711.0 | 1.7 | 15314.9 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.6 | 49.6 | 8.6 | 91.8 | 0.2 | 0.3 | 4433.0 | 102.8 | 15335.5 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.4 | 52.7 | 8.0 | 94.4 | 0.2 | 0.2 | 2394.5 | 28.5 | 12511.8 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.8 | 41.1 | 8.2 | 93.7 | 0.2 | 0.3 | 2396.0 | 45.4 | 13612.9 | 9 | 0.15 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 14 | B | 5.0 | 17.1 | 8.0 | 94.4 | 0.2 | 0.3 | 2256.9 | 194.1 | 12915.1 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 4.9 | 16.2 | 7.5 | 95.0 | 0.2 | 0.3 | 1863.6 | 8.2 | 12463.2 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.3 | 19.1 | 7.9 | 94.4 | 0.2 | 0.3 | 1760.4 | 39.3 | 11896.7 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 6.0 | 108.6 | 60.8 | 93.9 | 0.3 | 0.2 | 5602.4 | 205.6 | 24808.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 5.6 | 31.1 | 10.6 | 93.5 | 0.2 | 0.3 | 5535.8 | 16.2 | 13459.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 5.9 | 28.7 | 39.4 | 92.9 | 0.2 | 0.3 | 2962.1 | 35.6 | 14828.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 5.3 | 17.9 | 8.2 | 93.5 | 0.2 | 0.3 | 2784.6 | 5.2 | 12904.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 5.2 | 16.5 | 8.4 | 93.9 | 0.2 | 0.3 | 2898.8 | 2.8 | 12661.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 5.0 | 18.3 | 7.6 | 94.4 | 0.2 | 0.2 | 2416.9 | 3.2 | 12530.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 5.1 | 27.2 | 7.8 | 94.2 | 0.2 | 0.3 | 2329.2 | 3.5 | 13433.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 5.5 | 16.0 | 7.5 | 93.8 | 0.2 | 0.2 | 2429.9 | 3.0 | 12895.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 5.6 | 13.6 | 7.6 | 93.6 | 0.2 | 0.2 | 2320.8 | 1.1 | 12036.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 78.1 min for 6 rounds (12.5 min per round); a fixed 3 rounds would take 37.5 min (-37.5 min saved).
