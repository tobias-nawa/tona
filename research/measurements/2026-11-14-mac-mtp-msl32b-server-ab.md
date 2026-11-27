## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl-pick2a` against B `work/build/mtl`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: repetition of run 4 (decode@8000 -0.87 % against an unchanged GPU clock); the set for decode, one 8000-token request with an answer to compare (decode tokens per prompt as in the targets, default 200)
- Duration: 25-50 min (2-4 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-mtp-gpu-pick.md):
  - decode-code: -0.5 to +0.5 %
  - decode-text: -0.5 to +0.5 %
  - decode@8000: -0.5 to +0.5 %
  - prefill@8000: -0.5 to +0.5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1024-g9f7af719a1-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1024-g9f7af719a1-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_GPU_PICK=0

## Result

A: `work/build/mtl-pick2a` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_GPU_PICK': '0'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 168.7 / 172.0 / 170.5 / 168.9 | 169.4 / 168.1 / 172.1 / 170.9 | 169.7 (168.7-172.0) | 170.1 (168.1-172.1) | +0.08 % [-1.91, +2.06] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.8 / 25.0 / 24.7 / 24.8 | 25.3 / 24.8 / 24.7 / 24.7 | 24.8 (24.7-25.0) | 24.8 (24.7-25.3) | +0.19 % [-1.45, +1.84] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.3 / 38.2 / 38.2 / 38.2 | 38.8 / 38.2 / 38.2 / 38.3 | 38.2 (38.2-38.3) | 38.2 (38.2-38.8) | +0.39 % [-0.61, +1.39] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.9 / 26.6 / 26.6 / 26.6 | 27.1 / 26.6 / 26.6 / 26.6 | 26.6 (26.6-26.9) | 26.6 (26.6-27.1) | +0.19 % [-0.63, +1.00] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.46 % | +2.03 % | +1.31 % | +0.74 % | 750 |
| 2 | BA | -2.25 % | -0.80 % | +0.00 % | +0.00 % | 754 |
| 3 | AB | +0.91 % | -0.05 % | +0.00 % | +0.00 % | 753 |
| 4 | BA | +1.18 % | -0.41 % | +0.26 % | +0.00 % | 753 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.08 % | -0.5 to +0.5 % | within the expectation |
| decode@8000 | +0.19 % | -0.5 to +0.5 % | within the expectation |
| decode-code | +0.39 % | -0.5 to +0.5 % | within the expectation |
| decode-text | +0.19 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 7.2 | 46.6 | 13.1 | 89.7 | 2.2 | 0.6 | 3809.3 | 15.3 | 15651.8 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.8 | 17.3 | 7.8 | 93.6 | 0.3 | 0.2 | 2407.2 | 9.2 | 13330.8 | 7 | 0.17 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.9 | 18.3 | 7.7 | 93.5 | 0.3 | 0.2 | 2446.5 | 3.9 | 13115.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.8 | 17.0 | 7.9 | 93.6 | 0.3 | 0.3 | 2267.8 | 1.9 | 12309.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.3 | 23.0 | 8.2 | 93.0 | 0.2 | 0.3 | 2455.0 | 1.7 | 14312.5 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.7 | 14.7 | 7.7 | 93.2 | 0.2 | 0.3 | 2642.5 | 1.7 | 13110.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 7 | B | 5.6 | 31.0 | 7.8 | 93.3 | 0.2 | 0.4 | 2947.3 | 4.1 | 11823.1 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.6 | 15.8 | 7.4 | 93.6 | 0.2 | 0.4 | 2242.4 | 3.2 | 11826.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.4 | 14.8 | 7.7 | 94.0 | 0.2 | 0.3 | 2363.3 | 1.6 | 12688.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 5.7 | 17.7 | 7.7 | 93.4 | 0.2 | 0.3 | 2618.5 | 6.3 | 12993.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 5.6 | 14.5 | 7.7 | 93.8 | 0.2 | 0.2 | 2368.7 | 1.5 | 12847.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.6 | 15.3 | 7.7 | 93.7 | 0.2 | 0.3 | 2343.5 | 3.2 | 12655.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.5 | 14.2 | 7.6 | 93.9 | 0.2 | 0.2 | 2346.5 | 1.9 | 12449.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.6 | 14.9 | 7.7 | 93.8 | 0.2 | 0.3 | 2323.8 | 1.8 | 12675.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 5.6 | 16.1 | 13.8 | 93.6 | 0.2 | 0.4 | 2781.7 | 16.8 | 14218.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.6 | 15.1 | 7.6 | 93.7 | 0.2 | 0.3 | 2365.9 | 0.9 | 12030.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 5.6 | 15.5 | 7.7 | 93.5 | 0.2 | 0.3 | 2431.3 | 1.8 | 12197.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 53.4 min for 4 rounds (12.5 min per round); a fixed 3 rounds would take 37.6 min (-12.5 min saved).
