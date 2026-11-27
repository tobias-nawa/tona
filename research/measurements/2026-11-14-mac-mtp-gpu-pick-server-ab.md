## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: the set for decode, one 8000-token request with an answer to compare, as in the read-ahead run (decode tokens per prompt as in the targets, default 200)
- Duration: 25-50 min (2-4 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-mtp-gpu-pick.md):
  - decode-code: +0.5 to +1.5 %
  - decode-text: +0.5 to +1.5 %
  - decode@8000: +0.5 to +1.5 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1010-g519cc97108-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DRAFT_LOG=work/draft-a.log, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1010-g519cc97108-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DRAFT_LOG=work/draft-b.log, TONA_GPU_GAPS=500, TONA_METAL_GPU_PICK=1

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_DRAFT_LOG': 'work/draft-a.log'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_DRAFT_LOG': 'work/draft-b.log', 'TONA_METAL_GPU_PICK': '1'}. Alpha 0.05, margin 0.5 %, rounds 4 of at most 4, stop: 4 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 169.3 / 167.6 / 174.1 / 177.0 | 171.5 / 169.6 / 176.4 / 177.2 | 171.7 (167.6-177.0) | 174.0 (169.6-177.2) | +0.97 % [-0.00, +1.94] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.9 / 24.9 / 25.2 / 24.6 | 24.9 / 24.7 / 24.9 / 24.8 | 24.9 (24.6-25.2) | 24.9 (24.7-24.9) | -0.12 % [-1.47, +1.23] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.1 / 38.7 / 38.7 / 38.2 | 38.4 / 38.4 / 38.4 / 38.3 | 38.5 (38.1-38.7) | 38.4 (38.3-38.4) | -0.13 % [-1.27, +1.02] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.6 / 26.6 / 27.1 / 26.5 | 26.7 / 26.7 / 26.7 / 26.7 | 26.6 (26.5-27.1) | 26.7 (26.7-26.7) | +0.01 % [-1.36, +1.38] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.29 % | +0.10 % | +0.79 % | +0.38 % | 751 |
| 2 | BA | +1.15 % | -0.54 % | -0.78 % | +0.38 % | 752 |
| 3 | AB | +1.34 % | -1.17 % | -0.78 % | -1.48 % | 748 |
| 4 | BA | +0.10 % | +1.13 % | +0.26 % | +0.75 % | 745 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.97 % | -1 to +1 % | within the expectation |
| decode@8000 | -0.12 % | +0.5 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-code | -0.13 % | +0.5 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-text | +0.01 % | +0.5 to +1.5 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.7 | 16.3 | 8.6 | 88.8 | 0.3 | 0.3 | 23636.0 | 3.0 | 13294.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.5 | 15.9 | 8.3 | 93.6 | 0.2 | 0.3 | 2462.7 | 8.1 | 12514.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.7 | 16.3 | 7.8 | 93.5 | 0.2 | 0.3 | 2534.0 | 5.0 | 12727.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 4.9 | 14.6 | 7.5 | 94.5 | 0.2 | 0.2 | 2309.0 | 1.1 | 12329.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.1 | 14.9 | 7.8 | 94.3 | 0.2 | 0.3 | 2196.9 | 2.4 | 12613.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.1 | 16.4 | 7.5 | 94.2 | 0.2 | 0.3 | 2421.4 | 4.0 | 12951.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.1 | 22.0 | 7.8 | 94.1 | 0.2 | 0.3 | 2758.6 | 6.3 | 14101.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.8 | 20.1 | 8.0 | 93.5 | 0.2 | 0.2 | 2542.3 | 6.3 | 14298.8 | 7 | 0.12 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 6.8 | 58.6 | 10.2 | 92.1 | 0.5 | 0.3 | 3078.8 | 6.7 | 16061.8 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 6.6 | 58.0 | 8.6 | 91.6 | 0.2 | 0.3 | 4554.3 | 40.2 | 14449.5 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.0 | 53.6 | 7.9 | 93.9 | 0.2 | 0.2 | 2247.2 | 13.6 | 12597.9 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.1 | 32.1 | 7.8 | 94.4 | 0.2 | 0.4 | 2347.3 | 51.1 | 12764.5 | 9 | 0.15 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 4.9 | 16.3 | 7.7 | 94.9 | 0.2 | 0.3 | 1899.8 | 9.1 | 12410.6 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 4.8 | 15.4 | 7.5 | 95.1 | 0.2 | 0.3 | 1749.7 | 8.2 | 12527.3 | 10 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 4.7 | 16.5 | 7.5 | 95.0 | 0.2 | 0.4 | 1679.9 | 8.1 | 11134.8 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.3 | 16.5 | 7.7 | 94.5 | 0.2 | 0.3 | 1839.1 | 12.9 | 12326.5 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 5.3 | 15.7 | 7.9 | 94.5 | 0.2 | 0.3 | 1898.7 | 12.4 | 12508.8 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 53.2 min for 4 rounds (12.5 min per round); a fixed 3 rounds would take 37.5 min (-12.5 min saved).
