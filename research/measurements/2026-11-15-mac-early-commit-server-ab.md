## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: 6 rounds as the GPU pick's run: a gain of about 0.8 % against a spread of about 1 % (decode tokens per prompt as in the targets, default 200)
- Duration: 25-75 min (2-6 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-early-commit.md):
  - decode-code: +0.5 to +1 %
  - decode-text: +0.5 to +1 %
  - decode@8000: +0.5 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_EARLY_COMMIT=2

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_EARLY_COMMIT': '2'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 168.1 / 168.8 / 169.6 / 168.5 / 164.7 / 172.5 | 170.8 / 169.2 / 169.5 / 170.6 / 170.7 / 170.4 | 168.7 (164.7-172.5) | 170.5 (169.2-170.8) | +0.90 % [-0.99, +2.80] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.8 / 24.8 / 24.8 / 25.1 / 25.2 / 24.8 | 25.0 / 24.9 / 24.9 / 25.3 / 25.0 / 25.0 | 24.8 (24.8-25.2) | 25.0 (24.9-25.3) | +0.50 % [-0.27, +1.26] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.4 / 38.4 / 38.4 / 39.0 / 39.0 / 38.4 | 38.7 / 38.7 / 38.7 / 38.7 / 38.7 / 38.7 | 38.4 (38.4-39.0) | 38.7 (38.7-38.7) | +0.26 % [-0.75, +1.27] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.8 / 26.7 / 26.7 / 26.3 / 26.7 / 26.7 | 27.0 / 27.0 / 27.0 / 27.0 / 26.9 / 27.0 | 26.7 (26.3-26.8) | 27.0 (26.9-27.0) | +1.25 % [+0.33, +2.18] | **B faster** after round 3 |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.63 % | +0.76 % | +0.78 % | +0.75 % | 749 |
| 2 | BA | +0.24 % | +0.69 % | +0.78 % | +1.12 % | 748 |
| 3 | AB | -0.05 % | +0.64 % | +0.78 % | +1.12 % | 748 |
| 4 | BA | +1.20 % | +0.51 % | -0.77 % | +2.66 % | 748 |
| 5 | AB | +3.62 % | -0.54 % | -0.77 % | +0.75 % | 746 |
| 6 | BA | -1.21 % | +0.92 % | +0.78 % | +1.12 % | 746 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.90 % | -1 to +1 % | within the expectation |
| decode@8000 | +0.50 % | +0.5 to +1 % | **OUTSIDE the expectation: finding** |
| decode-code | +0.26 % | +0.5 to +1 % | **OUTSIDE the expectation: finding** |
| decode-text | +1.25 % | +0.5 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.2 | 16.1 | 7.8 | 89.0 | 0.2 | 0.3 | 23242.6 | 0.7 | 13282.5 | 6 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.1 | 14.3 | 7.5 | 93.9 | 0.2 | 0.3 | 2486.3 | 1.1 | 11468.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.1 | 18.1 | 8.0 | 93.9 | 0.2 | 0.3 | 2657.5 | 14.3 | 13257.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.2 | 14.9 | 7.7 | 94.8 | 0.3 | 0.3 | 2274.6 | 1.1 | 13029.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.1 | 19.6 | 7.7 | 94.8 | 0.3 | 0.3 | 2652.4 | 6.5 | 12461.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.2 | 14.4 | 7.8 | 94.5 | 0.2 | 0.3 | 2376.2 | 0.8 | 12786.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.2 | 15.1 | 7.6 | 94.5 | 0.2 | 0.3 | 2476.5 | 0.9 | 12754.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 4.9 | 15.1 | 8.3 | 94.5 | 0.2 | 0.3 | 2636.0 | 0.6 | 12655.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.0 | 13.9 | 7.6 | 94.0 | 0.2 | 0.3 | 2372.0 | 1.0 | 12065.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 4.9 | 15.5 | 7.6 | 94.5 | 0.2 | 0.2 | 2137.6 | 0.8 | 12246.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 5.0 | 16.1 | 7.5 | 94.1 | 0.2 | 0.3 | 2320.6 | 3.2 | 11773.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.1 | 15.7 | 7.6 | 94.6 | 0.2 | 0.3 | 2389.9 | 1.4 | 12117.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.2 | 14.7 | 7.7 | 94.6 | 0.2 | 0.3 | 2396.2 | 1.0 | 13291.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.2 | 15.1 | 7.7 | 94.6 | 0.2 | 0.2 | 2352.8 | 0.7 | 12761.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 5.1 | 14.9 | 7.8 | 94.7 | 0.2 | 0.3 | 2324.8 | 1.0 | 11927.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.3 | 28.6 | 8.0 | 94.0 | 0.2 | 0.2 | 2600.0 | 2.1 | 13589.4 | 8 | 0.79 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 6.5 | 63.2 | 10.5 | 92.3 | 0.5 | 0.3 | 2677.6 | 1.2 | 15202.4 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 6.1 | 65.0 | 11.7 | 92.6 | 0.3 | 0.5 | 5131.4 | 80.9 | 21379.5 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 5.6 | 59.0 | 8.8 | 93.1 | 0.2 | 0.3 | 3484.8 | 21.3 | 10331.0 | 9 | 0.8 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 5.1 | 19.4 | 16.5 | 94.9 | 0.2 | 0.3 | 2595.7 | 250.4 | 15258.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 5.2 | 17.1 | 7.7 | 94.8 | 0.2 | 0.3 | 2297.6 | 4.2 | 12286.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 22 | B | 5.2 | 25.8 | 7.9 | 94.9 | 0.2 | 0.2 | 2609.9 | 8.8 | 12948.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 23 | B | 5.2 | 17.4 | 7.7 | 94.6 | 0.2 | 0.4 | 2336.2 | 20.8 | 13380.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 24 | A | 4.9 | 17.3 | 9.2 | 94.4 | 0.2 | 0.3 | 2502.8 | 10.0 | 12367.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 25 | A | 5.0 | 14.5 | 7.5 | 94.4 | 0.2 | 0.4 | 2261.8 | 1.6 | 12632.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 78.1 min for 6 rounds (12.5 min per round); a fixed 3 rounds would take 37.4 min (-37.4 min saved).
