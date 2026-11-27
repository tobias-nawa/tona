## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: probe sweep, 2 rounds minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-early-commit.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_EARLY_COMMIT=1
  - B: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_EARLY_COMMIT=4

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_EARLY_COMMIT': '1'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_EARLY_COMMIT': '4'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 168.6 / 170.9 | 169.4 / 169.5 | 169.7 (168.6-170.9) | 169.4 (169.4-169.5) | -0.19 % [-1.30, +0.91] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.0 / 25.4 | 25.0 / 25.0 | 25.2 (25.0-25.4) | 25.0 (25.0-25.0) | -0.85 % [-2.00, +0.30] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.7 / 39.2 | 38.6 / 38.6 | 39.0 (38.7-39.2) | 38.6 (38.6-38.6) | -0.89 % [-1.99, +0.20] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 27.6 | 27.0 / 26.9 | 27.3 (27.0-27.6) | 26.9 (26.9-27.0) | -1.27 % [-2.94, +0.40] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.46 % | -0.15 % | -0.26 % | +0.00 % | 746 |
| 2 | BA | -0.85 % | -1.56 % | -1.53 % | -2.54 % | 744 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -0.19 % | -1 to +1 % | within the expectation |
| decode@8000 | -0.85 % | -1 to +1 % | within the expectation |
| decode-code | -0.89 % | -1 to +1 % | within the expectation |
| decode-text | -1.27 % | -1 to +1 % | **OUTSIDE the expectation: finding** |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.6 | 16.0 | 8.4 | 93.9 | 0.2 | 0.3 | 3155.8 | 1.9 | 11440.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.2 | 14.7 | 7.5 | 94.6 | 0.2 | 0.2 | 2260.6 | 1.2 | 12961.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.2 | 15.1 | 7.8 | 94.7 | 0.2 | 0.3 | 2351.5 | 1.5 | 12357.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.0 | 15.2 | 7.5 | 94.8 | 0.3 | 0.3 | 2336.8 | 1.1 | 12257.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.1 | 15.4 | 7.6 | 94.5 | 0.2 | 0.4 | 2406.8 | 2.3 | 12732.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.2 | 14.9 | 7.5 | 94.7 | 0.2 | 0.3 | 2331.1 | 0.7 | 12644.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.1 | 14.5 | 7.8 | 94.6 | 0.2 | 0.3 | 2342.8 | 4.6 | 12555.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.1 | 14.8 | 7.8 | 94.8 | 0.2 | 0.3 | 2409.6 | 2.3 | 13050.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.5 | 64.7 | 9.9 | 94.5 | 0.3 | 0.3 | 2826.5 | 9.5 | 14426.4 | 7 | 0.86 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 28.0 min for 2 rounds (12.4 min per round); a fixed 3 rounds would take 37.3 min (+12.4 min saved).
