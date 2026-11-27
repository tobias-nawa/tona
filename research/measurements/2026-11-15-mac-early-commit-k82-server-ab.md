## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: probe sweep, 2 rounds minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.3 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-early-commit.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_EARLY_COMMIT=8
  - B: build commit stand-2026-10-25-1038-gc3a3f537e2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500, TONA_METAL_EARLY_COMMIT=2

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_EARLY_COMMIT': '8'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500', 'TONA_METAL_EARLY_COMMIT': '2'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 175.3 / 171.4 | 177.0 / 176.2 | 173.4 (171.4-175.3) | 176.6 (176.2-177.0) | +1.86 % [+0.53, +3.19] | **B faster** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.2 / 24.8 | 24.9 / 25.1 | 25.0 (24.8-25.2) | 25.0 (24.9-25.1) | +0.00 % [-1.59, +1.60] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 39.0 / 38.6 | 38.8 / 38.9 | 38.8 (38.6-39.0) | 38.8 (38.8-38.9) | +0.13 % [-0.97, +1.23] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 26.9 | 27.1 / 27.0 | 26.9 (26.9-27.0) | 27.1 (27.0-27.1) | +0.37 % [-0.44, +1.18] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +0.94 % | -1.19 % | -0.51 % | +0.37 % | 739 |
| 2 | BA | +2.78 % | +1.20 % | +0.78 % | +0.37 % | 743 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +1.86 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +0.00 % | -1 to +1 % | within the expectation |
| decode-code | +0.13 % | -1 to +1 % | within the expectation |
| decode-text | +0.37 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.7 | 59.4 | 8.7 | 92.4 | 0.2 | 0.4 | 4362.1 | 51.2 | 16034.3 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.5 | 52.1 | 7.7 | 94.9 | 0.2 | 0.3 | 2397.3 | 15.3 | 12091.2 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.4 | 52.4 | 8.1 | 95.1 | 0.2 | 0.3 | 2169.9 | 34.4 | 13779.1 | 9 | 0.6 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.3 | 16.9 | 7.6 | 94.7 | 0.2 | 0.3 | 2369.8 | 6.7 | 12146.2 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.0 | 16.4 | 7.6 | 95.5 | 0.3 | 0.3 | 1688.0 | 5.5 | 11552.7 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 4.9 | 18.7 | 7.6 | 95.5 | 0.2 | 0.3 | 1674.0 | 11.9 | 11139.0 | 10 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.2 | 64.5 | 10.9 | 95.4 | 0.3 | 0.3 | 2950.4 | 359.4 | 14136.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.4 | 15.8 | 9.3 | 93.9 | 0.2 | 0.3 | 3088.8 | 134.0 | 13787.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.1 | 14.5 | 7.6 | 94.8 | 0.2 | 0.4 | 2411.0 | 2.2 | 12880.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 27.8 min for 2 rounds (12.3 min per round); a fixed 3 rounds would take 37.0 min (+12.3 min saved).
