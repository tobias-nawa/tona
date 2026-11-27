## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: probe only, 2 rounds minimum (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-15-mac-early-commit.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1040-gaf228094be-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1040-gaf228094be-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 164.3 / 170.8 | 174.5 / 170.8 | 167.5 (164.3-170.8) | 172.7 (170.8-174.5) | +3.14 % [-0.53, +6.81] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.2 / 25.1 | 25.3 / 25.2 | 25.1 (25.1-25.2) | 25.2 (25.2-25.3) | +0.31 % [-0.50, +1.13] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 39.0 / 38.7 | 39.2 / 39.1 | 38.9 (38.7-39.0) | 39.2 (39.1-39.2) | +0.77 % [-0.09, +1.64] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.4 / 27.1 | 27.4 / 27.6 | 27.2 (27.1-27.4) | 27.5 (27.4-27.6) | +0.92 % [-0.41, +2.26] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +6.26 % | +0.34 % | +0.51 % | +0.00 % | 745 |
| 2 | BA | +0.03 % | +0.29 % | +1.03 % | +1.85 % | 743 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +3.14 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode@8000 | +0.31 % | -1 to +1 % | within the expectation |
| decode-code | +0.77 % | -1 to +1 % | within the expectation |
| decode-text | +0.92 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.1 | 15.2 | 7.8 | 90.1 | 0.2 | 0.3 | 23004.6 | 5.8 | 12865.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.3 | 14.7 | 7.6 | 94.1 | 0.2 | 0.3 | 2660.1 | 0.6 | 12727.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.8 | 43.1 | 8.3 | 93.7 | 0.3 | 0.3 | 3038.4 | 4.4 | 14315.1 | 7 | 0.77 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 6.5 | 51.6 | 9.3 | 93.0 | 0.3 | 0.3 | 4631.4 | 66.2 | 21494.3 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 4.9 | 57.3 | 8.0 | 95.0 | 0.2 | 0.3 | 2574.8 | 41.4 | 4459.9 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.5 | 52.2 | 8.1 | 94.5 | 0.2 | 0.3 | 2443.1 | 23.6 | 12487.8 | 9 | 0.55 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.6 | 24.5 | 17.5 | 94.6 | 0.2 | 0.3 | 2539.6 | 330.1 | 14134.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.1 | 17.1 | 7.7 | 94.4 | 0.3 | 0.3 | 2751.8 | 6.5 | 8008.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.3 | 28.1 | 8.2 | 94.6 | 0.4 | 0.3 | 2974.8 | 39.4 | 13926.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 28.1 min for 2 rounds (12.4 min per round); a fixed 3 rounds would take 37.2 min (+12.4 min saved).
