## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl-pick2a` against B `work/build/mtl`; rounds 2 to 4, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: control of the library version (coordinator); the set for decode, one 8000-token request with an answer to compare (decode tokens per prompt as in the targets, default 200)
- Duration: 25-50 min (2-4 rounds of 12.6 min, from round 1)
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
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 166.0 / 167.8 / 161.2 / 170.8 | 168.7 / 167.2 / 169.6 / 170.5 | 166.9 (161.2-170.8) | 169.2 (167.2-170.5) | +1.57 % [-1.60, +4.74] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.8 / 25.1 / 25.0 / 24.9 | 24.8 / 24.7 / 24.7 / 24.7 | 24.9 (24.8-25.1) | 24.7 (24.7-24.8) | -0.87 % [-1.72, -0.01] | **B slower** after round 4 |
| decode-code: Decode tok/s (bench-set code) | 38.3 / 38.4 / 38.5 / 38.2 | 38.2 / 38.4 / 38.3 / 38.3 | 38.3 (38.2-38.5) | 38.3 (38.2-38.4) | -0.13 % [-0.92, +0.66] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.6 / 26.8 / 26.8 / 26.6 | 26.7 / 26.8 / 26.6 / 26.7 | 26.7 (26.6-26.8) | 26.7 (26.6-26.8) | +0.00 % [-0.92, +0.93] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.63 % | -0.33 % | -0.26 % | +0.38 % | 754 |
| 2 | BA | -0.39 % | -1.39 % | +0.00 % | +0.00 % | 754 |
| 3 | AB | +5.26 % | -0.96 % | -0.52 % | -0.75 % | 756 |
| 4 | BA | -0.20 % | -0.79 % | +0.26 % | +0.38 % | 752 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +1.57 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode@8000 | -0.87 % | -0.5 to +0.5 % | **OUTSIDE the expectation: finding** |
| decode-code | -0.13 % | -0.5 to +0.5 % | within the expectation |
| decode-text | +0.00 % | -0.5 to +0.5 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | 0, 0 / 0, 0 / 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.9 | 14.8 | 8.0 | 93.3 | 0.2 | 0.3 | 2887.3 | 2.2 | 13507.9 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.7 | 26.4 | 10.3 | 93.4 | 0.2 | 0.2 | 2787.5 | 4.6 | 14233.1 | 8 | 0.37 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.8 | 23.9 | 7.8 | 93.2 | 0.2 | 0.3 | 2533.1 | 0.9 | 13102.0 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.8 | 22.2 | 7.7 | 93.2 | 0.2 | 0.4 | 2644.3 | 2.2 | 13594.2 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.7 | 24.8 | 7.9 | 93.4 | 0.2 | 0.3 | 2607.1 | 10.7 | 13738.6 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.9 | 26.3 | 7.8 | 93.1 | 0.2 | 0.3 | 2645.4 | 2.6 | 13826.7 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.8 | 29.5 | 7.9 | 93.2 | 0.2 | 0.3 | 2714.1 | 2.1 | 13881.4 | 6 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 6.0 | 23.1 | 7.7 | 92.9 | 0.2 | 0.4 | 2710.6 | 0.5 | 13970.3 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 6.0 | 31.7 | 8.3 | 93.2 | 0.2 | 0.3 | 3299.9 | 4.9 | 15532.7 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 6.0 | 49.5 | 11.2 | 92.9 | 0.2 | 0.3 | 3769.7 | 15.2 | 19831.7 | 6 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 6.5 | 42.0 | 10.2 | 92.0 | 0.2 | 0.2 | 4951.9 | 63.0 | 21923.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 6.6 | 88.2 | 10.6 | 91.9 | 0.2 | 0.4 | 5701.1 | 296.9 | 23957.1 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.7 | 20.0 | 8.2 | 93.5 | 0.2 | 0.3 | 2609.7 | 42.1 | 12816.3 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 14 | B | 5.9 | 33.6 | 8.4 | 93.4 | 0.2 | 0.3 | 2918.8 | 28.6 | 13185.8 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 15 | B | 5.7 | 18.3 | 7.9 | 93.7 | 0.2 | 0.3 | 2504.9 | 60.9 | 13090.6 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 16 | A | 5.7 | 22.6 | 7.6 | 93.5 | 0.2 | 0.3 | 2398.8 | 9.3 | 13057.3 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 17 | A | 5.6 | 22.8 | 7.9 | 93.7 | 0.2 | 0.3 | 2271.1 | 3.8 | 12882.6 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 53.4 min for 4 rounds (12.6 min per round); a fixed 3 rounds would take 37.7 min (-12.6 min saved).
