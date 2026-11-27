## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: A/A for the split of the rest of inputs (encoding, MTP inputs); the set for decode, one 8000-token request with an answer to compare (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-gpu-gaps.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
  - decode@8000: -1 to +1 %
  - prefill@8000: -2 to +2 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1025-ga3018cb4a2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1025-ga3018cb4a2-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 170.6 / 170.7 | 168.7 / 174.4 | 170.7 (170.6-170.7) | 171.5 (168.7-174.4) | +0.52 % [-1.53, +2.56] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 24.8 / 24.8 | 25.2 / 24.8 | 24.8 (24.8-24.8) | 25.0 (24.8-25.2) | +0.72 % [-0.55, +2.00] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 38.4 / 38.4 | 38.4 / 38.4 | 38.4 (38.4-38.4) | 38.4 (38.4-38.4) | +0.00 % [-0.81, +0.81] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.7 / 26.7 | 26.8 / 26.9 | 26.7 (26.7-26.7) | 26.9 (26.8-26.9) | +0.56 % [-0.28, +1.40] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | -1.12 % | +1.58 % | +0.00 % | +0.37 % | 750 |
| 2 | BA | +2.15 % | -0.13 % | +0.00 % | +0.75 % | 749 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +0.52 % | -2 to +2 % | within the expectation |
| decode@8000 | +0.72 % | -1 to +1 % | within the expectation |
| decode-code | +0.00 % | -1 to +1 % | within the expectation |
| decode-text | +0.56 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.5 | 14.1 | 7.8 | 93.1 | 0.2 | 0.3 | 2930.8 | 2.5 | 11849.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.1 | 15.3 | 7.6 | 93.8 | 0.2 | 0.4 | 2518.5 | 12.2 | 12396.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.1 | 13.9 | 8.4 | 94.4 | 0.2 | 0.2 | 2717.5 | 2.4 | 12996.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 4.9 | 14.8 | 7.5 | 94.3 | 0.2 | 0.4 | 2407.0 | 2.4 | 13130.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.0 | 15.7 | 7.7 | 93.9 | 0.2 | 0.4 | 2411.3 | 8.0 | 12708.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.0 | 15.9 | 7.5 | 93.9 | 0.2 | 0.3 | 2520.3 | 13.8 | 12479.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 4.8 | 38.1 | 8.8 | 94.7 | 0.3 | 0.3 | 2119.2 | 7.3 | 12568.3 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 4.9 | 14.8 | 7.5 | 94.5 | 0.2 | 0.3 | 2301.0 | 2.1 | 12265.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.0 | 14.5 | 7.8 | 94.2 | 0.2 | 0.3 | 2245.5 | 3.2 | 12516.4 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 28.1 min for 2 rounds (12.5 min per round); a fixed 3 rounds would take 37.5 min (+12.5 min saved).
