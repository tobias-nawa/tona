## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: A/A for the probe split of inputs; the set for decode, one 8000-token request with an answer to compare (decode tokens per prompt as in the targets, default 200)
- Duration: 25-25 min (2-2 rounds of 12.4 min, from round 1)
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
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 167.3 / 176.6 | 170.2 / 170.4 | 171.9 (167.3-176.6) | 170.3 (170.2-170.4) | -0.87 % [-4.03, +2.28] | open (undecided at the last round) |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 25.3 / 24.9 | 25.2 / 24.8 | 25.1 (24.9-25.3) | 25.0 (24.8-25.2) | -0.51 % [-1.38, +0.35] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 39.1 / 38.4 | 39.0 / 38.4 | 38.8 (38.4-39.1) | 38.7 (38.4-39.0) | -0.13 % [-0.95, +0.70] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.4 / 26.7 | 27.0 / 26.7 | 27.0 (26.7-27.4) | 26.9 (26.7-27.0) | -0.73 % [-1.90, +0.44] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +1.78 % | -0.26 % | -0.26 % | -1.46 % | 746 |
| 2 | BA | -3.53 % | -0.76 % | +0.00 % | +0.00 % | 746 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | -0.87 % | -2 to +2 % | within the expectation |
| decode@8000 | -0.51 % | -1 to +1 % | within the expectation |
| decode-code | -0.13 % | -1 to +1 % | within the expectation |
| decode-text | -0.73 % | -1 to +1 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.7 | 15.9 | 7.9 | 92.9 | 0.2 | 0.3 | 3488.5 | 1.4 | 13984.3 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.2 | 21.8 | 7.8 | 93.8 | 0.2 | 0.3 | 2691.6 | 3.0 | 13142.8 | 7 | 0.23 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.9 | 55.9 | 8.8 | 93.3 | 0.2 | 0.3 | 3675.8 | 10.1 | 19180.1 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.2 | 53.7 | 8.0 | 93.7 | 0.2 | 0.3 | 3159.1 | 7.9 | 7349.9 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.6 | 70.0 | 8.3 | 93.7 | 0.2 | 0.3 | 2514.5 | 26.5 | 14512.5 | 8 | 0.94 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |
| 6 | B | 5.1 | 18.2 | 11.2 | 94.3 | 0.2 | 0.3 | 2456.0 | 15.3 | 14437.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.0 | 14.3 | 7.5 | 94.2 | 0.2 | 0.3 | 2391.2 | 6.6 | 11681.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 4.8 | 14.4 | 7.5 | 95.1 | 0.2 | 0.3 | 1819.6 | 3.2 | 12235.0 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 4.8 | 22.2 | 7.9 | 94.9 | 0.2 | 0.3 | 2178.6 | 27.4 | 12985.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 28.0 min for 2 rounds (12.4 min per round); a fixed 3 rounds would take 37.3 min (+12.4 min saved).
