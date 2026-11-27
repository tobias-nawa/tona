## Plan (written before the run)

- Scope: workload set,server (8000:100 bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: identity run: answers and pick checks, not speed (decode tokens per prompt as in the targets, default 200)
- Duration: 28-28 min (2-2 rounds of 13.9 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-mtp-gpu-pick.md):
  - decode-code: -10 to +10 %
  - decode-text: -10 to +10 %
  - decode@8000: -10 to +10 %
  - prefill@8000: -10 to +10 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1022-g42e1beb752-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1
  - B: build commit stand-2026-10-25-1022-g42e1beb752-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_DETERMINISTIC=1, TONA_METAL_GPU_PICK=1, TONA_METAL_PICK_CHECK=1

## Result

A: `work/build/mtl` env {'TONA_DETERMINISTIC': '1'}; B: `work/build/mtl` env {'TONA_DETERMINISTIC': '1', 'TONA_METAL_GPU_PICK': '1', 'TONA_METAL_PICK_CHECK': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| prefill@8000: Prefill tok/s after 8000 tokens of prompt | 169.2 / 171.2 | 176.0 / 173.9 | 170.2 (169.2-171.2) | 174.9 (173.9-176.0) | +2.79 % [+1.16, +4.41] | **B faster** after round 2 |
| decode@8000: Decode tok/s after 8000 tokens of prompt | 23.4 / 23.6 | 23.4 / 23.4 | 23.5 (23.4-23.6) | 23.4 (23.4-23.4) | -0.38 % [-1.45, +0.69] | open (undecided at the last round) |
| decode-code: Decode tok/s (bench-set code) | 33.4 / 33.9 | 33.4 / 33.4 | 33.6 (33.4-33.9) | 33.4 (33.4-33.4) | -0.74 % [-1.91, +0.44] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 26.4 / 26.9 | 26.3 / 26.3 | 26.6 (26.4-26.9) | 26.3 (26.3-26.3) | -1.30 % [-2.64, +0.03] | open (undecided at the last round) |

| Round | Order | prefill@8000 B vs A | decode@8000 B vs A | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|---|---|
| 1 | AB | +4.01 % | +0.22 % | +0.00 % | -0.38 % | 838 |
| 2 | BA | +1.56 % | -0.98 % | -1.47 % | -2.23 % | 833 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| prefill@8000 | +2.79 % | -10 to +10 % | within the expectation |
| decode@8000 | -0.38 % | -10 to +10 % | within the expectation |
| decode-code | -0.74 % | -10 to +10 % | within the expectation |
| decode-text | -1.30 % | -10 to +10 % | within the expectation |

Expert cache per request (counts from /status, no statistics):

| Request | A hits / misses per round | B hits / misses per round | identical |
|---|---|---|---|
| 8000 | 0, 0 / 0, 0 | 0, 0 / 0, 0 | yes |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 5.5 | 20.3 | 8.0 | 93.9 | 0.2 | 0.3 | 2601.2 | 12.2 | 13229.7 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.2 | 15.1 | 8.1 | 94.2 | 0.2 | 0.3 | 2651.0 | 5.8 | 12592.6 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 5.2 | 14.8 | 7.8 | 94.2 | 0.2 | 0.3 | 2345.2 | 4.1 | 12329.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.1 | 16.8 | 7.6 | 94.4 | 0.2 | 0.3 | 2294.5 | 29.3 | 11980.6 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.0 | 13.8 | 7.5 | 94.8 | 0.2 | 0.3 | 2008.1 | 3.0 | 12609.6 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.0 | 14.9 | 7.6 | 94.8 | 0.2 | 0.3 | 1978.0 | 4.4 | 12523.2 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.0 | 14.5 | 7.7 | 94.6 | 0.2 | 0.3 | 2288.4 | 4.3 | 11965.8 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.0 | 15.5 | 7.7 | 94.8 | 0.2 | 0.3 | 2108.7 | 5.8 | 12715.5 | 9 | 0.04 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.8 | 53.4 | 8.2 | 93.9 | 0.2 | 0.2 | 2509.1 | 7.5 | 15074.5 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | 0.2 | 2.0 |

| Request | SHA A | SHA B | A stable | B stable | A = B | Reference of main |
|---|---|---|---|---|---|---|
| 8000 | 727ff658ae742fa7 | 727ff658ae742fa7 | yes | yes | yes | - |

Runtime: 31.3 min for 2 rounds (13.9 min per round); a fixed 3 rounds would take 41.8 min (+13.9 min saved).
