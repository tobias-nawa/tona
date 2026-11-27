## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 2, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: A/A, not a speed comparison: the starts supply the gap lines of the probe (decode tokens per prompt as in the targets, default 200)
- Duration: 18-18 min (2-2 rounds of 9.2 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-ab-order.md):
  - decode-code: -3 to +3 %
  - decode-text: -3 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1009-g320afeebb6-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500
  - B: build commit stand-2026-10-25-1009-g320afeebb6-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_GPU_GAPS=500

## Result

A: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}; B: `work/build/mtl` env {'TONA_GPU_GAPS': '500'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 2, stop: 2 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 38.6 / 38.1 | 38.7 / 38.2 | 38.4 (38.1-38.6) | 38.5 (38.2-38.7) | +0.26 % [-0.55, +1.07] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 26.7 | 27.1 / 26.7 | 26.9 (26.7-27.0) | 26.9 (26.7-27.1) | +0.19 % [-0.66, +1.03] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +0.26 % | +0.37 % | 550 |
| 2 | BA | +0.26 % | +0.00 % | 553 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +0.26 % | -3 to +3 % | within the expectation |
| decode-text | +0.19 % | -3 to +3 % | within the expectation |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.0 | 24.7 | 7.8 | 88.2 | 0.1 | 0.1 | 21574.7 | 4.2 | 3975.1 | 8 | 0.41 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.7 | 52.6 | 8.3 | 93.6 | 0.1 | 0.1 | 1164.9 | 3.2 | 3233.8 | 7 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 6.4 | 67.0 | 9.4 | 93.1 | 0.1 | 0.1 | 3003.0 | 55.9 | 8873.2 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.3 | 60.9 | 7.9 | 94.0 | 0.1 | 0.1 | 1498.5 | 18.6 | 1830.9 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 5.6 | 68.1 | 8.0 | 93.7 | 0.1 | 0.1 | 943.8 | 37.1 | 2626.3 | 9 | 0.72 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.4 | 21.9 | 7.9 | 94.2 | 0.1 | 0.1 | 305.8 | 5.0 | 2475.3 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 4.6 | 13.9 | 7.4 | 94.8 | 0.1 | 0.1 | 146.4 | 5.5 | 1055.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 4.9 | 18.8 | 12.2 | 94.5 | 0.1 | 0.1 | 879.6 | 290.2 | 4112.4 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 4.8 | 14.5 | 7.0 | 94.5 | 0.1 | 0.1 | 502.0 | 4.8 | 1820.5 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |


Runtime: 20.8 min for 2 rounds (9.2 min per round); a fixed 3 rounds would take 27.6 min (+9.2 min saved).
