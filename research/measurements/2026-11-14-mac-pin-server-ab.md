## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set as in the A/A that showed the slow blocks (decode tokens per prompt as in the targets, default 200)
- Duration: 19-57 min (2-6 rounds of 9.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-pin.md):
  - decode-code: +0 to +5 %
  - decode-text: +0 to +5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-991-gec28d4461d-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-991-gec28d4461d-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_PIN=mlock

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_PIN': 'mlock'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 38.4 / 38.0 / 38.1 / 38.2 / 38.2 / 38.1 | 37.5 / 37.0 / 38.2 / 38.0 / 38.2 / 36.7 | 38.2 (38.0-38.4) | 37.8 (36.7-38.2) | -1.49 % [-3.33, +0.36] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 27.0 / 24.9 / 26.3 / 26.7 / 26.7 / 25.4 | 25.8 / 25.0 / 26.7 / 24.9 / 26.7 / 25.6 | 26.5 (24.9-27.0) | 25.7 (24.9-26.7) | -1.41 % [-5.14, +2.32] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -2.34 % | -4.44 % | 574 |
| 2 | BA | -2.63 % | +0.40 % | 580 |
| 3 | AB | +0.26 % | +1.52 % | 565 |
| 4 | BA | -0.52 % | -6.74 % | 570 |
| 5 | AB | +0.00 % | +0.00 % | 563 |
| 6 | BA | -3.67 % | +0.79 % | 584 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | -1.49 % | +0 to +5 % | **OUTSIDE the expectation: finding** |
| decode-text | -1.41 % | +0 to +5 % | **OUTSIDE the expectation: finding** |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.2 | 43.5 | 8.9 | 88.2 | 0.1 | 0.1 | 2465.6 | 34.6 | 7775.8 | 8 | 1.0 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 5.5 | 18.5 | 7.6 | 90.5 | 0.1 | 0.1 | 1104.1 | 8.8 | 2635.1 | 8 | 0.06 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 4.9 | 41.1 | 8.9 | 94.6 | 0.1 | 0.1 | 252.9 | 8.8 | 2161.8 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 6.2 | 26.4 | 21.0 | 90.6 | 0.1 | 0.1 | 1117.2 | 18.3 | 5420.0 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.3 | 19.1 | 8.3 | 90.6 | 0.1 | 0.1 | 995.1 | 16.5 | 3583.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 4.1 | 2.0 |
| 6 | B | 6.6 | 24.8 | 8.0 | 90.2 | 0.1 | 0.1 | 1073.8 | 3.3 | 3856.1 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 4.3 | 2.0 |
| 7 | B | 6.5 | 33.7 | 8.5 | 88.9 | 0.1 | 0.1 | 1501.5 | 10.3 | 4142.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 4.3 | 2.0 |
| 8 | A | 5.2 | 15.5 | 7.2 | 90.8 | 0.1 | 0.1 | 979.5 | 2.5 | 1543.6 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 3.7 | 2.0 |
| 9 | A | 5.0 | 16.2 | 9.7 | 92.0 | 0.1 | 0.1 | 815.2 | 7.4 | 3222.7 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 10 | A | 5.0 | 15.3 | 7.5 | 92.7 | 0.1 | 0.1 | 693.5 | 1.3 | 1683.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 11 | A | 4.7 | 13.3 | 7.2 | 94.1 | 0.1 | 0.1 | 219.1 | 1.2 | 952.9 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 12 | B | 5.1 | 13.7 | 7.0 | 94.8 | 0.1 | 0.1 | 143.6 | 1.2 | 985.7 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 13 | B | 5.1 | 13.6 | 6.9 | 94.8 | 0.1 | 0.1 | 42.9 | 0.7 | 1141.2 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 4.0 | 2.0 |
| 14 | B | 5.2 | 16.5 | 8.1 | 94.8 | 0.1 | 0.1 | 508.6 | 1.3 | 2164.6 | 8 | 0.0 | 0.0 | - | 0.0 | 2 | 3.7 | 2.0 |
| 15 | B | 5.6 | 21.1 | 7.9 | 91.8 | 0.1 | 0.1 | 5379.7 | 410.5 | 5548.3 | 3 | 0.0 | 0.0 | - | 0.0 | 2 | 4.0 | 2.0 |
| 16 | A | 5.0 | 14.8 | 7.5 | 91.8 | 0.1 | 0.1 | 928.7 | 45.7 | 1487.1 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 3.9 | 2.0 |
| 17 | A | 4.6 | 15.8 | 7.7 | 94.8 | 0.1 | 0.1 | 244.9 | 63.3 | 1279.9 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 18 | A | 4.6 | 14.8 | 7.1 | 94.5 | 0.1 | 0.1 | 180.0 | 13.9 | 749.8 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 19 | A | 4.6 | 14.7 | 7.0 | 94.8 | 0.1 | 0.1 | 90.9 | 4.8 | 582.6 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 20 | B | 5.1 | 13.4 | 7.2 | 94.8 | 0.1 | 0.1 | 90.2 | 8.2 | 715.4 | 9 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 21 | B | 5.1 | 13.4 | 7.1 | 94.8 | 0.1 | 0.1 | 1661.8 | 4.6 | 1718.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 4.0 | 2.0 |
| 22 | B | 6.5 | 38.0 | 9.1 | 88.7 | 0.1 | 0.1 | 2745.6 | 153.8 | 8128.2 | 4 | 0.47 | 0.0 | - | 0.0 | 2 | 4.2 | 2.0 |
| 23 | B | 7.8 | 70.4 | 12.8 | 87.3 | 0.6 | 0.2 | 3122.9 | 162.9 | 9455.3 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 4.1 | 2.0 |
| 24 | A | 5.9 | 67.1 | 8.4 | 88.9 | 0.1 | 0.1 | 2262.4 | 58.5 | 3046.4 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 4.3 | 2.0 |
| 25 | A | 5.8 | 64.7 | 8.3 | 90.8 | 0.1 | 0.1 | 1058.8 | 45.5 | 2786.7 | 9 | 0.7 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |


Runtime: 59.7 min for 6 rounds (9.5 min per round); a fixed 3 rounds would take 28.6 min (-28.6 min saved).
