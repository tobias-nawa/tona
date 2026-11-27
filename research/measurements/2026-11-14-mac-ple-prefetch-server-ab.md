## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: as many starts as the copy A/A, to catch slow starts in A (decode tokens per prompt as in the targets, default 200)
- Duration: 19-57 min (2-6 rounds of 9.4 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-ple-copy.md):
  - decode-code: +0 to +4 %
  - decode-text: +0 to +4 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-1000-g454ee0e7d8-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_STATS=1
  - B: build commit stand-2026-10-25-1000-g454ee0e7d8-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_PLE_PREFETCH=1, TONA_PLE_STATS=1

## Result

A: `work/build/mtl` env {'TONA_PLE_STATS': '1'}; B: `work/build/mtl` env {'TONA_PLE_STATS': '1', 'TONA_PLE_PREFETCH': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 36.9 / 37.0 | 38.7 / 38.3 | 37.0 (36.9-37.0) | 38.5 (38.3-38.7) | +4.20 % [+2.77, +5.62] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 25.5 / 25.2 | 27.0 / 26.7 | 25.4 (25.2-25.5) | 26.9 (26.7-27.0) | +5.92 % [+4.89, +6.95] | **B faster** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +4.88 % | +5.88 % | 565 |
| 2 | BA | +3.51 % | +5.95 % | 567 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +4.20 % | +0 to +4 % | **OUTSIDE the expectation: finding** |
| decode-text | +5.92 % | +0 to +4 % | **OUTSIDE the expectation: finding** |

Load per server start (percent of one core or of the GPU's time; ours = server and client):

| Start | Arm | CPU ours | CPU system | CPU other | GPU ours | GPU system | GPU other | page-ins MiB | swap-ins MiB | compressed MiB | memory level min | backup | thermal warning | CPU limit min | on battery | power mode | idle before s | to ready s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A | 6.5 | 86.8 | 11.7 | 88.0 | 0.1 | 0.1 | 4209.6 | 4.0 | 15282.5 | 6 | 0.75 | 0.0 | - | 0.0 | 2 | - | 2.0 |
| 2 | A | 7.8 | 92.6 | 11.8 | 86.7 | 0.7 | 0.1 | 3813.8 | 6.9 | 11368.6 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 3 | A | 6.0 | 71.3 | 8.9 | 88.9 | 0.1 | 0.1 | 2865.9 | 30.9 | 4531.0 | 5 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 4 | B | 5.8 | 81.8 | 8.5 | 93.2 | 0.1 | 0.1 | 1289.8 | 18.8 | 2556.9 | 9 | 1.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 5 | B | 6.0 | 31.2 | 8.3 | 92.9 | 0.1 | 0.1 | 1188.3 | 12.1 | 3553.2 | 8 | 0.11 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 6 | B | 5.2 | 19.2 | 7.9 | 93.9 | 0.1 | 0.1 | 752.7 | 6.2 | 2172.2 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 7 | B | 5.0 | 22.5 | 17.6 | 94.2 | 0.1 | 0.1 | 714.0 | 288.9 | 5899.0 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 8 | A | 5.4 | 16.5 | 8.0 | 89.3 | 0.1 | 0.1 | 1424.4 | 3.8 | 2941.4 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |
| 9 | A | 5.3 | 55.6 | 12.2 | 90.0 | 0.1 | 0.1 | 1253.4 | 4.2 | 2143.8 | 7 | 0.0 | 0.0 | - | 0.0 | 2 | 0.1 | 2.0 |


Runtime: 21.3 min for 2 rounds (9.4 min per round); a fixed 3 rounds would take 28.3 min (+9.4 min saved).
