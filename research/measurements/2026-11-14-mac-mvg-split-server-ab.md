## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set: the decode of the production path, 11 code and 3 text prompts (decode tokens per prompt as in the targets, default 200)
- Duration: 19-57 min (2-6 rounds of 9.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-mvg-split.md):
  - decode-code: +0 to +1.5 %
  - decode-text: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-985-g5ec2674b17-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-985-g5ec2674b17-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_MVG_SPLIT=4

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_MVG_SPLIT': '4'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 36.1 / 37.6 / 38.1 / 38.2 / 36.8 / 38.8 | 36.9 / 36.5 / 38.3 / 36.5 / 37.3 / 36.9 | 37.9 (36.1-38.8) | 36.9 (36.5-38.3) | -1.36 % [-4.82, +2.10] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 24.9 / 26.7 / 25.8 / 26.7 / 25.6 / 27.3 | 25.4 / 24.9 / 26.7 / 25.5 / 25.3 / 25.1 | 26.2 (24.9-27.3) | 25.4 (24.9-26.7) | -2.50 % [-7.70, +2.71] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +2.22 % | +2.01 % | 577 |
| 2 | BA | -2.93 % | -6.74 % | 578 |
| 3 | AB | +0.52 % | +3.49 % | 564 |
| 4 | BA | -4.45 % | -4.49 % | 563 |
| 5 | AB | +1.36 % | -1.17 % | 571 |
| 6 | BA | -4.90 % | -8.06 % | 567 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | -1.36 % | +0 to +1.5 % | **OUTSIDE the expectation: finding** |
| decode-text | -2.50 % | -0.5 to +1 % | **OUTSIDE the expectation: finding** |


Runtime: 59.4 min for 6 rounds (9.5 min per round); a fixed 3 rounds would take 28.5 min (-28.5 min saved).
