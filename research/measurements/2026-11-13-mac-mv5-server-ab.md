## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set: the decode of the production path, 11 code and 3 text prompts (decode tokens per prompt as in the targets, default 200)
- Duration: 19-58 min (2-6 rounds of 9.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-13-mac-mv8.md):
  - decode-code: +0 to +2 %
  - decode-text: -0.5 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-972-gd280fe7e94-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-972-gd280fe7e94-dirty; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_MV5=1

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_MV5': '1'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 35.8 / 36.8 / 36.6 / 36.8 / 36.7 / 35.7 | 37.5 / 37.7 / 37.4 / 36.9 / 38.4 / 38.4 | 36.7 (35.7-36.8) | 37.6 (36.9-38.4) | +3.64 % [+0.79, +6.49] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 25.7 / 26.6 / 26.0 / 27.3 / 25.5 / 25.6 | 26.1 / 26.1 / 25.7 / 24.6 / 26.9 / 26.8 | 25.9 (25.5-27.3) | 26.1 (24.6-26.9) | -0.20 % [-6.39, +5.99] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +4.75 % | +1.56 % | 583 |
| 2 | BA | +2.45 % | -1.88 % | 573 |
| 3 | AB | +2.19 % | -1.15 % | 579 |
| 4 | BA | +0.27 % | -9.89 % | 584 |
| 5 | AB | +4.63 % | +5.49 % | 569 |
| 6 | BA | +7.56 % | +4.69 % | 571 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +3.64 % | +0 to +2 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.20 % | -0.5 to +1 % | within the expectation |


Runtime: 60.3 min for 6 rounds (9.6 min per round); a fixed 3 rounds would take 28.8 min (-28.8 min saved).
