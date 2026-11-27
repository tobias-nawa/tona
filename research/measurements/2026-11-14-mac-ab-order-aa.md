## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set as in the A/B that showed the order effect: the same workload, so the same states (decode tokens per prompt as in the targets, default 200)
- Duration: 19-57 min (2-6 rounds of 9.6 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-14-mac-ab-order.md):
  - decode-code: -1 to +1 %
  - decode-text: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-988-gc60b97c022; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-988-gc60b97c022; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 35.9 / 36.9 / 36.9 / 36.6 / 38.2 / 36.5 | 36.3 / 36.5 / 36.9 / 37.1 / 38.2 / 38.2 | 36.8 (35.9-38.2) | 37.0 (36.3-38.2) | +1.01 % [-1.24, +3.26] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 25.0 / 25.3 / 25.6 / 26.5 / 26.6 / 25.5 | 24.9 / 24.9 / 25.7 / 25.5 / 26.4 / 26.1 | 25.6 (25.0-26.6) | 25.6 (24.9-26.4) | -0.63 % [-2.93, +1.67] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.11 % | -0.40 % | 585 |
| 2 | BA | -1.08 % | -1.58 % | 582 |
| 3 | AB | +0.00 % | +0.39 % | 578 |
| 4 | BA | +1.37 % | -3.77 % | 577 |
| 5 | AB | +0.00 % | -0.75 % | 555 |
| 6 | BA | +4.66 % | +2.35 % | 569 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +1.01 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.63 % | -1 to +1 % | within the expectation |


Runtime: 59.9 min for 6 rounds (9.6 min per round); a fixed 3 rounds would take 28.7 min (-28.7 min saved).
