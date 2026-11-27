## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set: the decode of the production path, 11 code and 3 text prompts (decode tokens per prompt as in the targets, default 200)
- Duration: 21-64 min (2-6 rounds of 10.7 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-12-mac-mma-pass-costs.md):
  - decode-code: +2 to +8 %
  - decode-text: +1 to +7 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-934-g061c2ab33c; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-934-g061c2ab33c; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_MMA=1

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_MMA': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 31.3 / 32.3 | 33.8 / 34.5 | 31.8 (31.3-32.3) | 34.1 (33.8-34.5) | +7.40 % [+6.06, +8.74] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 24.0 / 25.1 | 24.9 / 26.6 | 24.6 (24.0-25.1) | 25.8 (24.9-26.6) | +4.86 % [+2.95, +6.78] | **B faster** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +7.99 % | +3.75 % | 655 |
| 2 | BA | +6.81 % | +5.98 % | 633 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +7.40 % | +2 to +8 % | within the expectation |
| decode-text | +4.86 % | +1 to +7 % | within the expectation |


Runtime: 24.4 min for 2 rounds (10.7 min per round); a fixed 3 rounds would take 32.2 min (+10.7 min saved).

## Result (21:05 to 21:29)

Conditions: quiet check ok at 21:05:08; `tools/hw_report.py` 21:05: load 1.78 / 1.88 / 2.02, other programs 4.7 %
of all CPU threads, commit `061c2ab` (tree dirty only by this file); after (21:29:37): "the GPU is 96 % busy" (the
run's own tail), `hw_report` load 1.35 / 1.51 / 1.67, other programs 3.8 %. Raw:
[2026-11-12-mac-mma-server-ab.json](2026-11-12-mac-mma-server-ab.json) (ab.py's raw data). ab.py's `set` workload runs
at context 32768.

| Metric | A per round | B per round | B vs A, mean [95 % CI] | Expected | Decision |
|---|---|---|---|---|---|
| decode code (tok/s) | 31.3 / 32.3 | 33.8 / 34.5 | **+7.40 % [+6.06, +8.74]** | +2 to +8 % | B faster after round 2 |
| decode text (tok/s) | 24.0 / 25.1 | 24.9 / 26.6 | **+4.86 % [+2.95, +6.78]** | +1 to +7 % | B faster after round 2 |

Rounds: AB +7.99 / +3.75 %, BA +6.81 / +5.98 %. Both within the expectation, code at its upper end (the cheaper W ≥ 4
costs move the width choice up; not separated here).

## Decision

`TONA_METAL_MMA` default on (coordinator 2026-11-12, condition "A/B not slower" met): passes with a 4-token chunk use
the half MMA kernel for dense matrices from 2048 rows below the vocabulary; the LM head and W = 1 to 3 stay f32.
`TONA_METAL_MMA=0` or `TONA_DETERMINISTIC=1` bring the f32 path back; the widths check (check 2 of the Metal backend)
runs with `TONA_DETERMINISTIC=1`. Checks with the default on (DM): 39 of 39 top-1 (1.40e-6, width 1 unchanged);
widths with `TONA_DETERMINISTIC=1` 0 of 66 and 0 of 160 (2 slots) rows differ, with `TONA_METAL_MMA=0` 0 of 66,
default 66 of 66 (from the first 4-token pass on, the state differs); prefill cut invariance unchanged; MTP golden
241 of 256, median 5.84e-2 (5.83e-2 before: the head's catch-up passes of 4 entries take the MMA kernel too).
