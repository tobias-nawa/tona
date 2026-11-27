# Mac: server A/B, MMA in 4-token chunks against one MMA dispatch for up to 8 tokens (#108)

M1 Max 32 GB, DM-T (the Mac's file) with the MTP head cut to 164 experts. A: defaults; B: `TONA_METAL_MMA_WIDE=1`,
which also selects the wide pass-cost table (2026-11-13-mac-mma-wide-pass-costs.md). Same build `bf4d8f9`. Decides
whether the wide path becomes the default (as for MMA: on if not slower).

Expectation (written before the run): passes of 5 to 8 tokens 13 to 19 ms cheaper (8 to 12 %), W1 to W4 unchanged;
the server picks wider passes more often with the cheaper costs. Bench-set passes average ~2.8 tokens, so only a part
of the passes gains: decode code +2 to +8 %, text −1 to +4 % (text drafts shorter, fewer wide passes).

## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set: the decode of the production path, 11 code and 3 text prompts (decode tokens per prompt as in the targets, default 200)
- Duration: 20-60 min (2-6 rounds of 10.1 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-13-mac-mma-wide-pass-costs.md):
  - decode-code: +2 to +8 %
  - decode-text: -1 to +4 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-960-gbf4d8f922e; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-960-gbf4d8f922e; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_METAL_MMA_WIDE=1

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env {'TONA_METAL_MMA_WIDE': '1'}. Alpha 0.05, margin 0.5 %, rounds 2 of at most 6, stop: all metrics decided after round 2. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 34.7 / 34.7 | 35.4 / 35.8 | 34.7 (34.7-34.7) | 35.6 (35.4-35.8) | +2.59 % [+1.27, +3.92] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 25.6 / 25.5 | 25.9 / 25.8 | 25.6 (25.5-25.6) | 25.9 (25.8-25.9) | +1.17 % [+0.15, +2.20] | **B faster** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +2.02 % | +1.17 % | 605 |
| 2 | BA | +3.17 % | +1.18 % | 602 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +2.59 % | +2 to +8 % | within the expectation |
| decode-text | +1.17 % | -1 to +4 % | within the expectation |


Runtime: 22.7 min for 2 rounds (10.1 min per round); a fixed 3 rounds would take 30.2 min (+10.1 min saved).

## Conditions and reading (00:33 to 00:55)

Quiet check 00:33:02 warning (other programs 8.1 % of all CPU threads; measured anyway), `hw_report` load 1.92 / 1.28 /
1.67, other programs 5.3 %; after (00:55:48) "the GPU is 95 % busy" (the run's tail) and load 2.67, 20 s later a load
warning only (3.24); `hw_report` load 3.14 / 2.75 / 2.03, other programs 7.0 %. Raw:
[2026-11-13-mac-mma-wide-server-ab.json](2026-11-13-mac-mma-wide-server-ab.json).

- Both metrics B faster after 2 rounds, within the expectation: decode code +2.59 % [+1.27, +3.92], text +1.17 %
  [+0.15, +2.20].
- The server takes wider passes: tokens per pass over all requests of the server logs 2.823 (A) and 3.074 (B). The
  drafts are as long as before; the cheaper W5 to W8 costs let the width choice keep more of them.

## Decision

`TONA_METAL_MMA_WIDE` on by default (`=0` restores the 4-token chunks; `TONA_METAL_MMA=0` and `TONA_DETERMINISTIC=1`
still give the f32 path). Rows of W8 are bit-equal to the checked 4-token kernel (2026-11-12-mac-w8-mma.md); W5 to W7
move their tokens past the fourth from f32 to the same MMA arithmetic, inside the maintainer's rule (half MMA only at
W ≥ 4, W1 f32).
