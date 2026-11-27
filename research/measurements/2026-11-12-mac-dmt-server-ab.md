## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `work/build/mtl` against B `work/build/mtl`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first; a discarded start of each arm before its measured one, every round.
- Why this size: bench-set: the decode of the production path, 11 code and 3 text prompts (decode tokens per prompt as in the targets, default 200)
- Duration: 21-62 min (2-6 rounds of 10.3 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: research/measurements/2026-11-12-mac-dmt-pass-costs.md):
  - decode-code: +2 to +5 %
  - decode-text: +2 to +5 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit stand-2026-10-25-950-g4d38665d5a; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0
  - B: build commit stand-2026-10-25-950-g4d38665d5a; OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0

## Result

A: `work/build/mtl` env -; B: `work/build/mtl` env -. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 32.9 / 34.1 / 33.0 / 34.0 / 33.8 / 34.5 | 35.3 / 35.7 / 34.8 / 34.5 / 35.7 / 34.9 | 33.9 (32.9-34.5) | 35.1 (34.5-35.7) | +4.28 % [+1.54, +7.03] | **B faster** after round 2 |
| decode-text: Decode tok/s (bench-set text) | 24.9 / 26.6 / 24.9 / 25.2 / 25.9 / 26.5 | 27.0 / 26.9 / 25.6 / 25.4 / 26.8 / 25.2 | 25.5 (24.9-26.6) | 26.2 (25.2-27.0) | +1.96 % [-2.85, +6.76] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +7.29 % | +8.43 % | 625 |
| 2 | BA | +4.69 % | +1.13 % | 607 |
| 3 | AB | +5.45 % | +2.81 % | 618 |
| 4 | BA | +1.47 % | +0.79 % | 620 |
| 5 | AB | +5.62 % | +3.47 % | 617 |
| 6 | BA | +1.16 % | -4.91 % | 612 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +4.28 % | +2 to +5 % | within the expectation |
| decode-text | +1.96 % | +2 to +5 % | **OUTSIDE the expectation: finding** |


Runtime: 64.3 min for 6 rounds (10.3 min per round); a fixed 3 rounds would take 30.8 min (-30.8 min saved).

## Result (22:19 to 23:23)

Conditions: quiet check ok at 22:18:51; `tools/hw_report.py` before: load 1.13 / 1.41 / 2.16, other programs 3.6 %
of all CPU threads, commit `4d38665`; after: quiet check 23:23:24 "the GPU is 96 % busy" (the run's own tail, the
same as in [2026-11-12-mac-mma-server-ab.md](2026-11-12-mac-mma-server-ab.md)), again at 23:23:44: ok; `hw_report`
load 0.91 / 0.98 / 1.14, other programs 4.8 %. Raw: [2026-11-12-mac-dmt-server-ab.json](2026-11-12-mac-dmt-server-ab.json).
A is DM, B is DM-T, both with defaults (half MMA on for W ≥ 4), the cut MTP head (164 experts) as drafter, context 32768.

| Metric | B vs A, mean [95 % CI] | Expected | Decision |
|---|---|---|---|
| decode code (tok/s) | **+4.28 % [+1.54, +7.03]** | +2 to +5 % | B faster after round 2 |
| decode text (tok/s) | +1.96 % [-2.85, +6.76] | +2 to +5 % | open after 6 rounds |

- Code: DM-T faster, within the expectation.
- Text: the mean lies just below the band, the interval covers the whole band and zero. Not a contradiction of the
  pass costs but undecided: three text prompts give round results from -4.91 to +8.43 %, and the BA rounds are lower
  than the AB rounds in both metrics (code AB +7.29 / +5.45 / +5.62, BA +4.69 / +1.47 / +1.16 %). That order effect
  also makes the code figure an average of two levels; not separated here.
- Acceptance unchanged: tokens per pass over all requests of the server logs 2.791 (A, DM) and 2.823 (B, DM-T), so
  the gain is the cheaper pass, not a different draft acceptance.

## Decision

DM-T is the faster Mac file for code decode (+4.3 %), text decode at least not slower within this resolution. Whether
the installer recommends it is a distribution decision for the maintainer; nothing in `installer/` changed.
