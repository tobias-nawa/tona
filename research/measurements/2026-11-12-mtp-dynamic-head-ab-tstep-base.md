## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona125base` against B `tona125base`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode with the MTP drafter; only the width policy constants change (decode tokens per prompt as in the targets, default 200)
- Duration: 5-15 min (2-6 rounds of 2.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: constants measured in S0n3/S0n7 (step 1.09, fill 0.15 ms against 1.15 and 1.3): a small shift of the chosen widths, noise of set about 0.5 %):
  - decode: -1 to +1 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-625-g1f17a936ef; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-625-g1f17a936ef; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_FILL_MS=0.15, TONA_MTP_STEP_MS=1.09

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_MTP_STEP_MS': '1.09', 'TONA_MTP_FILL_MS': '0.15'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 91.4 / 91.5 / 91.8 / 92.2 / 91.0 / 91.9 | 90.6 / 90.7 / 90.6 / 90.2 / 90.4 / 90.3 | 91.7 (91.0-92.2) | 90.5 (90.2-90.7) | -1.27 % [-2.08, -0.46] | **B slower** after round 3 |
| decode-text: Decode tok/s (bench-set text) | 67.9 / 67.5 / 67.5 / 67.9 / 67.1 / 67.8 | 67.5 / 67.7 / 67.2 / 67.6 / 67.1 / 67.6 | 67.7 (67.1-67.9) | 67.5 (67.1-67.7) | -0.25 % [-0.86, +0.37] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | -0.88 % | -0.59 % | 149 |
| 2 | BA | -0.87 % | +0.30 % | 150 |
| 3 | AB | -1.31 % | -0.44 % | 150 |
| 4 | BA | -2.17 % | -0.44 % | 150 |
| 5 | AB | -0.66 % | +0.00 % | 150 |
| 6 | BA | -1.74 % | -0.29 % | 150 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | -1.27 % | -1 to +1 % | **OUTSIDE the expectation: finding** |
| decode-text | -0.25 % | -1 to +1 % | within the expectation |


Runtime: 16.2 min for 6 rounds (2.5 min per round); a fixed 3 rounds would take 7.5 min (-7.5 min saved).
