## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode with the MTP drafter (docs/contribute/measurement.md); the change touches only the drafter head, so decode of the 14 bench prompts answers it (decode tokens per prompt as in the targets, default 200)
- Duration: 5-16 min (2-6 rounds of 2.7 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: fixed-depth runs F4/G7/P4 of 2026-11-12-mtp-dynamic-head.md: MTP 8.1 -> 7.4 ms of a 66 ms pass = +1.1 %, tokens per pass +0.9 % code, text -0.6 %):
  - decode: +0.5 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-548-gf9b1629719; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-552-gadb0ca3fa0-dirty; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_SCORER=128, TONA_MTP_SCORER_ROWS=512

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_MTP_SCORER': '128', 'TONA_MTP_SCORER_ROWS': '512'}. Alpha 0.05, margin 0.5 %, rounds 6 of at most 6, stop: 6 rounds (undecided metrics stay open). All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 71.1 / 87.3 / 87.3 / 87.3 / 87.1 / 87.7 | 79.6 / 85.3 / 87.2 / 86.3 / 87.3 / 87.6 | 87.3 (71.1-87.7) | 86.8 (79.6-87.6) | +1.42 % [-4.38, +7.22] | open (undecided at the last round) |
| decode-text: Decode tok/s (bench-set text) | 55.4 / 67.1 / 67.2 / 66.6 / 67.2 / 67.1 | 66.8 / 67.0 / 66.9 / 67.0 / 67.0 / 67.2 | 67.1 (55.4-67.2) | 67.0 (66.8-67.2) | +3.41 % [-5.88, +12.70] | open (undecided at the last round) |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +11.95 % | +20.58 % | 189 |
| 2 | BA | -2.29 % | -0.15 % | 157 |
| 3 | AB | -0.11 % | -0.45 % | 157 |
| 4 | BA | -1.15 % | +0.60 % | 157 |
| 5 | AB | +0.23 % | -0.30 % | 157 |
| 6 | BA | -0.11 % | +0.15 % | 156 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +1.42 % | +0.5 to +3 % | within the expectation |
| decode-text | +3.41 % | +0.5 to +3 % | **OUTSIDE the expectation: finding** |


Runtime: 17.5 min for 6 rounds (2.7 min per round); a fixed 3 rounds would take 8.1 min (-8.1 min saved).
