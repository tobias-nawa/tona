## Plan (written before the run)

- Scope: workload set (bench-set at --ctx 32768); A `tona125base` against B `tona125`; rounds 2 to 6, alternating AB/BA, stop when the primary metrics (all) are decided (alpha 0.05, margin 0.5 %); one discarded warm-up start of A first.
- Why this size: set is the production decode with the MTP drafter (docs/contribute/measurement.md); the change touches only the drafter head, so decode of the 14 bench prompts answers it (decode tokens per prompt as in the targets, default 200)
- Duration: 5-15 min (2-6 rounds of 2.5 min, from round 1)
- Known noise: none for this workload (bench/noise.toml); at least 2 rounds.
- Expectation (B against A, source: fixed-depth runs F4/G7/P4 of 2026-11-12-mtp-dynamic-head.md: MTP 8.1 -> 7.4 ms of a 66 ms pass = +1.1 %, tokens per pass +0.9 % code, text -0.6 %):
  - decode: +0.5 to +3 %
- Active tooling (TONA_*, HSA_*, HIP_*, OMP_*, ROCR_* of each server):
  - A: build commit v0.0.2-625-g1f17a936ef; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin
  - B: build commit v0.0.2-628-gb3d881f089; HIP_PATH=<repo>/work/rocm-10.1/lib/python3.14/site-packages/_rocm_sdk_devel, OMP_NUM_THREADS=6, ROCR_VISIBLE_DEVICES=0, TONA_EXPERT_PROFILE=<worktree>/work/profile/experts-coder.bin, TONA_MTP_DYN=4096, TONA_MTP_SCORER=128, TONA_MTP_SCORER_ROWS=1024, TONA_MTP_VOCAB=<worktree>/work/mtp-vocab-125-md3-n24k.txt

## Result

A: `tona125base` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin'}; B: `tona125` env {'TONA_EXPERT_PROFILE': '<worktree>/work/profile/experts-coder.bin', 'TONA_MTP_SCORER': '128', 'TONA_MTP_SCORER_ROWS': '1024', 'TONA_MTP_DYN': '4096', 'TONA_MTP_VOCAB': '<worktree>/work/mtp-vocab-125-md3-n24k.txt'}. Alpha 0.05, margin 0.5 %, rounds 3 of at most 6, stop: all metrics decided after round 3. All rounds are listed; none was dropped.

| Metric | A, per round | B, per round | A median (range) | B median (range) | B vs A, mean [CI] | Decision |
|---|---|---|---|---|---|---|
| decode-code: Decode tok/s (bench-set code) | 90.8 / 91.4 / 91.0 | 92.1 / 92.1 / 91.9 | 91.0 (90.8-91.4) | 92.1 (91.9-92.1) | +1.06 % [+0.05, +2.07] | **B faster** after round 3 |
| decode-text: Decode tok/s (bench-set text) | 67.5 / 67.5 / 67.3 | 64.1 / 64.2 / 64.7 | 67.5 (67.3-67.5) | 64.2 (64.1-64.7) | -4.60 % [-5.82, -3.37] | **B slower** after round 2 |

| Round | Order | decode-code B vs A | decode-text B vs A | Seconds |
|---|---|---|---|---|
| 1 | AB | +1.43 % | -5.04 % | 148 |
| 2 | BA | +0.77 % | -4.89 % | 147 |
| 3 | AB | +0.99 % | -3.86 % | 147 |

Result against the plan:

| Metric | Measured B vs A | Expected | Verdict |
|---|---|---|---|
| decode-code | +1.06 % | +0.5 to +3 % | within the expectation |
| decode-text | -4.60 % | +0.5 to +3 % | **OUTSIDE the expectation: finding** |


Runtime: 8.6 min for 3 rounds (2.5 min per round); a fixed 3 rounds would take 7.4 min (+0.0 min saved).
